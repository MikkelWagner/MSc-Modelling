"""Scalar interface between native Parallel EGO and the fixed-load HPP model.

No search code lives here. Generation, dispatch, reliability and economics are
in the existing with_hpp_cost assembly. Only the EGO score, rejected-evaluation
status and independent result checks are added. No previous sizing code is used.
"""

import json
import os
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from docplex.mp.model import Model

from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost import (
    hpp_model_simple_datacenter_with_hpp_cost,
)
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.costs.cost_simple_hpp_with_hpp_cost import hpp_cost_annualized
from hydesign.ems import ems_simple_datacenter as dc_ems
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch
from hydesign.wind.wind import get_rotor_d


SCORE_NAME = "hpp_sizing_score [MEUR/year]"
VALID_NAME = "dispatch_validated [-]"
CAPACITY_VARIABLES = ("Nwt", "solar_MW [MW AC]", "b_P [MW]", "b_E_h [h]")
VARIABLE_NAMES = (
    "clearance [m]", "sp [W/m2]", "p_rated [MW]", "Nwt",
    "wind_MW_per_km2 [MW/km2]", "solar_MW [MW AC]", "surface_tilt [deg]",
    "surface_azimuth [deg]", "DC_AC_ratio", "b_P [MW]", "b_E_h [h]",
)


def write_json_atomic(path, value):
    """Publish one complete record; interrupted temporary files are never read."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False,
                  default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def annual_cost_bound(config, variables, facility_load_max_mw=None):
    """Bound native costs over the box with fixed geometry and nonnegative rates.

    Wind variable OPEX uses continuous nameplate output for 8760 hours. Native
    cost classes evaluate the upper capacities; this is not a dispatch shortcut.
    """
    def maximum(name):
        var = variables[name]
        return var["limits"][1] if var["var_type"] == "design" else var["value"]

    def parameters(names):
        values = {name: config[name] for name in names}
        if any(not np.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("Cost-bound coefficients must be finite and nonnegative")
        return values

    rated = maximum("p_rated [MW]")
    diameter = get_rotor_d(rated * 1e6 / maximum("sp [W/m2]"))
    count = maximum("Nwt")
    wind = wpp_cost(**parameters((
        "wind_turbine_cost", "wind_civil_works_cost", "wind_fixed_onm_cost",
        "wind_variable_onm_cost", "d_ref", "hh_ref", "p_rated_ref",
    )), N_time=8760)
    cw, ow = wind.compute(
        count, diameter / 2 + maximum("clearance [m]"), diameter, rated,
        np.full(8760, count * rated * max(1.0, config["wpp_efficiency"])),
    )
    pv = pvp_cost(**parameters((
        "solar_PV_cost", "solar_hardware_installation_cost",
        "solar_inverter_cost", "solar_fixed_onm_cost",
    )))
    cs, os = pv.compute(maximum("solar_MW [MW AC]"), maximum("DC_AC_ratio"))
    battery = battery_cost(**parameters((
        "battery_energy_cost", "battery_power_cost",
        "battery_BOP_installation_commissioning_cost", "battery_control_system_cost",
        "battery_energy_onm_cost",
    )), life_y=1, battery_price_reduction_per_year=0)
    cb, ob = battery.compute(
        maximum("b_P [MW]") * maximum("b_E_h [h]"), maximum("b_P [MW]"), np.ones(8760),
    )
    annual = hpp_cost_annualized(
        config["wind_WACC"], config["solar_WACC"], config["battery_WACC"],
        config["hpp_economic_lifetime_years"],
        battery_replacement_year=config["battery_replacement_year"],
        battery_replacement_cost_fraction=config["battery_replacement_cost_fraction"],
    )
    facility_max = config["facility_load_mw"] if facility_load_max_mw is None else facility_load_max_mw
    if not np.isfinite(facility_max) or facility_max < 0:
        raise ValueError("Invalid maximum facility demand for shared-cost bound")
    shared = shared_cost(**parameters(("hpp_BOS_soft_cost", "hpp_grid_connection_cost", "land_cost")))
    csh, osh, *_ = shared.compute(
        facility_max + maximum("b_P [MW]"),
        count * rated / maximum("wind_MW_per_km2 [MW/km2]"),
        maximum("solar_MW [MW AC]") * config["land_use_per_solar_MW"],
    )
    bound = float(annual.compute(cw, ow, cs, os, cb, ob, csh, osh)[-1])
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("Invalid annual HPP cost bound")
    return bound


def validate_result(model, design):
    """Independently check a completed physical-model evaluation, in original units."""
    cfg = model.sim_pars
    power, duration = design[-2:]
    residuals = validate_dispatch(
        model.hourly, model.soc_mwh, model.solver_status, power, power * duration,
        cfg["battery_depth_of_discharge"], cfg["battery_charge_efficiency"],
        cfg["battery_initial_soc"], cfg["pue"],
    )
    supplied = np.asarray(model.hourly["served_facility_mw"])
    if supplied.shape != (8760,):
        raise ValueError("Final validation requires 8760 hourly results")
    np.testing.assert_allclose(
        model.hourly["facility_load_mw"], cfg["facility_load_mw"], atol=1e-8, rtol=0,
    )
    metrics = firm_load_metrics(
        np.maximum(cfg["facility_load_mw"] - supplied, 0.0), epsilon_mw=cfg["epsilon_mw"],
    )
    allowed = int(np.floor((1 - cfg["reliability_target"]) * 8760 + 1e-9))
    if metrics["LOLH"] > allowed or metrics["firm_reliability"] < cfg["reliability_target"] - 1e-12:
        raise ValueError("Final hourly reliability is below the requested target")
    for name in ("CAPEX_hpp", "annualized_cost_hpp"):
        cost = model.economics[name]
        if not np.isfinite(cost) or cost < 0:
            raise ValueError(f"Invalid {name}")
    return {"metrics": metrics, "max_failed_hours": allowed, "physical_residuals": residuals}


def rejected_status(observations):
    """Classify structured solver codes without interpreting exception messages."""
    if not observations:
        return "MODEL_ERROR"
    last = observations[-1]
    code = last["status_code"]
    if code in (3, 103) and not last["has_solution"]:
        return "PROVEN_INFEASIBLE"
    if code in (11, 107, 108):
        return "TIME_LIMIT_WITH_INCUMBENT" if last["has_solution"] else "TIME_LIMIT_UNRESOLVED"
    if code in (1, 101, 102) and last["has_solution"]:
        return "VALIDATION_FAILURE"
    return "SOLVER_UNRESOLVED_OR_ERROR"


class hpp_model_simple_datacenter_hpp_sizing_base(hpp_model_simple_datacenter_with_hpp_cost):
    """Native model interface: maximize -annual_cost/1e6, reject failed evaluations.

    A process-local Model observer records CPLEX details, including failed solves.
    It changes no parameters or equations, and restores the EMS binding on exit.
    Native EGO workers evaluate one model at a time. Do not use this adapter from
    concurrent threads in one process. Parallel EGO itself is never patched.
    """

    def __init__(self, sim_pars_fn, variables, baseline_log_dir=None, **kwargs):
        super().__init__(sim_pars_fn=sim_pars_fn, **kwargs)
        if set(variables) != set(self.list_vars):
            raise ValueError("Variable names must match the physical model's list_vars")
        for name, var in variables.items():
            if var["var_type"] not in ("fixed", "design"):
                raise ValueError(f"Invalid var_type: {name}")
            if var["var_type"] == "design":
                if name not in CAPACITY_VARIABLES:
                    raise ValueError(f"Geometry/technology must remain fixed: {name}")
                lo, hi = var["limits"]
                if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
                    raise ValueError(f"Invalid bounds: {name}")
                if var["types"] != ("int" if name == "Nwt" else "float"):
                    raise ValueError(f"Incorrect native variable type: {name}")
                if name == "Nwt" and (lo != round(lo) or hi != round(hi)):
                    raise ValueError("Turbine-count bounds must be integers")
            elif not np.isfinite(var["value"]) or var["value"] < 0:
                raise ValueError(f"Invalid fixed value: {name}")
        self.variables = variables
        self.cost_upper_bound = annual_cost_bound(self.sim_pars, variables)
        self.rejection_score = -(self.cost_upper_bound + max(1.0, .01 * self.cost_upper_bound)) / 1e6
        self.list_out_vars += [VALID_NAME, SCORE_NAME]
        self.baseline_log_dir = None if baseline_log_dir is None else Path(baseline_log_dir)
        self.candidate_status = None
        self.candidate_error = None
        self.solve_observations = []

    def evaluate(self, *design):
        start = time.perf_counter()
        self.candidate_error = None
        self.solve_observations = []
        adapter = self
        record_path = None
        if self.baseline_log_dir is not None:
            record_path = self.baseline_log_dir / f"{uuid4().hex}_hpp_sizing_base.json"
            record = {"design": list(map(float, design)), "status": "RUNNING",
                      "annualized_cost_hpp": None, "started_at": time.time(),
                      "pid": os.getpid(), "error": None}
            write_json_atomic(record_path, record)

        class ObservedModel(Model):
            def solve(mdl, *args, **kwargs):
                solution = None
                try:
                    solution = super().solve(*args, **kwargs)
                    return solution
                finally:
                    details = mdl.solve_details
                    adapter.solve_observations.append({
                        "status_code": getattr(details, "status_code", None),
                        "status": getattr(details, "status", None),
                        "seconds": getattr(details, "time", None),
                        "has_solution": solution is not None,
                    })

        try:
            design = list(map(float, design))
            if len(design) != len(self.list_vars) or not np.isfinite(design).all():
                raise ValueError("Expected a complete finite physical design vector")
            # Remove inverse-scaling roundoff only; native EGO chooses integers.
            if abs(design[3] - round(design[3])) > 1e-9:
                raise ValueError("Native EGO supplied a noninteger turbine count")
            design[3] = float(round(design[3]))
            for name, value in zip(self.list_vars, design):
                var = self.variables[name]
                lo, hi = var["limits"] if var["var_type"] == "design" else (var["value"], var["value"])
                if value < lo - 1e-9 or value > hi + 1e-9:
                    raise ValueError(f"Design outside configured domain: {name}")
            with patch.object(dc_ems, "Model", ObservedModel):
                outputs = super().evaluate(*design)
            self.final_checks = validate_result(self, design)
            cost = self.economics["annualized_cost_hpp"]
            if cost > self.cost_upper_bound + 1e-6 or not np.isfinite(outputs).all():
                raise ValueError("Nonfinite outputs or HPP cost exceeds the configured bound")
            self.candidate_status = "FEASIBLE"
            self.outputs = np.append(outputs, [1.0, -cost / 1e6])
        except Exception as exc:
            self.candidate_status = rejected_status(self.solve_observations)
            self.candidate_error = f"{type(exc).__name__}: {exc}"
            # Only the last element is optimized. Other entries are placeholders,
            # explicitly marked dispatch_validated=0, never reported as real costs.
            self.outputs = np.zeros(len(self.list_out_vars))
            self.outputs[-1] = self.rejection_score
            self.hourly = self.soc_mwh = self.metrics = self.economics = None
            self.validation = self.final_checks = self.solver_status = None

        if record_path is not None:
            record.update({
                "design": design, "status": self.candidate_status, "error": self.candidate_error,
                "score_meur_per_year": float(self.outputs[-1]),
                "annualized_cost_hpp": None if self.economics is None else self.economics["annualized_cost_hpp"],
                "solver": self.solve_observations, "seconds": time.perf_counter() - start,
                "completed_at": time.time(),
                "validation": self.final_checks,
                "metrics": None if self.final_checks is None else self.final_checks["metrics"],
                "economics": self.economics,
            })
            # Each worker owns one record. Rename replaces RUNNING atomically.
            write_json_atomic(record_path, record)
        return self.outputs

    def print_design(self, x_opt=None, outs=None):
        if self.candidate_status != "FEASIBLE":
            print(f"No validated feasible design: {self.candidate_status}. {self.candidate_error}")
            print("The rejection score and placeholder outputs are not physical/economic results.")
        else:
            super().print_design(x_opt, outs)
