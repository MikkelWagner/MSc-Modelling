"""Joint native-EGO interface, composing the unchanged fixed-load adapter.

Only installed IT capacity, candidate load synchronization and the outer score
are new. The inner adapter retains all dispatch, solver and physical checks.
"""
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import (
    hpp_model_simple_datacenter_hpp_sizing_base, VARIABLE_NAMES as HPP_VARIABLE_NAMES,
    write_json_atomic,
    annual_cost_bound,
)
from hydesign.costs.cost_simple_datacenter import datacenter_cost_annualized

IT_NAME = "it_capacity_mw [MW_IT]"
VARIABLE_NAMES = (*HPP_VARIABLE_NAMES, IT_NAME)
SCORE_NAME = "joint_system_value_score [MEUR/year]"
DC_COST_PARAMETERS = (
    "datacenter_facility_capex_per_mw_it", "datacenter_it_capex_per_mw_it",
    "datacenter_fixed_opex_per_mw_it_year", "datacenter_discount_rate",
    "datacenter_facility_lifetime_years", "datacenter_it_lifetime_years",
)


def validate_it_variable(variable):
    """Positive installed IT capacity, with native continuous design handling."""
    if variable["var_type"] == "design":
        lo, hi = variable["limits"]
        if variable["types"] != "float" or not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
            raise ValueError("IT capacity needs positive increasing bounds and types='float'")
    elif variable["var_type"] == "fixed":
        if not np.isfinite(variable["value"]) or variable["value"] <= 0:
            raise ValueError("Fixed IT capacity must be positive and finite")
    else:
        raise ValueError("IT capacity var_type must be fixed or design")


def system_value_score(economics, total_cost_bound):
    """Positive affine score preserves value order, even for negative values.

    Nonnegative revenue and the conservative annual cost bound imply V >= -B.
    score = 1 + (V+B)/1e6 >= 1; every rejection receives -1.
    """
    revenue = economics["Revenue_dc"]
    cost = economics["annualized_cost_dc"] + economics["annualized_cost_hpp"]
    value = economics["net_annual_value_system"]
    if not np.isfinite([revenue, cost, value, total_cost_bound]).all():
        raise ValueError("Nonfinite system economics")
    if revenue < -1e-6 or cost < 0 or cost > total_cost_bound + 1e-6:
        raise ValueError("System economics outside the conservative score bound")
    if not np.isclose(value, revenue - cost, rtol=1e-10, atol=1e-6):
        raise ValueError("Inconsistent net annual system value")
    return 1.0 + (value + total_cost_bound) / 1e6


class hpp_model_simple_datacenter_joint_hpp_sizing_base:
    """Expose twelve inputs to EGO, passing only eleven HPP inputs to physics."""

    def __init__(self, sim_pars_fn, variables, baseline_log_dir=None, **kwargs):
        if set(variables) != set(VARIABLE_NAMES):
            raise ValueError("Expected the eleven HPP inputs plus installed IT capacity")
        validate_it_variable(variables[IT_NAME])
        self.variables = variables
        self.hpp = hpp_model_simple_datacenter_hpp_sizing_base(
            sim_pars_fn=sim_pars_fn,
            variables={name: variables[name] for name in HPP_VARIABLE_NAMES}, **kwargs,
        )
        self.prob, self.sim_pars = self.hpp.prob, self.hpp.sim_pars
        self.input_ts_fn, self.altitude = self.hpp.input_ts_fn, self.hpp.altitude
        self.list_vars = list(VARIABLE_NAMES)
        self.list_out_vars = self.hpp.list_out_vars[:-1] + [SCORE_NAME]
        self.pue = float(self.sim_pars["pue"])
        if not np.isfinite(self.pue) or self.pue < 1:
            raise ValueError("PUE must be finite and at least one")
        var = variables[IT_NAME]
        maximum_it = var["limits"][1] if var["var_type"] == "design" else var["value"]
        # Both the inner HPP-cost check and joint score must cover the FULL
        # demand domain, including facility demand plus maximum rated b_P.
        self.hpp.cost_upper_bound = annual_cost_bound(
            self.sim_pars, {name: variables[name] for name in HPP_VARIABLE_NAMES},
            facility_load_max_mw=maximum_it * self.pue)
        self.hpp.rejection_score = -(self.hpp.cost_upper_bound
                                    + max(1., .01*self.hpp.cost_upper_bound)) / 1e6
        dc_cost = datacenter_cost_annualized(**{k: self.sim_pars[k] for k in DC_COST_PARAMETERS})
        self.total_cost_bound = self.hpp.cost_upper_bound + float(dc_cost.compute(maximum_it)[-1])
        self.rejection_score = -1.0
        self.baseline_log_dir = None if baseline_log_dir is None else Path(baseline_log_dir)
        self.candidate_status = None

    def evaluate(self, *design):
        start = time.perf_counter()
        design = list(map(float, design))
        record = {"design": design, "status": "RUNNING", "started_at": time.time(),
                  "pid": os.getpid(), "annualized_cost_hpp": None, "error": None}
        path = None if self.baseline_log_dir is None else (
            self.baseline_log_dir / f"{uuid4().hex}_joint_hpp_sizing_base.json")
        if path is not None:
            write_json_atomic(path, record)
        self.candidate_error = None
        self.solve_observations = []
        self.hourly = self.soc_mwh = self.economics = self.metrics = None
        self.validation = self.final_checks = self.solver_status = None
        self.outputs = np.zeros(len(self.list_out_vars))
        self.outputs[-1] = self.rejection_score
        self.candidate_status = "MODEL_ERROR"
        try:
            if len(design) != len(VARIABLE_NAMES) or not np.isfinite(design).all():
                raise ValueError("Expected twelve finite joint design inputs")
            it_capacity = design[-1]
            var = self.variables[IT_NAME]
            lo, hi = var["limits"] if var["var_type"] == "design" else (var["value"], var["value"])
            if it_capacity <= 0 or not lo - 1e-9 <= it_capacity <= hi + 1e-9:
                raise ValueError("IT capacity outside configured bounds")
            # Both independent validation metadata and OpenMDAO see this load.
            facility_load = it_capacity * self.pue
            self.sim_pars["facility_load_mw"] = facility_load
            self.prob.set_val("facility_load_mw", facility_load)
            self.prob.set_val("pue", self.pue)
            outputs = self.hpp.evaluate(*design[:-1])
            self.candidate_status = self.hpp.candidate_status
            self.candidate_error = self.hpp.candidate_error
            self.solve_observations = self.hpp.solve_observations
            if self.candidate_status == "FEASIBLE":
                score = system_value_score(self.hpp.economics, self.total_cost_bound)
                for name in ("hourly", "soc_mwh", "economics", "metrics", "validation",
                             "final_checks", "solver_status"):
                    setattr(self, name, getattr(self.hpp, name))
                self.outputs = np.append(outputs[:-1], score)
        except Exception as exc:
            self.candidate_status = "VALIDATION_FAILURE" if self.solve_observations else "MODEL_ERROR"
            self.candidate_error = f"{type(exc).__name__}: {exc}"
        self.inputs = design
        record.update(
            status=self.candidate_status, error=self.candidate_error,
            score_meur_per_year=float(self.outputs[-1]), score_cost_bound_eur=self.total_cost_bound,
            it_capacity_mw=design[-1] if len(design) == len(VARIABLE_NAMES) else None,
            facility_load_mw=self.sim_pars["facility_load_mw"],
            annualized_cost_hpp=None if self.economics is None else self.economics["annualized_cost_hpp"],
            net_annual_value_system=None if self.economics is None else self.economics["net_annual_value_system"],
            solver=self.solve_observations, validation=self.final_checks,
            metrics=None if self.final_checks is None else self.final_checks["metrics"],
            economics=self.economics, seconds=time.perf_counter()-start, completed_at=time.time(),
        )
        if path is not None:
            write_json_atomic(path, record)
        return self.outputs

    def print_design(self, x_opt=None, outs=None):
        if self.candidate_status != "FEASIBLE":
            print(f"No validated feasible design: {self.candidate_status}. {self.candidate_error}")
            return
        for name, value in zip(self.list_vars, self.inputs if x_opt is None else x_opt):
            print(f"{name}: {value:g}")
        print("Net annual system value [EUR/year]:", self.economics["net_annual_value_system"])
        print("Firm reliability:", self.metrics["firm_reliability"])

