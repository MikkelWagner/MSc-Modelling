"""Native EGO scalar adapter: target-free EMS, outer reliability, retained dispatch."""
import hashlib
import os
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4
import numpy as np
from docplex.mp.model import Model
from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost_max_reliability import (
    hpp_model_simple_datacenter_with_hpp_cost_max_reliability)
from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import (
    CAPACITY_VARIABLES, VARIABLE_NAMES, VALID_NAME, LCOE_OBJECTIVE, COST_OBJECTIVE,
    annual_cost_bound, objective_bound, objective_name, objective_value,
    rejected_status, write_json_atomic)
from hydesign.ems import ems_simple_datacenter_max_reliability as dc_ems
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch


def validate_result(model, design):
    """Check saved/evaluated physics and economics; below-target is not an error."""
    cfg = model.sim_pars
    power, duration = design[-2:]
    residuals = validate_dispatch(model.hourly, model.soc_mwh, model.solver_status,
        power, power*duration, cfg["battery_depth_of_discharge"],
        cfg["battery_charge_efficiency"], cfg["battery_initial_soc"], cfg["pue"])
    supplied = np.asarray(model.hourly["served_facility_mw"])
    if supplied.shape != (8760,):
        raise ValueError("Expected 8760 hourly results")
    np.testing.assert_allclose(model.hourly["facility_load_mw"], cfg["facility_load_mw"], atol=1e-8, rtol=0)
    metrics = firm_load_metrics(np.maximum(cfg["facility_load_mw"]-supplied,0), epsilon_mw=cfg["epsilon_mw"])
    allowed = int(np.floor((1-cfg["reliability_target"])*8760+1e-9))
    feasible = metrics["LOLH"] <= allowed and metrics["firm_reliability"] >= cfg["reliability_target"]-1e-12
    for key in ("CAPEX_hpp", "annualized_cost_hpp"):
        if not np.isfinite(model.economics[key]) or model.economics[key] < 0:
            raise ValueError(f"Invalid {key}")
    np.testing.assert_allclose(model.economics["served_facility_energy_mwh"], supplied.sum(), rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(supplied.sum()+metrics["EENS"], cfg["facility_load_mw"]*8760, rtol=1e-10, atol=1e-5)
    np.testing.assert_allclose(model.hourly["unserved_facility_mw"],
                               cfg["facility_load_mw"]-supplied, atol=1e-5, rtol=0)
    return dict(metrics=metrics, max_failed_hours=allowed, outer_feasible=bool(feasible),
                physical_residuals=residuals, certificate=model.solver_status.get("certificate"))


class hpp_model_simple_datacenter_hpp_sizing_base_max_reliability(
        hpp_model_simple_datacenter_with_hpp_cost_max_reliability):
    """Maximize -LCoE, or return a finite rejection score without dropping physics."""

    def __init__(self, sim_pars_fn, variables, baseline_log_dir=None,
                 hpp_sizing_objective=LCOE_OBJECTIVE, **kwargs):
        if hpp_sizing_objective != LCOE_OBJECTIVE:
            raise ValueError("The max-reliability sizing workflow minimizes HPP LCoE")
        self.objective = hpp_sizing_objective
        score_name = objective_name(self.objective)
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
        self.objective_upper_bound = objective_bound(self.sim_pars, self.cost_upper_bound, self.objective)
        self.objective_scale = 1.0 if self.objective == LCOE_OBJECTIVE else 1e6
        self.rejection_score = -(self.objective_upper_bound + max(1.0, .01 * self.objective_upper_bound)) / self.objective_scale
        self.list_out_vars += [VALID_NAME, score_name]
        self.baseline_log_dir = None if baseline_log_dir is None else Path(baseline_log_dir)
        self.candidate_status = None
        self.candidate_error = None
        self.solve_observations = []

    def evaluate(self, *design):
        start = time.perf_counter()
        self.candidate_error = None
        self.solve_observations = []
        self.hourly = self.soc_mwh = self.metrics = self.economics = None
        self.final_checks = self.solver_status = None
        adapter = self
        record_path = None
        if self.baseline_log_dir is not None:
            self.baseline_log_dir.mkdir(parents=True, exist_ok=True)
            record_path = self.baseline_log_dir / f'{uuid4().hex}_hpp_sizing_base.json'
            record = dict(design=list(map(float, design)), status='RUNNING', started_at=time.time(),
                          annualized_cost_hpp=None, pid=os.getpid(), error=None)
            write_json_atomic(record_path, record)

        class ObservedModel(Model):
            def solve(mdl, *args, **kwargs):
                solution = None
                try:
                    solution = super().solve(*args, **kwargs)
                    return solution
                finally:
                    d = mdl.solve_details
                    adapter.solve_observations.append(dict(
                        status_code=getattr(d, 'status_code', None), status=getattr(d, 'status', None),
                        seconds=getattr(d, 'time', None), has_solution=solution is not None))

        dispatch_path = dispatch_hash = None
        value = None
        try:
            design = list(map(float, design))
            if len(design) != len(self.list_vars) or not np.isfinite(design).all():
                raise ValueError('Expected a complete finite design vector')
            if abs(design[3]-round(design[3])) > 1e-9:
                raise ValueError('Native EGO supplied a noninteger turbine count')
            design[3] = float(round(design[3]))
            for name, val in zip(self.list_vars, design):
                var = self.variables[name]
                lo, hi = var['limits'] if var['var_type']=='design' else (var['value'], var['value'])
                if val < lo-1e-9 or val > hi+1e-9:
                    raise ValueError(f'Design outside configured domain: {name}')
            # Observe only the separate Phase 1 module; original EMS is untouched.
            with patch.object(dc_ems, 'Model', ObservedModel):
                outputs = super().evaluate(*design)
            self.final_checks = validate_result(self, design)
            if self.economics['annualized_cost_hpp'] > self.cost_upper_bound+1e-6 or not np.isfinite(outputs).all():
                raise ValueError('Nonfinite outputs or cost exceeds configured bound')
            if self.economics['served_facility_energy_mwh'] > 0:
                value = objective_value(self.economics, LCOE_OBJECTIVE)
            accepted = self.final_checks['outer_feasible'] and value is not None
            if accepted and value > self.objective_upper_bound+1e-6:
                raise ValueError('Feasible LCoE exceeds configured bound')
            self.candidate_status = 'FEASIBLE' if accepted else 'BELOW_TARGET'
            self.outputs = np.append(outputs, [1.0, -value if accepted else self.rejection_score])
            if record_path is not None:
                dispatch_path = record_path.with_suffix('.npz')
                temporary = dispatch_path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **self.hourly, soc_mwh=self.soc_mwh)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, dispatch_path)
                dispatch_hash = hashlib.sha256(dispatch_path.read_bytes()).hexdigest()
        except Exception as exc:
            self.candidate_status = rejected_status(self.solve_observations)
            if self.candidate_status == 'PROVEN_INFEASIBLE':
                # No reliability cap exists: this is a solver/model issue, not
                # evidence that the requested service target is impossible.
                self.candidate_status = 'SOLVER_MODEL_INFEASIBLE'
            self.candidate_error = f'{type(exc).__name__}: {exc}'
            self.outputs = np.zeros(len(self.list_out_vars))
            self.outputs[-1] = self.rejection_score
            # A model/validation failure is different from a valid BELOW_TARGET
            # dispatch. Do not serialize invalid or stale physical/economic data.
            self.final_checks = self.metrics = self.economics = None
            self.hourly = self.soc_mwh = None
            value = None
        if record_path is not None:
            record.update(design=design, status=self.candidate_status, error=self.candidate_error,
                score=float(self.outputs[-1]), objective=LCOE_OBJECTIVE, objective_units='EUR/MWh',
                lcoe_hpp_eur_per_mwh=value, annualized_cost_hpp=(self.economics or {}).get('annualized_cost_hpp'),
                solver=self.solve_observations, seconds=time.perf_counter()-start, completed_at=time.time(),
                validation=self.final_checks, metrics=self.metrics, economics=self.economics,
                solver_status=self.solver_status, physical_validation_passed=self.final_checks is not None,
                outer_feasible=self.candidate_status=='FEASIBLE',
                dispatch_path=str(dispatch_path) if dispatch_path else None, dispatch_sha256=dispatch_hash,
                rejection_score=self.rejection_score)
            write_json_atomic(record_path, record)
        return self.outputs

    def print_design(self, x_opt=None, outs=None):
        if self.candidate_status != 'FEASIBLE':
            print(f'No outer-feasible design: {self.candidate_status}. {self.candidate_error}')
            print('The finite rejection score is not a physical LCoE.')
        else:
            super().print_design(x_opt, outs)

