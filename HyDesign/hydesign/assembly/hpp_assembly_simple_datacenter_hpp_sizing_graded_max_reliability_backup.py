"""Separate native EGO adapter with normalized excess-failed-hour rejection.

The EMS, physical assembly, cost bound and final acceptance remain unchanged.
``score`` is the negative adapter output; ``ego_objective_score`` is the
positive quantity native EGO minimizes. Neither rejection score is an LCoE.
"""
import hashlib
import math
import os
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from docplex.mp.model import Model
from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base_max_reliability import (
    hpp_model_simple_datacenter_hpp_sizing_base_max_reliability as FixedAdapter,
    hpp_model_simple_datacenter_with_hpp_cost_max_reliability,
    validate_result, VARIABLE_NAMES, LCOE_OBJECTIVE, objective_name,
    objective_value, rejected_status, write_json_atomic, dc_ems,
)


def penalty_configuration(reliability_target, objective_upper_bound,
                          lcoe_ref=100.0, gamma=1.0, mode="graded"):
    """Freeze the scale for a run; preserve the existing bound-based M."""
    for name, value in (("reliability_target", reliability_target),
                        ("objective_upper_bound", objective_upper_bound),
                        ("lcoe_ref", lcoe_ref), ("gamma", gamma)):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
    if not 0 < reliability_target <= 1:
        raise ValueError("reliability_target must be in (0, 1]")
    if objective_upper_bound < 0 or lcoe_ref <= 0 or gamma <= 0:
        raise ValueError("Require nonnegative objective bound and positive lcoe_ref/gamma")
    if mode not in ("fixed", "graded"):
        raise ValueError("hpp_sizing_rejection_mode must be fixed or graded")
    allowed = int(math.floor((1-reliability_target)*8760+1e-9))
    scale = 8760-allowed
    base = objective_upper_bound + max(
    1.0, 0.01 * objective_upper_bound)
    amplitude = gamma*lcoe_ref
    error_score = base+amplitude+max(1.0, .01*(base+amplitude))
    if not math.isfinite(error_score):
        raise ValueError("Penalty scale overflows; reduce lcoe_ref/gamma")
    return dict(rejection_mode=mode, max_failed_hours=allowed,
                penalty_hours_scale=scale, lcoe_ref_eur_per_mwh=float(lcoe_ref),
                penalty_gamma=float(gamma), penalty_coefficient=amplitude/scale,
                base_rejection_objective=float(base),
                invalid_rejection_objective=float(error_score if mode == "graded" else base))


def score_candidate(penalty, *, valid, feasible=False, lcoe=None, lolh=None):
    """Score only validated LOLH; failures never acquire invented physical data.

    Fixed mode reproduces the existing score, including its error score.
    Graded mode places invalid evaluations above every valid rejection.
    """
    if not valid:
        if feasible:
            raise ValueError("An invalid candidate cannot be feasible")
        return dict(ego_objective_score=penalty["invalid_rejection_objective"],
                    excess_failed_hours=None, penalty_contribution=None)
    if lolh is None or not math.isfinite(lolh) or not 0 <= lolh <= 8760 or int(lolh) != lolh:
        raise ValueError("Validated annual LOLH must be an integer in [0, 8760]")
    excess = max(0, int(lolh)-penalty["max_failed_hours"])
    if feasible != (excess == 0):
        raise ValueError("Acceptance and validated failed-hour count disagree")
    if feasible:
        if lcoe is None or not math.isfinite(lcoe) or lcoe < 0 or lcoe >= penalty["base_rejection_objective"]:
            raise ValueError("Invalid feasible LCoE or bound violation")
        score, contribution = float(lcoe), 0.0
    else:
        contribution = penalty["penalty_coefficient"]*excess if penalty["rejection_mode"] == "graded" else 0.0
        score = penalty["base_rejection_objective"]+contribution
    return dict(ego_objective_score=score, excess_failed_hours=excess,
                penalty_contribution=contribution)


def reliability_assessment(checks, solver_status, epsilon_mw, observations=()):
    """Keep solver evidence separate from the tolerance-based acceptance rule."""
    status = solver_status or {}
    certificate = status.get("certificate") or {}
    excluded = False
    if checks is None:
        unresolved = any(not o.get("has_solution") and
                         (o.get("status_code") in (104, 106, 108) or
                          "time limit" in str(o.get("status", "")).lower())
                         for o in observations)
        assessment = "UNRESOLVED_NO_VALID_DISPATCH" if unresolved else "INVALID_EVALUATION"
    elif checks["outer_feasible"]:
        assessment = "VALIDATED_FEASIBLE"
    else:
        lower = certificate.get("failed_hours_lower_bound")
        excluded = lower is not None and lower > checks["max_failed_hours"]
        if excluded:
            assessment = "BELOW_TARGET_EXACT_SERVICE_CERTIFIED"
        elif status.get("termination") == "TIME_LIMIT_INCUMBENT":
            assessment = "BELOW_TARGET_TIME_LIMIT_UNRESOLVED"
        else:
            assessment = "BELOW_TARGET_UNCERTIFIED"
    return dict(reliability_assessment=assessment,
                exact_service_target_excluded=bool(excluded) if checks is not None else None,
                reported_target_excluded=bool(excluded and epsilon_mw == 0) if checks is not None else None,
                reported_target_feasibility_unresolved=bool(
                    not checks["outer_feasible"] and not (excluded and epsilon_mw == 0))
                    if checks is not None else None)


class hpp_model_simple_datacenter_hpp_sizing_graded_max_reliability(FixedAdapter):
    """Reuse native components and validation; only change rejection scoring."""

    def __init__(self, *args, hpp_sizing_rejection_mode="graded",
                 hpp_sizing_lcoe_ref=100.0, hpp_sizing_penalty_gamma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty_configuration(
            self.sim_pars["reliability_target"], self.objective_upper_bound,
            hpp_sizing_lcoe_ref, hpp_sizing_penalty_gamma, hpp_sizing_rejection_mode)
        self.scoring = None

    def evaluate(self, *design):
        start = time.perf_counter()
        self.candidate_error = None
        self.solve_observations = []
        self.hourly = self.soc_mwh = self.metrics = self.economics = None
        self.final_checks = self.solver_status = None
        adapter = self
        self.scoring = None
        record_path = None
        if self.baseline_log_dir is not None:
            self.baseline_log_dir.mkdir(parents=True, exist_ok=True)
            record_path = self.baseline_log_dir / f'{uuid4().hex}_hpp_sizing_base.json'
            record = dict(design=list(map(float, design)), status='RUNNING', started_at=time.time(),
                          annualized_cost_hpp=None, pid=os.getpid(), error=None,
                          **self.penalty, ego_objective_score=None, excess_failed_hours=None,
                          penalty_contribution=None, reliability_assessment='RUNNING')
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
                outputs = hpp_model_simple_datacenter_with_hpp_cost_max_reliability.evaluate(self, *design)
            self.final_checks = validate_result(self, design)
            if self.economics['annualized_cost_hpp'] > self.cost_upper_bound+1e-6 or not np.isfinite(outputs).all():
                raise ValueError('Nonfinite outputs or cost exceeds configured bound')
            if self.economics['served_facility_energy_mwh'] > 0:
                value = objective_value(self.economics, LCOE_OBJECTIVE)
            accepted = self.final_checks['outer_feasible'] and value is not None
            if accepted and value > self.objective_upper_bound+1e-6:
                raise ValueError('Feasible LCoE exceeds configured bound')
            self.candidate_status = 'FEASIBLE' if accepted else 'BELOW_TARGET'
            self.scoring = score_candidate(self.penalty, valid=True, feasible=accepted,
                lcoe=value, lolh=self.final_checks['metrics']['LOLH'])
            self.outputs = np.append(outputs, [1.0, -self.scoring['ego_objective_score']])
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
            self.scoring = score_candidate(self.penalty, valid=False)
            self.outputs[-1] = -self.scoring['ego_objective_score']
            # A model/validation failure is different from a valid BELOW_TARGET
            # dispatch. Do not serialize invalid or stale physical/economic data.
            self.final_checks = self.metrics = self.economics = None
            self.hourly = self.soc_mwh = None
            value = None
        self.assessment = reliability_assessment(
            self.final_checks, self.solver_status, self.sim_pars['epsilon_mw'], self.solve_observations)
        if record_path is not None:
            record.update(design=design, status=self.candidate_status, error=self.candidate_error,
                score=float(self.outputs[-1]), objective=LCOE_OBJECTIVE, objective_units='EUR/MWh',
                lcoe_hpp_eur_per_mwh=value, annualized_cost_hpp=(self.economics or {}).get('annualized_cost_hpp'),
                solver=self.solve_observations, seconds=time.perf_counter()-start, completed_at=time.time(),
                validation=self.final_checks, metrics=self.metrics, economics=self.economics,
                solver_status=self.solver_status, physical_validation_passed=self.final_checks is not None,
                outer_feasible=self.candidate_status=='FEASIBLE',
                dispatch_path=str(dispatch_path) if dispatch_path else None, dispatch_sha256=dispatch_hash,
                rejection_score=self.rejection_score,
                **self.penalty, **self.scoring, **self.assessment)
            write_json_atomic(record_path, record)
        return self.outputs

