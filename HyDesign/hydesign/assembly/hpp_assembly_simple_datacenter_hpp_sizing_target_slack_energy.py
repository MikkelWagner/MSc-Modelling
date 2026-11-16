"""Isolated target-slack adapter; native components, costs and scoring are imported unchanged."""
import hashlib
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
    objective_value, rejected_status, write_json_atomic,
)


from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_graded_max_reliability import (
    penalty_configuration, score_candidate,
)
from hydesign.ems import ems_simple_datacenter_target_slack_energy as dc_ems
from hydesign.ems.ems_simple_datacenter_comp_target_slack_energy import (
    ems_simple_datacenter_comp_target_slack_energy,
)


def reliability_assessment(checks, solver_status, epsilon_mw, observations=()):
    """Keep solver evidence separate from the tolerance-based acceptance rule."""
    status = solver_status or {}
    certificate = status.get("certificate") or {}
    lower = certificate.get("minimum_slack_lower_bound")
    excluded = lower is not None and lower > 0
    if checks is None:
        unresolved = any(not o.get("has_solution") and
                         (o.get("status_code") in (104, 106, 108) or
                          "time limit" in str(o.get("status", "")).lower())
                         for o in observations)
        assessment = "UNRESOLVED_NO_VALID_DISPATCH" if unresolved else "INVALID_EVALUATION"
    elif checks["outer_feasible"]:
        assessment = "VALIDATED_FEASIBLE"
    else:
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


class hpp_model_simple_datacenter_hpp_sizing_target_slack_energy(FixedAdapter):
    """Target-slack EMS with unchanged costs and cost-aware rational EGO scoring."""

    def __init__(self, *args, hpp_sizing_rejection_mode="cost_aware_rational",
                 hpp_sizing_lcoe_ref=100.0, hpp_sizing_penalty_gamma=1.0,
                 hpp_sizing_cost_aware_lambda_ref=10.0,
                 hpp_sizing_cost_aware_lambda_min=0.5,
                 hpp_sizing_cost_aware_smoothing_hours=50.0, **kwargs):
        if hpp_sizing_rejection_mode != "cost_aware_rational":
            raise ValueError("This workflow uses cost_aware_rational scoring")
        super().__init__(*args, **kwargs)
        self.penalty = penalty_configuration(
            self.sim_pars["reliability_target"], self.objective_upper_bound,
            hpp_sizing_lcoe_ref, hpp_sizing_penalty_gamma, hpp_sizing_rejection_mode,
            hpp_sizing_cost_aware_lambda_ref,
            hpp_sizing_cost_aware_lambda_min,
            hpp_sizing_cost_aware_smoothing_hours)
        self.scoring = None

    def get_prob(self, comps):
        """Replace only EMS before setup; inherit native generation and cost wiring."""
        cfg = self.sim_pars
        replacement = ems_simple_datacenter_comp_target_slack_energy(
            N_time=self.N_time, ems_type=cfg['ems_type'],
            reliability_target=cfg['reliability_target'],
            time_limit_seconds=cfg['ems_time_limit_seconds'], threads=cfg['ems_threads'],
            beta_energy=cfg.get('ems_beta_energy_target_slack_energy', .25),
            kappa_battery=cfg.get('ems_kappa_battery_target_slack_energy', .001),
            epsilon_mw=cfg['epsilon_mw'],
            mip_relative_gap=cfg.get('ems_mip_gap_target_slack_energy', 1e-4),
            random_seed=cfg.get('ems_seed_target_slack_energy', 0))
        return super().get_prob([('ems', replacement) if component[0] == 'ems' else component
                                 for component in comps])

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
            # Observe only the new EMS; every evaluation invokes one CPLEX solve.
            with patch.object(dc_ems, 'Model', ObservedModel):
                outputs = hpp_model_simple_datacenter_with_hpp_cost_max_reliability.evaluate(self, *design)
            self.final_checks = validate_result(self, design)
            if self.economics['annualized_cost_hpp'] > self.cost_upper_bound+1e-6 or not np.isfinite(outputs).all():
                raise ValueError('Nonfinite outputs or cost exceeds configured bound')
            if self.economics['served_facility_energy_mwh'] > 0:
                value = objective_value(self.economics, LCOE_OBJECTIVE)
            accepted = self.final_checks['outer_feasible'] and value is not None
            # The objective upper bound is an artificial legacy scoring
            # reference, NOT an actual cost feasibility constraint. Preserve
            # legacy behavior only in fixed/graded modes for reproducibility.
            if (self.penalty['rejection_mode'] in ('fixed', 'graded') and accepted and
                    value > self.objective_upper_bound+1e-6):
                raise ValueError('Feasible LCoE exceeds configured bound')
            self.candidate_status = 'FEASIBLE' if accepted else 'BELOW_TARGET'
            if self.penalty['rejection_mode'] in ('cost_aware', 'cost_aware_rational') and value is None:
                # Zero annual served energy means LCoE is undefined. Keep the
                # physically validated below-target record, but never invent
                # an LCoE for surrogate training.
                self.scoring = score_candidate(self.penalty, valid=False)
            else:
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
                # The reliability cap has unrestricted slack: this is a model issue, not
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
                lcoe_exceeds_reference_objective_bound=(
                    bool(value > self.objective_upper_bound+1e-6)
                    if value is not None else None),
                **self.penalty, **self.scoring, **self.assessment)
            write_json_atomic(record_path, record)
        return self.outputs

