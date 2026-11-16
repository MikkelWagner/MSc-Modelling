"""Joint capacity adapter: unchanged maximum-reliability EMS and asset economics.

The actual EGO output is J = annual system value - rational penalty, EUR/year.
Native EGO maximizes this custom output by training its surrogate on -J.
Economic bounds and penalty calibration are fixed over the complete domain.
"""
import hashlib
import os
from pathlib import Path
import time
from uuid import uuid4
from unittest.mock import patch

import numpy as np
from docplex.mp.model import Model
from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost_max_reliability import (
    hpp_model_simple_datacenter_with_hpp_cost_max_reliability as PhysicalModel,
)
from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_sizing_base import (
    IT_NAME, DC_COST_PARAMETERS, validate_it_variable,
)
from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import (
    VARIABLE_NAMES as HPP_VARIABLE_NAMES, annual_cost_bound, write_json_atomic, rejected_status,
)
from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base_max_reliability import validate_result
from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_graded_max_reliability import reliability_assessment
from hydesign.costs.cost_simple_datacenter import datacenter_cost_annualized
from hydesign.ems import ems_simple_datacenter_max_reliability as dc_ems

VARIABLE_NAMES = (*HPP_VARIABLE_NAMES, IT_NAME)
SCORE_NAME = 'penalized_annual_value [EUR/year]'
JOINT_DEFAULTS = dict(joint_penalty_mode='rational', joint_penalty_A=2.,
    joint_penalty_lambda=3., joint_penalty_epsilon=1e-4,
    joint_economic_bound_factor=1., joint_value_scale_factor=1.,
    joint_penalty_scaling='economic_bound', joint_penalty_gamma=.25)


def penalty_parameters(config):
    params = {key:config.get(key, value) for key,value in JOINT_DEFAULTS.items()}
    if params['joint_penalty_mode'] != 'rational':
        raise ValueError('Only the rational joint penalty is supported')
    if params['joint_penalty_scaling'] not in ('economic_bound','reference_revenue'):
        raise ValueError('joint_penalty_scaling must be economic_bound or reference_revenue')
    for key,value in params.items():
        if key in ('joint_penalty_mode','joint_penalty_scaling'): continue
        if isinstance(value,bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if min(params['joint_economic_bound_factor'],params['joint_value_scale_factor']) < 1:
        raise ValueError('Economic bound and value-scale factors must be >= 1')
    if params['joint_penalty_scaling']=='economic_bound' and rational_penalty(1, params) <= 1 + 1e-8:
        raise ValueError('Require P(1) > 1 to guarantee feasible-over-infeasible ranking')
    return params


def rational_penalty(excess, params):
    """Dimensionless rational contribution plus linear tail; no allowed-hour penalty."""
    if not np.isfinite(excess) or excess < 0 or int(excess) != excess:
        raise ValueError('Excess failed hours must be a nonnegative integer')
    a, slope, tail = (params[k] for k in
                     ('joint_penalty_A','joint_penalty_lambda','joint_penalty_epsilon'))
    return float(a * (slope*excess/(a+slope*excess)) + tail*excess)


def economic_bounds(config, variables):
    """Conservative annual bounds, including shared BOS at maximum DC demand.

    Revenue <= IT price * maximum installed IT MW * 8760; costs are bounded
    by the unchanged asset-cost bound and DC cost at maximum installed capacity.
    These bounds never depend on the currently evaluated candidate.
    """
    params = penalty_parameters(config)
    validate_it_variable(variables[IT_NAME])
    var = variables[IT_NAME]
    maximum_it = float(var['limits'][1] if var['var_type']=='design' else var['value'])
    pue, rate = float(config['pue']), float(config['datacenter_firm_revenue_per_mwh_it'])
    if not np.isfinite([pue,rate]).all() or pue < 1 or rate < 0:
        raise ValueError('Require PUE >= 1 and finite nonnegative IT revenue coefficient')
    hpp = annual_cost_bound(config, {name:variables[name] for name in HPP_VARIABLE_NAMES},
                           facility_load_max_mw=maximum_it*pue)
    dc = float(datacenter_cost_annualized(**{k:config[k] for k in DC_COST_PARAMETERS}).compute(maximum_it)[-1])
    factor = params['joint_economic_bound_factor']
    costs, revenue = factor*(hpp+dc), factor*rate*maximum_it*8760
    scale = (costs+revenue)*params['joint_value_scale_factor']
    if not np.isfinite([costs,revenue,scale]).all() or scale <= 0:
        raise ValueError('Require a finite, positive full-domain economic range')
    reference = rate*maximum_it*8760  # Fixed domain reference; never actual served revenue.
    penalty_scale = scale
    if params['joint_penalty_scaling']=='reference_revenue':
        penalty_scale = params['joint_penalty_gamma']*reference/params['joint_penalty_A']
        if not np.isfinite(penalty_scale) or penalty_scale <= 0:
            raise ValueError('reference_revenue scaling requires positive finite potential DC revenue and penalty scale')
    return dict(cost_upper_bound_eur_per_year=costs, revenue_upper_bound_eur_per_year=revenue,
        hpp_cost_upper_bound_eur_per_year=factor*hpp, dc_cost_upper_bound_eur_per_year=factor*dc,
        value_lower_bound_eur_per_year=-costs, value_upper_bound_eur_per_year=revenue,
        value_scale_eur_per_year=scale, maximum_it_capacity_mw=maximum_it,
        maximum_facility_load_mw=maximum_it*pue,
        reference_revenue_eur_per_year=reference, penalty_scale_eur_per_year=penalty_scale, **params)


def objective_values(economics, lolh, reliability_target, bounds):
    """Return the actual annual objective consumed by EGO, with separate diagnostics."""
    if not np.isfinite(reliability_target) or not 0 < reliability_target <= 1:
        raise ValueError('Reliability target must be in (0, 1]')
    if not np.isfinite(lolh) or int(lolh) != lolh or not 0 <= lolh <= 8760:
        raise ValueError('Expected validated integer hourly LOLH in [0, 8760]')
    revenue = float(economics['Revenue_dc'])
    hpp, dc = float(economics['annualized_cost_hpp']), float(economics['annualized_cost_dc'])
    value = float(economics['net_annual_value_system'])
    if not np.isfinite([revenue,hpp,dc,value]).all() or min(revenue,hpp,dc) < 0:
        raise ValueError('Invalid annual economics')
    np.testing.assert_allclose(value,revenue-hpp-dc,rtol=1e-12,atol=1e-6)
    if (revenue > bounds['revenue_upper_bound_eur_per_year']+1e-6
            or hpp > bounds['hpp_cost_upper_bound_eur_per_year']+1e-6
            or dc > bounds['dc_cost_upper_bound_eur_per_year']+1e-6
            or hpp+dc > bounds['cost_upper_bound_eur_per_year']+1e-6):
        raise ValueError('Economics outside full-domain bounds; not clipped')
    allowed = int(np.floor((1-reliability_target)*8760+1e-9))
    excess = max(0,int(lolh)-allowed)
    penalty_scale = bounds.get('penalty_scale_eur_per_year',bounds['value_scale_eur_per_year'])
    penalty = penalty_scale*rational_penalty(excess,bounds)
    objective = value-penalty
    if not np.isfinite([penalty,objective]).all():
        raise ValueError('Nonfinite rational objective')
    return dict(net_annual_value_system=value, reliability_penalty_eur_per_year=penalty,
        penalized_annual_value=objective, ego_objective_score=objective,
        ego_minimization_target=-objective, max_failed_hours=allowed, excess_failed_hours=excess,
        normalized_annual_value=(value+bounds['cost_upper_bound_eur_per_year'])/
            (bounds['revenue_upper_bound_eur_per_year']+bounds['cost_upper_bound_eur_per_year']),
        normalized_reliability_penalty=rational_penalty(excess,bounds))


def invalid_objective(bounds, target):
    allowed = int(np.floor((1-target)*8760+1e-9))
    scale = bounds.get('penalty_scale_eur_per_year',bounds['value_scale_eur_per_year'])
    floor = -bounds['cost_upper_bound_eur_per_year'] - scale*rational_penalty(8760-allowed,bounds)
    return float(floor-max(1.,.01*scale))


class hpp_model_simple_datacenter_joint_hpp_dc_sizing(PhysicalModel):
    """Twelve outer inputs; eleven unchanged HPP inputs and candidate IT capacity."""
    def __init__(self, sim_pars_fn, variables, baseline_log_dir=None, **kwargs):
        super().__init__(sim_pars_fn=sim_pars_fn, **kwargs)
        if set(variables) != set(VARIABLE_NAMES):
            raise ValueError('Expected eleven HPP inputs plus installed IT capacity')
        validate_it_variable(variables[IT_NAME])
        self.variables = variables
        self.list_vars = list(VARIABLE_NAMES)
        self.list_out_vars += ['reliability_penalty [EUR/year]', SCORE_NAME]
        self.bounds = economic_bounds(self.sim_pars,variables)
        self.rejection_score = invalid_objective(self.bounds,self.sim_pars['reliability_target'])
        self.baseline_log_dir = None if baseline_log_dir is None else Path(baseline_log_dir)
        self.candidate_status = None

    def evaluate(self, *design):
        start = time.perf_counter()
        design = list(map(float,design))
        self.hourly = self.soc_mwh = self.metrics = self.economics = None
        self.final_checks = self.solver_status = self.scoring = None
        self.candidate_error = None
        self.solve_observations = []
        self.outputs = np.zeros(len(self.list_out_vars))
        self.outputs[-1] = self.rejection_score
        record = dict(design=design,status='RUNNING',started_at=time.time(),pid=os.getpid(),
                      annualized_cost_hpp=None,error=None)
        path = None
        if self.baseline_log_dir is not None:
            self.baseline_log_dir.mkdir(parents=True,exist_ok=True)
            # Preserve the existing telemetry reader's checkpoint convention.
            path = self.baseline_log_dir/f'{uuid4().hex}_hpp_sizing_base.json'
            write_json_atomic(path,record)
        adapter = self
        class ObservedModel(Model):
            def solve(mdl,*args,**kwargs):
                solution = None
                try:
                    solution = super().solve(*args,**kwargs)
                    return solution
                finally:
                    d = mdl.solve_details
                    adapter.solve_observations.append(dict(status_code=getattr(d,'status_code',None),
                        status=getattr(d,'status',None),seconds=getattr(d,'time',None),has_solution=solution is not None))
        archive = digest = None
        try:
            if len(design)!=12 or not np.isfinite(design).all():
                raise ValueError('Expected twelve finite inputs')
            for name,value in zip(VARIABLE_NAMES,design):
                var = self.variables[name]
                lo,hi = var['limits'] if var['var_type']=='design' else (var['value'],var['value'])
                if not lo-1e-9 <= value <= hi+1e-9:
                    raise ValueError(f'Design outside configured domain: {name}')
            if design[-1] <= 0 or abs(design[3]-round(design[3])) > 1e-9:
                raise ValueError('Require positive IT capacity and integer turbine count')
            design[3] = float(round(design[3]))
            load = design[-1]*self.sim_pars['pue']
            self.sim_pars['facility_load_mw'] = load
            self.prob.set_val('facility_load_mw',load)
            self.prob.set_val('pue',self.sim_pars['pue'])
            with patch.object(dc_ems,'Model',ObservedModel):
                outputs = super().evaluate(*design[:-1])
            self.final_checks = validate_result(self,design[:-1])
            np.testing.assert_allclose(self.economics['it_capacity_mw'],design[-1],rtol=0,atol=1e-8)
            self.scoring = objective_values(self.economics,self.final_checks['metrics']['LOLH'],
                                           self.sim_pars['reliability_target'],self.bounds)
            self.candidate_status = 'FEASIBLE' if self.final_checks['outer_feasible'] else 'BELOW_TARGET'
            if not np.isfinite(outputs).all(): raise ValueError('Nonfinite model outputs')
            self.outputs = np.append(outputs,[self.scoring['reliability_penalty_eur_per_year'],
                                             self.scoring['penalized_annual_value']])
            if path is not None:
                archive = path.with_suffix('.npz')
                temporary = archive.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,**self.hourly,soc_mwh=self.soc_mwh)
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary,archive)
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        except Exception as exc:
            self.candidate_status = rejected_status(self.solve_observations)
            if self.candidate_status=='PROVEN_INFEASIBLE': self.candidate_status='SOLVER_MODEL_INFEASIBLE'
            self.candidate_error = f'{type(exc).__name__}: {exc}'
            self.hourly = self.soc_mwh = self.metrics = self.economics = self.final_checks = self.scoring = None
            self.outputs = np.zeros(len(self.list_out_vars)); self.outputs[-1] = self.rejection_score
        self.inputs = design
        assessment = reliability_assessment(self.final_checks,self.solver_status,
                                           self.sim_pars['epsilon_mw'],self.solve_observations)
        status, economics = self.solver_status or {}, self.economics or {}
        record.update(status=self.candidate_status,error=self.candidate_error,design=design,
            score=float(self.outputs[-1]),objective='maximize_penalized_annual_system_value',objective_units='EUR/year',
            joint_penalty_scaling=self.bounds['joint_penalty_scaling'],
            joint_penalty_gamma=self.bounds['joint_penalty_gamma'],
            reference_revenue_eur_per_year=self.bounds['reference_revenue_eur_per_year'],
            penalty_scale_eur_per_year=self.bounds['penalty_scale_eur_per_year'],
            it_capacity_mw=design[-1] if len(design)==12 else None,
            facility_load_mw=economics.get('facility_load_mw'),pue=self.sim_pars['pue'],
            reliability_target=self.sim_pars['reliability_target'],
            annualized_cost_hpp=economics.get('annualized_cost_hpp'),
            lcoe_hpp_eur_per_mwh=(economics['annualized_cost_hpp']/economics['served_facility_energy_mwh']
                                if economics.get('served_facility_energy_mwh',0)>0 else None),
            solver=self.solve_observations,solver_status=self.solver_status,
            validation=self.final_checks,metrics=self.metrics,economics=self.economics,
            physical_validation_passed=self.final_checks is not None,outer_feasible=self.candidate_status=='FEASIBLE',
            dispatch_path=str(archive) if archive else None,dispatch_sha256=digest,bounds=self.bounds,
            seconds=time.perf_counter()-start,completed_at=time.time(),
            **assessment,**(self.scoring or dict(ego_objective_score=self.rejection_score,
                                                ego_minimization_target=-self.rejection_score)))
        self.record = record
        if path is not None: write_json_atomic(path,record)
        return self.outputs

    def print_design(self,x_opt=None,outs=None):
        print('Candidate:',self.candidate_status,self.candidate_error or '')
        if self.economics is not None:
            print('IT MW:',self.economics['it_capacity_mw'],'facility MW:',self.economics['facility_load_mw'])
            print('Annual value / penalty / EGO objective [EUR/year]:',
                  self.scoring['net_annual_value_system'],self.scoring['reliability_penalty_eur_per_year'],
                  self.scoring['penalized_annual_value'])
