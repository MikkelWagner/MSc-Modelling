"""Joint financial sizing through unchanged native Parallel EGO.

Search, saved-dispatch recovery and plotting have separate contained deadlines.
No search algorithm or EMS is implemented here. Saved results can be inspected
and plotted without constructing a model or invoking CPLEX.
"""
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

from hydesign.sizing_native_hpp_sizing_base import (
    ROOT, NATIVE_SETTINGS, read_configuration, physical_inputs, _output_path,
    prepare_configuration as prepare_hpp_configuration,
)
from hydesign.process_control_graded_max_reliability import run_contained, Deadline

MODULE = 'hydesign.sizing_native_joint_hpp_dc_sizing'
OBJECTIVE = 'maximize_penalized_annual_system_value'
CONTROL_DEFAULTS = dict(hpp_sizing_cleanup_seconds=10.,
    hpp_sizing_recovery_time_limit_seconds=300., hpp_sizing_reporting_time_limit_seconds=30.,
    hpp_sizing_heartbeat_seconds=10., hpp_sizing_prevent_sleep=True)


def prepare_configuration(config_path, overrides=None):
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import (
        IT_NAME, JOINT_DEFAULTS, penalty_parameters, validate_it_variable, selected_objective)
    cfg = read_configuration(config_path)
    requested = dict(ems_time_limit_seconds=30, ems_threads=1, ems_beta_max_reliability=.25,
                     ems_kappa_max_reliability=.10, ems_mip_gap_max_reliability=1e-4,
                     ems_seed_max_reliability=0)
    requested.update(deepcopy(overrides or {}))
    cfg.update(requested)
    cfg.pop('battery_power_fluctuation_penalty',None)
    cfg['dispatch_policy']='maximum_availability'
    cfg.setdefault('pue', 1.15)
    variables = deepcopy(cfg['variables'])
    it = variables.pop(IT_NAME, dict(var_type='design', types='float',
        value=16/cfg['pue'], limits=[1/cfg['pue'],50/cfg['pue']]))
    # Reuse native HPP-domain validation, without changing its allowed objectives.
    cfg.update(variables=variables,hpp_sizing_objective='minimize_annualized_cost')
    cfg = prepare_hpp_configuration(config_path,cfg)
    cfg.pop('battery_power_fluctuation_penalty',None)  # Not an input to this unchanged EMS.
    validate_it_variable(it)
    cfg['variables'][IT_NAME] = it
    mode=selected_objective(cfg)
    cfg['hpp_sizing_objective'] = OBJECTIVE if mode=='net_annual_value_system' else 'maximize_penalized_'+mode
    for key,value in {**JOINT_DEFAULTS,**CONTROL_DEFAULTS}.items(): cfg.setdefault(key,value)
    penalty_parameters(cfg)
    import math
    if not math.isfinite(cfg['pue']) or cfg['pue'] < 1:
        raise ValueError('PUE must be finite and >= 1')
    if not 0 < cfg['reliability_target'] <= 1:
        raise ValueError('Reliability target must be in (0, 1]')
    minimum_it = it['limits'][0] if it['var_type']=='design' else it['value']
    if minimum_it*cfg['pue'] <= cfg['epsilon_mw']:
        raise ValueError('Minimum facility demand must exceed the failed-hour tolerance')
    for key in ('hpp_sizing_time_limit_seconds','hpp_sizing_cleanup_seconds',
                'hpp_sizing_recovery_time_limit_seconds','hpp_sizing_reporting_time_limit_seconds',
                'hpp_sizing_heartbeat_seconds'):
        Deadline(cfg[key])
    if not isinstance(cfg['hpp_sizing_prevent_sleep'],bool):
        raise ValueError('hpp_sizing_prevent_sleep must be boolean')
    if not overrides or 'hpp_sizing_output_dir' not in overrides:
        cfg['hpp_sizing_output_dir'] = str(ROOT/'hpp_sizing_runs/joint_hpp_dc_sizing')
    return cfg


def read_saved_evaluations(directory, termination_reason='SAVED'):
    import math
    records, errors = [], []
    for path in Path(directory).glob('*_hpp_sizing_base.json'):
        try:
            record = json.loads(path.read_text(encoding='utf8'))
            if (len(record['design']) != 12 or not all(math.isfinite(v) for v in record['design'])
                    or not math.isfinite(record['started_at'])):
                raise ValueError('Invalid joint evaluation checkpoint')
            if record['status']=='RUNNING':
                record.update(status='INTERRUPTED',error=f'Interrupted: {termination_reason}')
            records.append(record)
        except Exception as exc:
            errors.append(dict(path=str(path),error=str(exc)))
    return sorted(records,key=lambda r:r.get('completed_at',r['started_at'])),errors


def write_evaluation_history(records,path,started_at):
    """One row per attempted design; preserve diagnostics for below-target results."""
    from hydesign.finance.finance_simple_datacenter import SCALAR_OUTPUTS
    rows = []
    for i,r in enumerate(records,1):
        d = r['design']; e = r.get('economics') or {}; m = r.get('metrics') or {}
        row = dict(evaluation_index=i,elapsed_seconds=r.get('completed_at',r['started_at'])-started_at,
            status=r['status'],accepted=r.get('outer_feasible',False),Nwt=d[3],wind_MW=d[2]*d[3],
            solar_MW=d[5],b_P=d[9],b_E_h=d[10],b_E=d[9]*d[10],it_capacity_mw=d[11],
            reliability=m.get('firm_reliability'),failed_hours=m.get('LOLH'),EENS=m.get('EENS'))
        row.update(e)
        for key in SCALAR_OUTPUTS: row.setdefault(key,None)
        row['joint_penalty_scaling']=r.get('joint_penalty_scaling','economic_bound')
        for k in ('joint_penalty_gamma','reference_revenue_eur_per_year','penalty_scale_eur_per_year'):
            row[k]=r.get(k)
        for k in ('score','objective','objective_units','reliability_target','lcoe_hpp_eur_per_mwh',
                  'unpenalized_objective','penalized_objective','reliability_penalty',
                  'max_failed_hours','excess_failed_hours','reliability_penalty_eur_per_year',
                  'penalized_annual_value','ego_minimization_target','physical_validation_passed',
                  'reliability_assessment','reported_target_feasibility_unresolved','error','seconds'):
            row[k]=r.get(k)
        for k in ('reference_revenue_pv_eur','reference_capex_pv_eur','npv_penalty_scale_eur',
                  'objective_penalty_scale','reference_capex_method','financial_penalty_scaling'):
            row[k]=(r.get('bounds') or {}).get(k)
        for k in ('solver_status','solver','validation','bounds'):
            row[k]=json.dumps(r.get(k),allow_nan=False)
        rows.append(row)
    columns=list(dict.fromkeys(k for row in rows for k in row)) or ['evaluation_index','status',*SCALAR_OUTPUTS]
    with Path(path).open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def independently_validate(records,inputs,variables,run_dir):
    """Recover by the selected unpenalized metric, independently of EGO ranking."""
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import (
        economic_bounds, objective_values, DC_COST_PARAMETERS, datacenter_cost_annualized,
        validate_result, IT_NAME, VARIABLE_NAMES, selected_objective, economic_objective, ANNUAL_OBJECTIVE)
    from hydesign.sizing_native_hpp_sizing_base_max_reliability import annual_cost_from_dispatch
    start=time.monotonic(); errors=[]
    bounds=economic_bounds(inputs,variables)
    mode=selected_objective(inputs)
    validated=[]
    candidates=[]
    for r in records:
        if r['status']!='FEASIBLE': continue
        try:
            value=economic_objective(r['economics'],mode)
            if not np.isfinite(value): raise ValueError('Nonfinite annual value')
            candidates.append((value,r))
        except Exception as exc: errors.append(dict(design=r['design'],error=str(exc)))
    for _,r in sorted(candidates,key=lambda item:item[0],reverse=True):
        try:
            d=r['design']; cfg=dict(inputs,facility_load_mw=d[-1]*inputs['pue'])
            for name,value in zip(VARIABLE_NAMES,d):
                var=variables[name]
                lo,hi=var['limits'] if var['var_type']=='design' else (var['value'],var['value'])
                if not lo-1e-9<=value<=hi+1e-9: raise ValueError('Saved design outside domain')
            if abs(d[3]-round(d[3]))>1e-9: raise ValueError('Noninteger turbine count')
            path=Path(r['dispatch_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=r['dispatch_sha256']:
                raise ValueError('Dispatch archive checksum mismatch')
            with np.load(path,allow_pickle=False) as archive:
                hourly={k:archive[k].copy() for k in archive.files if k!='soc_mwh'}
                soc=archive['soc_mwh'].copy()
            economics=dict(r['economics'])
            model=SimpleNamespace(sim_pars=cfg,hourly=hourly,soc_mwh=soc,
                                  solver_status=r['solver_status'],economics=economics)
            checks=validate_result(model,d[:-1])
            if not checks['outer_feasible']: raise ValueError('Saved dispatch is below target')
            def equal(actual,expected):
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-6)
            for key,value in checks['metrics'].items(): equal(r['metrics'][key],value)
            equal(economics['facility_load_mw'],cfg['facility_load_mw'])
            equal(economics['it_capacity_mw'],d[-1])
            dc_names=('CAPEX_dc_facility','CAPEX_dc_it','CAPEX_dc_total','OPEX_dc',
                    'annualized_CAPEX_dc_facility','annualized_CAPEX_dc_it','annualized_CAPEX_dc_total',
                    'annualized_cost_dc')
            if 'CAPEX_system' in economics:
                from hydesign.finance.finance_simple_datacenter import reconstruct_costs
                costs=reconstruct_costs(cfg,d,hourly['wind_mw'])
                equal(economics['annualized_cost_hpp'],costs['annualized_cost_hpp'])
                dc=[costs[k] for k in dc_names]
            else:
                equal(economics['annualized_cost_hpp'],annual_cost_from_dispatch(cfg,d[:-1],hourly['wind_mw']))
                dc=datacenter_cost_annualized(**{k:cfg[k] for k in DC_COST_PARAMETERS}).compute(d[-1])
            for name,value in zip(dc_names,dc): equal(economics[name],value)
            served_it=float(hourly['served_firm_it_mw'].sum())
            equal(economics['served_it_energy_mwh'],served_it)
            equal(economics['Revenue_dc'],served_it*cfg['datacenter_firm_revenue_per_mwh_it'])
            legacy = 'joint_penalty_scaling' not in r['bounds']
            saved_scaling=r['bounds'].get('joint_penalty_scaling','economic_bound')
            if saved_scaling != bounds['joint_penalty_scaling']:
                raise ValueError('Saved penalty scaling differs from the run configuration')
            if r.get('joint_penalty_scaling',saved_scaling) != saved_scaling:
                raise ValueError('Inconsistent saved penalty scaling')
            for key,value in bounds.items():
                # Old economic-bound archives predate these four metadata fields.
                if legacy and key in ('joint_penalty_scaling','joint_penalty_gamma',
                                      'reference_revenue_eur_per_year','penalty_scale_eur_per_year'):
                    continue
                if isinstance(value,(int,float)): equal(r['bounds'][key],value)
                elif r['bounds'][key] != value: raise ValueError(f'Saved bound metadata mismatch: {key}')
            # Lifetime reporting is independently reconstructed from this exact
            # validated dispatch and native raw costs, never from annualized costs.
            # Older checkpoints have no lifetime metrics and remain loadable.
            finance = None
            if 'CAPEX_system' in economics:
                from hydesign.finance.finance_simple_datacenter import (
                    FINANCE_PARAMETERS, INPUT_NAMES, OUTPUT_NAMES, SCALAR_OUTPUTS,
                    finance_simple_datacenter)
                finance_inputs = dict(costs,served_it_energy_mwh=served_it,
                    datacenter_firm_revenue_per_mwh_it=cfg['datacenter_firm_revenue_per_mwh_it'])
                component = finance_simple_datacenter(**{k:cfg[k] for k in FINANCE_PARAMETERS})
                calculated = dict(zip(OUTPUT_NAMES,component.compute(**{k:finance_inputs[k] for k in INPUT_NAMES})))
                for key in SCALAR_OUTPUTS:
                    if key in economics:
                        equal(np.nan if economics[key] is None else economics[key],calculated[key])
                    economics[key]=float(calculated[key]) if np.isfinite(calculated[key]) else None
                finance = dict(outputs={k:(v.tolist() if isinstance(v,np.ndarray) else
                    float(v) if np.isfinite(v) else None) for k,v in calculated.items()},
                    inputs={k:finance_inputs[k] for k in INPUT_NAMES},
                    assumptions={k:cfg[k] for k in FINANCE_PARAMETERS})
            scoring=objective_values(economics,checks['metrics']['LOLH'],cfg['reliability_target'],bounds)
            for key,value in scoring.items():
                if key not in r and mode==ANNUAL_OBJECTIVE and key in (
                        'unpenalized_objective','penalized_objective','reliability_penalty'): continue
                equal(r[key],value)
            equal(r['score'],scoring['penalized_objective'])
            var=variables[IT_NAME]
            lo,hi=var['limits'] if var['var_type']=='design' else (var['value'],var['value'])
            active='fixed' if var['var_type']=='fixed' else ('lower' if abs(d[-1]-lo)<1e-6 else
                    'upper' if abs(d[-1]-hi)<1e-6 else None)
            best=dict(design=d,metrics=r['metrics'],economics=economics,scoring=scoring,
                validation=checks,physical_validation_passed=True,solver_status=r['solver_status'],
                reliability_assessment=r.get('reliability_assessment'),it_capacity_bound=active,
                lcoe_hpp_eur_per_mwh=r.get('lcoe_hpp_eur_per_mwh'),
                dispatch_path=str(path),dispatch_sha256=r['dispatch_sha256'],
                objective=mode,objective_units=bounds.get('objective_units','EUR/year'),
                penalty_scales=bounds,
                recovery_method='independent saved-dispatch physics and finance; maximum feasible '+mode)
            if finance is not None:
                best['finance'] = finance
            validated.append((economic_objective(economics,mode),best))
            if mode==ANNUAL_OBJECTIVE: break  # Preserve legacy descending-value recovery.
        except Exception as exc:
            errors.append(dict(design=r['design'],error=f'{type(exc).__name__}: {exc}'))
    if not validated: return None,errors,time.monotonic()-start
    best=max(validated,key=lambda item:item[0])[1]
    with np.load(best['dispatch_path'],allow_pickle=False) as archive:
        pd.DataFrame({k:archive[k] for k in archive.files if k!='soc_mwh'}).to_csv(
            Path(run_dir)/'hourly_joint_hpp_dc_sizing.csv',index=False)
        np.save(Path(run_dir)/'soc_joint_hpp_dc_sizing.npy',archive['soc_mwh'])
    save_financial_tables(best,run_dir)
    return best,errors,time.monotonic()-start


def _worker(request):
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import (
        hpp_model_simple_datacenter_joint_hpp_dc_sizing as Model, VARIABLE_NAMES,OBJECTIVES,selected_objective,write_json_atomic)
    directory=Path(request['run_directory']); cfg=request['configuration']
    inputs=dict(request['physical_inputs'],variables=cfg['variables'],num_batteries=1,
                baseline_log_dir=str(directory/'evaluations'))
    try:
        if any(v['var_type']=='design' for v in cfg['variables'].values()):
            inputs.update({k:cfg[k] for k in NATIVE_SETTINGS})
            inputs.update(hpp_model=Model,opt_var=OBJECTIVES[selected_objective(cfg)][0],
                final_design_fn=str(directory/'native_proposal_joint_hpp_dc_sizing.csv'))
            ego=EfficientGlobalOptimizationDriver(**inputs); ego.run()
            write_json_atomic(directory/'native_recorder_joint_hpp_dc_sizing.json',ego.recorder)
            status=dict(status='NATIVE_COMPLETED')
        else:
            model=Model(**inputs)
            try: model.evaluate(*[cfg['variables'][name]['value'] for name in VARIABLE_NAMES])
            finally: model.prob.cleanup()
            status=dict(status='FIXED_DESIGN_EVALUATED')
    except Exception as exc:
        traceback.print_exc(); status=dict(status='NATIVE_ERROR',error=f'{type(exc).__name__}: {exc}')
    write_json_atomic(directory/'completion_joint_hpp_dc_sizing.json',status)


def _recover_worker(request):
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import write_json_atomic, selected_objective, OBJECTIVES
    directory=Path(request['run_directory']); cfg=request['configuration']; outcome=request['outcome']
    records,errors=read_saved_evaluations(directory/'evaluations',outcome['termination_reason'])
    history=directory/'evaluations.csv'
    write_evaluation_history(records,history,outcome['started_at'])
    result=dict(outcome,configuration=cfg,run_directory=str(directory),history_path=str(history),
        objective=selected_objective(cfg),objective_units=OBJECTIVES[selected_objective(cfg)][1],
        penalty_scales=request['economic_bounds'],
        evaluated_designs=sum('completed_at' in r for r in records),attempted_evaluations=len(records),
        feasible_evaluations=sum(r['status']=='FEASIBLE' for r in records),
        below_target_evaluations=sum(r['status']=='BELOW_TARGET' for r in records),
        checkpoint_errors=errors,globally_optimal=False,best=None,recovery_complete=False,
        final_status='RECOVERY_INCOMPLETE')
    path=directory/'result.json'; write_json_atomic(path,result)
    best,errors,seconds=independently_validate(records,request['physical_inputs'],cfg['variables'],directory)
    result.update(best=best,validation_errors=errors,recovery_seconds=seconds,recovery_complete=True,
        final_status='VALIDATED_FEASIBLE_INCUMBENT' if best else 'NO_VALIDATED_FEASIBLE_SOLUTION_FOUND')
    write_json_atomic(path,result)


def _run_phase(phase,request,cfg,directory):
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import write_json_atomic
    token=uuid4().hex[:8]; path=directory/f'{phase}_{token}.json'; gate=directory/f'{phase}_{token}.flag'
    write_json_atomic(path,request)
    limit=cfg[{'search':'hpp_sizing_time_limit_seconds','recover':'hpp_sizing_recovery_time_limit_seconds',
               'report':'hpp_sizing_reporting_time_limit_seconds'}[phase]]
    outcome=run_contained([sys.executable,'-B','-u','-m',MODULE,'--'+phase,str(path),str(gate)],
        directory,limit,gate,phase=phase,cleanup_seconds=cfg['hpp_sizing_cleanup_seconds'],
        heartbeat_seconds=cfg['hpp_sizing_heartbeat_seconds'],prevent_sleep=cfg['hpp_sizing_prevent_sleep'])
    write_json_atomic(directory/f'{phase}_status.json',outcome)
    return outcome


def run_native_joint_hpp_dc_sizing(config_path,overrides=None):
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import write_json_atomic,economic_bounds
    cfg=prepare_configuration(config_path,overrides)
    directory=_output_path(Path(cfg['hpp_sizing_output_dir'])/(
        datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid4().hex[:8]+'_joint_hpp_dc_sizing'))
    (directory/'evaluations').mkdir(parents=True)
    inputs=physical_inputs(config_path,cfg,directory)
    request=dict(configuration=cfg,physical_inputs=inputs,run_directory=str(directory),
        economic_bounds=economic_bounds(inputs,cfg['variables']),
        native_ego_sha256=hashlib.sha256((ROOT/'hydesign/Parallel_EGO.py').read_bytes()).hexdigest())
    write_json_atomic(directory/'configuration.json',request)
    print('Joint HPP-DC results:',directory,flush=True)
    outcome=_run_phase('search',request,cfg,directory)
    completion=directory/'completion_joint_hpp_dc_sizing.json'
    if outcome['termination_reason']=='NORMAL_EXIT':
        completed=json.loads(completion.read_text()) if completion.exists() else {}
        outcome['termination_reason']=completed.get('status','PROCESS_ERROR')
        outcome['error']=completed.get('error')
    request['outcome']=outcome
    recovery=_run_phase('recover',request,cfg,directory)
    path=directory/'result.json'
    result=json.loads(path.read_text()) if path.exists() else dict(outcome,best=None,
        configuration=cfg,run_directory=str(directory),recovery_complete=False)
    result['recovery_process']=recovery
    if not result.get('recovery_complete'): result['final_status']='RECOVERY_INCOMPLETE'
    write_json_atomic(path,result)
    report=_run_phase('report',request,cfg,directory)
    result.update(reporting_process=report,total_seconds=time.time()-outcome['started_at'])
    write_json_atomic(path,result)
    return result


def run_joint_revenue_scenarios(config_path, overrides, revenue_scenarios):
    """Group independent native runs and checkpoint their combined summary.

    Each run retains its own evaluations.csv, dispatches and configuration.
    Scalars retain their model units; the selected objective has an explicit unit.
    """
    return _run_joint_scenarios(config_path,overrides,revenue_scenarios,'revenue')


def run_joint_penalty_scenarios(config_path, overrides, penalty_epsilons):
    """Fresh native EGO runs at fixed EUR 800/MWh IT; vary only epsilon.

    Resolve common YAML/default inputs once so all scenarios share the same
    seed, DOE settings, design domain, finances and search budget.
    """
    requested=dict(datacenter_firm_revenue_per_mwh_it=800,
        joint_sizing_objective='NPV_over_PV_CAPEX_system',reliability_target=.999,pue=1.15,
        joint_penalty_scaling='reference_revenue',joint_penalty_mode='rational',
        joint_penalty_gamma=.25,joint_penalty_A=2.,joint_penalty_lambda=1e-9)
    requested.update(deepcopy(overrides or {}))
    cfg=prepare_configuration(config_path,requested)
    if cfg['datacenter_firm_revenue_per_mwh_it']!=800:
        raise ValueError('This penalty batch requires one fixed revenue of EUR 800/MWh IT')
    if cfg['joint_sizing_objective']!='NPV_over_PV_CAPEX_system':
        raise ValueError('This penalty batch compares NPV_over_PV_CAPEX_system')
    return _run_joint_scenarios(config_path,cfg,penalty_epsilons,'penalty')


def _penalty_summary_values(config):
    """Use the actual objective calibration, not an assumed monetary reference."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import economic_bounds,rational_penalty
    bounds=economic_bounds(config,config['variables'])
    scale=bounds['objective_penalty_scale']
    return dict(joint_penalty_epsilon=config['joint_penalty_epsilon'],
        joint_penalty_lambda=config['joint_penalty_lambda'],
        objective_penalty_scale=scale,reference_revenue_pv_eur=bounds['reference_revenue_pv_eur'],
        reference_capex_pv_eur=bounds['reference_capex_pv_eur'],
        effective_penalty_per_excess_hour=scale*rational_penalty(1,bounds),
        linear_tail_penalty_per_hour=scale*config['joint_penalty_epsilon'],
        initial_penalty_slope=scale*(config['joint_penalty_epsilon']+config['joint_penalty_lambda']),
        penalty_slope_units='NPV/PV CAPEX per excess failed hour')


def _run_joint_scenarios(config_path, overrides, scenario_values, batch_kind):
    """Shared timestamped batch/checkpoint loop for revenue and epsilon sweeps."""
    import math
    import os
    import pandas as pd
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import write_json_atomic
    rates=list(scenario_values)
    penalty_batch=batch_kind=='penalty'
    if not rates or any(isinstance(r,bool) or not isinstance(r,(int,float)) or
                        not math.isfinite(r) or r<0 for r in rates):
        raise ValueError('Provide finite positive epsilons' if penalty_batch else
                         'Provide finite nonnegative revenue assumptions in EUR/MWh_IT')
    if penalty_batch and (any(v<=0 for v in rates) or len(set(rates))!=len(rates)):
        raise ValueError('Provide distinct, finite, strictly positive penalty epsilons')
    base=deepcopy(overrides)
    parent=base.get('hpp_sizing_output_dir',ROOT/'hpp_sizing_runs/joint_hpp_dc_sizing')
    batch=_output_path(Path(parent)/(batch_kind+'_batch_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid4().hex[:8]))
    batch.mkdir(parents=True,exist_ok=False)
    write_json_atomic(batch/'configuration.json',dict(config_path=str(Path(config_path).resolve()),
        **({'penalty_epsilons':rates,'batch_kind':'penalty','varied_parameter':'joint_penalty_epsilon'}
           if penalty_batch else {'revenue_scenarios':rates}),overrides=base,summary_units=dict(
            CAPEX='EUR',NPV='EUR',annual_flows='EUR/year',ratios='dimensionless',payback='years',
            selected_objective='see objective_units column')))
    rows=[dict(scenario_index=i+1,revenue_eur_per_mwh_it=base['datacenter_firm_revenue_per_mwh_it'] if penalty_batch else r,
               status='PENDING',run_directory=None,
               evaluations_csv=None,error=None) for i,r in enumerate(rates)]
    if penalty_batch:
        for row,epsilon in zip(rows,rates):
            row.update(dict.fromkeys(('evaluations','feasible_evaluations','search_seconds','total_seconds',
                'Nwt','wind_MW','PV_MW_AC','battery_MW','battery_MWh','battery_duration_h',
                'it_capacity_mw','facility_load_mw','LOLH','reliability','physical_validation_passed',
                'effective_penalty_per_excess_hour','linear_tail_penalty_per_hour',
                'initial_penalty_slope','objective_penalty_scale')))
            row.update(joint_penalty_epsilon=epsilon,joint_penalty_lambda=base['joint_penalty_lambda'],
                reliability_target=base['reliability_target'],
                feasible_percentage=None,time_to_best_feasible_seconds=None,
                time_to_best_evaluated_feasible_seconds=None)
    from hydesign.finance.finance_simple_datacenter import SCALAR_OUTPUTS
    for row in rows: row.update(dict.fromkeys(SCALAR_OUTPUTS))
    def save_summary():
        temporary=batch/'summary.tmp'
        pd.DataFrame(rows).to_csv(temporary,index=False)
        os.replace(temporary,batch/'summary.csv')
    save_summary()
    print(batch_kind.capitalize()+' batch:',batch,flush=True)
    for row in rows:
        rate=row['revenue_eur_per_mwh_it']
        label=(f"epsilon_{row['joint_penalty_epsilon']:.12g}".replace('.','p') if penalty_batch
               else f"scenario_{row['scenario_index']:02d}_revenue_{rate:g}_eur_per_mwh_it")
        scenario=_output_path(batch/label)
        cfg=deepcopy(base)
        cfg.update(datacenter_firm_revenue_per_mwh_it=rate,hpp_sizing_output_dir=str(scenario))
        if penalty_batch: cfg['joint_penalty_epsilon']=row['joint_penalty_epsilon']
        row.update(status='RUNNING',scenario_directory=str(scenario))
        save_summary()
        print('Penalty epsilon:' if penalty_batch else 'Revenue assumption [EUR/MWh_IT]:',
              cfg['joint_penalty_epsilon'] if penalty_batch else rate,flush=True)
        result={}
        try:
            if penalty_batch: row.update(_penalty_summary_values(cfg))
            result=run_native_joint_hpp_dc_sizing(config_path,cfg)
            directory=_output_path(result['run_directory'])
            history=directory/'evaluations.csv'
            row.update(status=result['final_status'],run_directory=str(directory),
                objective=result['configuration'].get('joint_sizing_objective','net_annual_value_system'),
                objective_units=result.get('objective_units'),
                evaluations_csv=str(history) if history.exists() else None,
                reliability_target=result['configuration']['reliability_target'],
                evaluations=result.get('evaluated_designs'),feasible_evaluations=result.get('feasible_evaluations'),
                search_seconds=result.get('optimization_seconds'),total_seconds=result.get('total_seconds'),
                termination_reason=result.get('termination_reason'),error=result.get('error'))
            best=result.get('best')
            if best:
                d=best['design']; e=best['economics']; m=best['metrics']; s=best['scoring']
                row.update(Nwt=d[3],wind_MW=d[2]*d[3],PV_MW_AC=d[5],battery_MW=d[9],
                    battery_duration_h=d[10],battery_MWh=d[9]*d[10],it_capacity_mw=d[11],
                    facility_load_mw=e['facility_load_mw'],reliability=m['firm_reliability'],
                    LOLH=m['LOLH'],EENS=m['EENS'],physical_validation_passed=best.get('physical_validation_passed'),
                    lcoe_hpp_eur_per_mwh=best.get('lcoe_hpp_eur_per_mwh'))
                for key in ('Revenue_dc','annualized_cost_hpp','annualized_cost_dc',
                            'net_annual_value_system','served_it_energy_mwh','served_facility_energy_mwh'):
                    row[key]=e.get(key)
                row.update({key:e.get(key) for key in SCALAR_OUTPUTS})
                row['objective']=best.get('objective','net_annual_value_system')
                row['objective_units']=best.get('objective_units','EUR/year')
                for key in ('reliability_penalty_eur_per_year','penalized_annual_value',
                            'unpenalized_objective','penalized_objective','reliability_penalty'):
                    row[key]=s.get(key)
                for key in ('reference_revenue_pv_eur','reference_capex_pv_eur','objective_penalty_scale'):
                    row[key]=best.get('penalty_scales',{}).get(key,row.get(key))
            if penalty_batch:
                count=result.get('evaluated_designs',0)
                row['feasible_percentage']=100*result.get('feasible_evaluations',0)/count if count else None
                if history.exists():
                    curve=_feasible_penalty_history(history)
                    if not curve.empty:
                        peak=curve['NPV_over_PV_CAPEX_system'].max()
                        row['time_to_best_evaluated_feasible_seconds']=float(curve.loc[
                            curve['NPV_over_PV_CAPEX_system']==peak,'elapsed_seconds'].min())
                        if best:
                            import numpy as np
                            d=best['design']
                            matches=np.ones(len(curve),dtype=bool)
                            for key,value in zip(('Nwt','solar_MW','b_P','b_E_h','it_capacity_mw'),
                                                 (d[3],d[5],d[9],d[10],d[11])):
                                matches &= np.isclose(curve[key],value,rtol=1e-10,atol=1e-9)
                            matches &= np.isclose(curve['NPV_over_PV_CAPEX_system'],
                                best['economics']['NPV_over_PV_CAPEX_system'],rtol=1e-10,atol=1e-12)
                            if matches.any(): row['time_to_best_feasible_seconds']=float(curve.loc[matches,'elapsed_seconds'].min())
        except KeyboardInterrupt:
            row.update(status='INTERRUPTED',error='Batch interrupted by user')
            raise
        except Exception as exc:
            row.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        finally:
            save_summary()
        if result.get('termination_reason')=='USER_INTERRUPTED':
            break
    if penalty_batch: load_joint_penalty_scenarios(batch,save_figures=True)
    else: load_joint_revenue_scenarios(batch,save_figures=True)
    return batch,pd.DataFrame(rows)


def _feasible_penalty_history(path):
    """Only validated feasible, finite completed evaluations enter the curve."""
    import numpy as np
    import pandas as pd
    frame=pd.read_csv(path)
    if frame.empty: return frame
    valid=frame['status'].eq('FEASIBLE') & frame['physical_validation_passed'].astype(str).str.lower().eq('true')
    valid &= np.isfinite(frame['NPV_over_PV_CAPEX_system']) & np.isfinite(frame['elapsed_seconds'])
    return frame.loc[valid].sort_values('elapsed_seconds',kind='stable')


def load_joint_penalty_scenarios(batch_directory,save_figures=False):
    """Reload summaries and feasible histories without EMS/EGO or current YAML.

    Fresh figure directories preserve earlier plots. Running-best curves use
    evaluation-level validation; final summary values require final recovery.
    """
    import pandas as pd
    directory=_output_path(batch_directory)
    summary=pd.read_csv(directory/'summary.csv')
    if save_figures:
        import matplotlib.pyplot as plt
        report=_output_path(directory/('figures_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid4().hex[:8]))
        report.mkdir()
        ordered=summary.sort_values('joint_penalty_epsilon')
        for name,column,label in (
            ('penalty_comparison','NPV_over_PV_CAPEX_system','Best recovered feasible NPV / discounted lifetime CAPEX [-]'),
            ('penalty_feasible_percentage','feasible_percentage','Feasible evaluations / completed evaluations [%]')):
            fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
            ax.plot(ordered.joint_penalty_epsilon,ordered[column],marker='o')
            ax.set(xlabel='Penalty epsilon [-]',ylabel=label); ax.grid(alpha=.2)
            if column=='feasible_percentage': ax.set_ylim(0,100)
            for ext in ('png','pdf'): fig.savefig(report/(name+'.'+ext),dpi=180)
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        for _,row in summary.iterrows():
            path=row.get('evaluations_csv')
            if pd.isna(path): continue
            path=_output_path(path)
            if not path.exists():
                # Support moving the complete batch to another parent directory.
                candidates=list(directory.glob('epsilon_'+f"{row.joint_penalty_epsilon:.12g}".replace('.','p')+'/*/evaluations.csv'))
                if not candidates: continue
                path=candidates[-1]
            feasible=_feasible_penalty_history(path)
            if feasible.empty: continue
            x=feasible.elapsed_seconds.to_list()
            y=feasible.NPV_over_PV_CAPEX_system.cummax().to_list()
            end=row.get('search_seconds')
            if pd.notna(end) and end>x[-1]: x.append(end); y.append(y[-1])
            ax.step(x,y,where='post',label=f"epsilon = {row.joint_penalty_epsilon:g}")
        ax.set(xlabel='Elapsed optimization time [s]',ylabel='Best-so-far feasible NPV / discounted lifetime CAPEX [-]')
        ax.grid(alpha=.2)
        if ax.lines: ax.legend()
        else: ax.text(.5,.5,'No feasible evaluations saved',ha='center',transform=ax.transAxes)
        for ext in ('png','pdf'): fig.savefig(report/('penalty_convergence.'+ext),dpi=180)
        plt.close(fig)
    return summary


def load_joint_revenue_scenarios(batch_directory,save_figures=False):
    """Plot a saved batch summary, using only saved values and objective labels."""
    import pandas as pd
    directory=_output_path(batch_directory)
    summary=pd.read_csv(directory/'summary.csv')
    columns=('unpenalized_objective','NPV_system','it_capacity_mw')
    if save_figures and all(k in summary for k in columns):
        import matplotlib.pyplot as plt
        report=directory/('figures_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid4().hex[:8])
        report.mkdir()
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for objective,group in summary.groupby('objective',dropna=False):
            group=group.sort_values('revenue_eur_per_mwh_it')
            for ax,column,scale,label in zip(axes,columns,(1.,1e6,1.),
                    (str(objective)+' ['+str(group.objective_units.iloc[0])+']','NPV [million EUR]','Installed IT [MW]')):
                ax.plot(group.revenue_eur_per_mwh_it,group[column]/scale,marker='o',label=str(objective))
                ax.set(xlabel='Served IT revenue [EUR/MWh IT]',ylabel=label); ax.grid(alpha=.2)
        fig.savefig(report/'financial_revenue_sensitivity.png',dpi=180)
        fig.savefig(report/'financial_revenue_sensitivity.pdf'); plt.close(fig)
    return summary


def save_financial_tables(best, directory):
    """Export final-design reporting from saved vectors; no model or solver calls."""
    import numpy as np
    import pandas as pd
    finance=best.get('finance')
    if not finance: return None
    directory=_output_path(directory)
    f=finance['outputs']; raw=finance['inputs']
    discount=(1+finance['assumptions']['project_discount_rate'])**np.asarray(f['finance_years'])
    replacements=np.asarray(f['replacement_battery_vec'])+np.asarray(f['replacement_it_vec'])
    cashflows=pd.DataFrame(dict(year=f['finance_years'],annual_revenue=f['Revenue_dc_vec'],
        OPEX=f['OPEX_system_vec'],initial_investment=np.asarray(f['CAPEX_system_vec'])-replacements,
        replacement_expenditure=replacements,battery_replacement=f['replacement_battery_vec'],
        it_replacement=f['replacement_it_vec'],depreciation=f['depreciation_vec'],taxes=f['taxes_vec'],
        residual_value=f['residual_value_vec'],net_cash_flow=f['cashflow_vec'],
        discounted_cash_flow=f['discounted_cashflow_vec'],
        cumulative_discounted_cash_flow=f['cumulative_discounted_cashflow_vec']))
    cashflows['capital_expenditure']=f['CAPEX_system_vec']
    cashflows['discounted_capital_expenditure']=np.asarray(f['CAPEX_system_vec'])/discount
    cashflows['discounted_initial_investment']=cashflows.initial_investment/discount
    cashflows['discounted_replacement_expenditure']=replacements/discount
    cashflows.to_csv(directory/'cashflows.csv',index=False)
    rows=[dict(technology=label,CAPEX_EUR=raw['CAPEX_'+asset],OPEX_EUR_per_year=raw['OPEX_'+asset])
          for asset,label in (('w','Wind'),('s','PV'),('b','Battery'),('sh','Shared infrastructure'))]
    # The DC component supplies one aggregate OPEX; do not invent an allocation.
    rows.extend([dict(technology='DC facility',CAPEX_EUR=raw['CAPEX_dc_facility'],OPEX_EUR_per_year=None),
                 dict(technology='IT equipment',CAPEX_EUR=raw['CAPEX_dc_it'],OPEX_EUR_per_year=None),
                 dict(technology='DC aggregate OPEX',CAPEX_EUR=0.,OPEX_EUR_per_year=raw['OPEX_dc'])])
    for row,asset in zip(rows,('w','s','b','sh','dc_facility','dc_it',None)):
        payments=np.zeros(len(discount)) if asset is None else np.asarray(f['CAPEX_'+asset+'_vec'])
        replacement=f['replacement_battery_vec'] if asset=='b' else f['replacement_it_vec'] if asset=='dc_it' else np.zeros(len(discount))
        row.update(CAPEX_lifetime_EUR=float(payments.sum()),CAPEX_PV_EUR=float(np.sum(payments/discount)),
                   replacement_CAPEX_PV_EUR=float(np.sum(np.asarray(replacement)/discount)))
    pd.DataFrame(rows).to_csv(directory/'cost_breakdown.csv',index=False)
    return cashflows


def load_joint_hpp_dc_sizing(run_dir,save_figures=False):
    """Read saved outputs; optional figures use a fresh subfolder, never run EMS/EGO."""
    import pandas as pd
    directory=_output_path(run_dir)
    result=json.loads((directory/'result.json').read_text(encoding='utf8'))
    history=pd.read_csv(directory/'evaluations.csv')
    if save_figures:
        import matplotlib.pyplot as plt
        report=_output_path(directory/('figures_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid4().hex[:8]))
        report.mkdir()
        valid=history[history['status'].isin(['FEASIBLE','BELOW_TARGET'])].copy()
        if not valid.empty:
            fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
            for status,label in [('FEASIBLE','Validated feasible'),('BELOW_TARGET','Validated below target')]:
                rows=valid[valid.status==status]
                axes[0].scatter(rows.elapsed_seconds,rows.net_annual_value_system/1e6,label=label,s=24)
                axes[1].scatter(rows.it_capacity_mw,rows.net_annual_value_system/1e6,label=label,s=24)
            feasible=valid[valid.status=='FEASIBLE']
            axes[0].plot(feasible.elapsed_seconds,feasible.net_annual_value_system.cummax()/1e6,
                         color='black',label='Best feasible value')
            axes[0].set_xlabel('Elapsed search time [s]'); axes[1].set_xlabel('Installed IT capacity [MW IT]')
            for ax in axes:
                ax.set_ylabel('Annual system value [million EUR/year]'); ax.legend(); ax.grid(alpha=.2)
            fig.savefig(report/'joint_hpp_dc_sizing.png',dpi=180); fig.savefig(report/'joint_hpp_dc_sizing.pdf')
            plt.close(fig)
            if 'penalized_objective' in valid:
                fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
                mode=result['configuration'].get('joint_sizing_objective','net_annual_value_system')
                for status in ('FEASIBLE','BELOW_TARGET'):
                    rows=valid[valid.status==status]
                    ax.scatter(rows.elapsed_seconds,rows.penalized_objective,label=status,s=24)
                ax.plot(feasible.elapsed_seconds,feasible.unpenalized_objective.cummax(),color='black',label='Best feasible')
                ax.set(xlabel='Elapsed search time [s]',ylabel=mode+' ('+valid.objective_units.iloc[0]+')')
                ax.legend(); ax.grid(alpha=.2)
                fig.savefig(report/'financial_objective_convergence.png',dpi=180)
                fig.savefig(report/'financial_objective_convergence.pdf'); plt.close(fig)
        best=result.get('best')
        if best:
            cashflows=save_financial_tables(best,directory)
            if cashflows is not None:
                for name,column,label in (
                    ('cashflow','net_cash_flow','Net cash flow [million EUR]'),
                    ('cumulative_discounted_cashflow','cumulative_discounted_cash_flow',
                     'Cumulative discounted cash flow [million EUR]')):
                    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
                    if name=='cashflow': ax.bar(cashflows.year,cashflows[column]/1e6)
                    else: ax.plot(cashflows.year,cashflows[column]/1e6,marker='o',markersize=3)
                    ax.axhline(0,color='black',linewidth=.8)
                    ax.set(xlabel='Project year',ylabel=label); ax.grid(alpha=.2)
                    fig.savefig(report/(name+'.png'),dpi=180); fig.savefig(report/(name+'.pdf'))
                    plt.close(fig)
            d=best['design']
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            axes[0].bar(['Wind','PV AC','Battery','IT','Facility'],[d[2]*d[3],d[5],d[9],d[-1],best['economics']['facility_load_mw']])
            axes[0].set_ylabel('Installed capacity [MW]')
            axes[1].bar(['Battery energy'],[d[9]*d[10]]); axes[1].set_ylabel('Storage [MWh]')
            fig.savefig(report/'capacity_distribution.png',dpi=180)
            fig.savefig(report/'capacity_distribution.pdf'); plt.close(fig)
            pd.DataFrame([{**best['economics'],**best['metrics'],**best['scoring'],
                'objective':best.get('objective','net_annual_value_system'),
                'objective_units':best.get('objective_units','EUR/year'),
                'Nwt':d[3],'wind_MW':d[2]*d[3],'PV_MW_AC':d[5],
                'battery_MW':d[9],'battery_duration_h':d[10],'battery_MWh':d[9]*d[10],
                'lcoe_hpp_eur_per_mwh':best['lcoe_hpp_eur_per_mwh'],
                'evaluations':result.get('evaluated_designs'),'feasible_evaluations':result.get('feasible_evaluations'),
                'search_seconds':result.get('optimization_seconds'),'final_status':result['final_status'],
                'physical_validation_passed':best['physical_validation_passed'],
                'target':result['configuration']['reliability_target'],
                'it_capacity_bound':best['it_capacity_bound']}]).to_csv(report/'summary.csv',index=False)
        result['figure_directory']=str(report)
    return result,history


if __name__=='__main__':
    phase,path,gate=sys.argv[1:]
    while not Path(gate).exists(): time.sleep(.02)
    request=json.loads(Path(path).read_text(encoding='utf8'))
    if phase=='--search': _worker(request)
    elif phase=='--recover': _recover_worker(request)
    elif phase=='--report': load_joint_hpp_dc_sizing(request['run_directory'],save_figures=True)
    else: raise ValueError(f'Unknown phase: {phase}')
