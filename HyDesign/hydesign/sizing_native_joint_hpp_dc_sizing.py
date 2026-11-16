"""Joint annual-value sizing through unchanged native Parallel EGO.

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
    hpp_sizing_recovery_time_limit_seconds=60., hpp_sizing_reporting_time_limit_seconds=30.,
    hpp_sizing_heartbeat_seconds=10., hpp_sizing_prevent_sleep=True)


def prepare_configuration(config_path, overrides=None):
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import (
        IT_NAME, JOINT_DEFAULTS, penalty_parameters, validate_it_variable)
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
    cfg['hpp_sizing_objective'] = OBJECTIVE
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
    rows = []
    for i,r in enumerate(records,1):
        d = r['design']; e = r.get('economics') or {}; m = r.get('metrics') or {}
        row = dict(evaluation_index=i,elapsed_seconds=r.get('completed_at',r['started_at'])-started_at,
            status=r['status'],accepted=r.get('outer_feasible',False),Nwt=d[3],wind_MW=d[2]*d[3],
            solar_MW=d[5],b_P=d[9],b_E_h=d[10],b_E=d[9]*d[10],it_capacity_mw=d[11],
            reliability=m.get('firm_reliability'),failed_hours=m.get('LOLH'),EENS=m.get('EENS'))
        row.update(e)
        row['joint_penalty_scaling']=r.get('joint_penalty_scaling','economic_bound')
        for k in ('joint_penalty_gamma','reference_revenue_eur_per_year','penalty_scale_eur_per_year'):
            row[k]=r.get(k)
        for k in ('score','objective','objective_units','reliability_target','lcoe_hpp_eur_per_mwh',
                  'max_failed_hours','excess_failed_hours','reliability_penalty_eur_per_year',
                  'penalized_annual_value','ego_minimization_target','physical_validation_passed',
                  'reliability_assessment','reported_target_feasibility_unresolved','error','seconds'):
            row[k]=r.get(k)
        for k in ('solver_status','solver','validation','bounds'):
            row[k]=json.dumps(r.get(k),allow_nan=False)
        rows.append(row)
    columns=list(dict.fromkeys(k for row in rows for k in row)) or ['evaluation_index','status']
    with Path(path).open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def independently_validate(records,inputs,variables,run_dir):
    """Recover by annual value, verifying dispatch and economics without EMS solves."""
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import (
        economic_bounds, objective_values, DC_COST_PARAMETERS, datacenter_cost_annualized,
        validate_result, IT_NAME, VARIABLE_NAMES)
    from hydesign.sizing_native_hpp_sizing_base_max_reliability import annual_cost_from_dispatch
    start=time.monotonic(); errors=[]
    bounds=economic_bounds(inputs,variables)
    candidates=[]
    for r in records:
        if r['status']!='FEASIBLE': continue
        try:
            value=float(r['economics']['net_annual_value_system'])
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
            economics=r['economics']
            model=SimpleNamespace(sim_pars=cfg,hourly=hourly,soc_mwh=soc,
                                  solver_status=r['solver_status'],economics=economics)
            checks=validate_result(model,d[:-1])
            if not checks['outer_feasible']: raise ValueError('Saved dispatch is below target')
            def equal(actual,expected):
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-6)
            for key,value in checks['metrics'].items(): equal(r['metrics'][key],value)
            equal(economics['facility_load_mw'],cfg['facility_load_mw'])
            equal(economics['it_capacity_mw'],d[-1])
            equal(economics['annualized_cost_hpp'],annual_cost_from_dispatch(cfg,d[:-1],hourly['wind_mw']))
            dc=datacenter_cost_annualized(**{k:cfg[k] for k in DC_COST_PARAMETERS}).compute(d[-1])
            for name,value in zip(('CAPEX_dc_facility','CAPEX_dc_it','CAPEX_dc_total','OPEX_dc',
                    'annualized_CAPEX_dc_facility','annualized_CAPEX_dc_it','annualized_CAPEX_dc_total',
                    'annualized_cost_dc'),dc): equal(economics[name],value)
            served_it=float(hourly['served_firm_it_mw'].sum())
            equal(economics['served_it_energy_mwh'],served_it)
            equal(economics['Revenue_dc'],served_it*cfg['datacenter_firm_revenue_per_mwh_it'])
            scoring=objective_values(economics,checks['metrics']['LOLH'],cfg['reliability_target'],bounds)
            for key,value in scoring.items(): equal(r[key],value)
            equal(r['score'],scoring['penalized_annual_value'])
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
            pd.DataFrame(hourly).to_csv(Path(run_dir)/'hourly_joint_hpp_dc_sizing.csv',index=False)
            np.save(Path(run_dir)/'soc_joint_hpp_dc_sizing.npy',soc)
            var=variables[IT_NAME]
            lo,hi=var['limits'] if var['var_type']=='design' else (var['value'],var['value'])
            active='fixed' if var['var_type']=='fixed' else ('lower' if abs(d[-1]-lo)<1e-6 else
                    'upper' if abs(d[-1]-hi)<1e-6 else None)
            best=dict(design=d,metrics=r['metrics'],economics=economics,scoring=scoring,
                validation=checks,physical_validation_passed=True,solver_status=r['solver_status'],
                reliability_assessment=r.get('reliability_assessment'),it_capacity_bound=active,
                lcoe_hpp_eur_per_mwh=r.get('lcoe_hpp_eur_per_mwh'),
                dispatch_path=str(path),dispatch_sha256=r['dispatch_sha256'],
                recovery_method='independent saved-dispatch physics and economics; maximum feasible annual value')
            return best,errors,time.monotonic()-start
        except Exception as exc:
            errors.append(dict(design=r['design'],error=f'{type(exc).__name__}: {exc}'))
    return None,errors,time.monotonic()-start


def _worker(request):
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import (
        hpp_model_simple_datacenter_joint_hpp_dc_sizing as Model, VARIABLE_NAMES,SCORE_NAME,write_json_atomic)
    directory=Path(request['run_directory']); cfg=request['configuration']
    inputs=dict(request['physical_inputs'],variables=cfg['variables'],num_batteries=1,
                baseline_log_dir=str(directory/'evaluations'))
    try:
        if any(v['var_type']=='design' for v in cfg['variables'].values()):
            inputs.update({k:cfg[k] for k in NATIVE_SETTINGS})
            inputs.update(hpp_model=Model,opt_var=SCORE_NAME,
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
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import write_json_atomic
    directory=Path(request['run_directory']); cfg=request['configuration']; outcome=request['outcome']
    records,errors=read_saved_evaluations(directory/'evaluations',outcome['termination_reason'])
    history=directory/'evaluations.csv'
    write_evaluation_history(records,history,outcome['started_at'])
    result=dict(outcome,configuration=cfg,run_directory=str(directory),history_path=str(history),
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
    Summary monetary values remain in EUR/year; notebook conversion is display-only.
    """
    import math
    import os
    import pandas as pd
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import write_json_atomic
    rates=list(revenue_scenarios)
    if not rates or any(isinstance(r,bool) or not isinstance(r,(int,float)) or
                        not math.isfinite(r) or r<0 for r in rates):
        raise ValueError('Provide finite nonnegative revenue assumptions in EUR/MWh_IT')
    base=deepcopy(overrides)
    parent=base.get('hpp_sizing_output_dir',ROOT/'hpp_sizing_runs/joint_hpp_dc_sizing')
    batch=_output_path(Path(parent)/('revenue_batch_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid4().hex[:8]))
    batch.mkdir(parents=True,exist_ok=False)
    write_json_atomic(batch/'configuration.json',dict(config_path=str(Path(config_path).resolve()),
        revenue_scenarios=rates,overrides=base,summary_monetary_units='EUR/year'))
    rows=[dict(scenario_index=i+1,revenue_eur_per_mwh_it=r,status='PENDING',run_directory=None,
               evaluations_csv=None,error=None) for i,r in enumerate(rates)]
    def save_summary():
        temporary=batch/'summary.tmp'
        pd.DataFrame(rows).to_csv(temporary,index=False)
        os.replace(temporary,batch/'summary.csv')
    save_summary()
    print('Revenue batch:',batch,flush=True)
    for row in rows:
        rate=row['revenue_eur_per_mwh_it']
        scenario=_output_path(batch/f"scenario_{row['scenario_index']:02d}_revenue_{rate:g}_eur_per_mwh_it")
        cfg=deepcopy(base)
        cfg.update(datacenter_firm_revenue_per_mwh_it=rate,hpp_sizing_output_dir=str(scenario))
        row.update(status='RUNNING',scenario_directory=str(scenario))
        save_summary()
        print('Revenue assumption [EUR/MWh_IT]:',rate,flush=True)
        result={}
        try:
            result=run_native_joint_hpp_dc_sizing(config_path,cfg)
            directory=_output_path(result['run_directory'])
            history=directory/'evaluations.csv'
            row.update(status=result['final_status'],run_directory=str(directory),
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
                for key in ('reliability_penalty_eur_per_year','penalized_annual_value'):
                    row[key]=s.get(key)
        except KeyboardInterrupt:
            row.update(status='INTERRUPTED',error='Batch interrupted by user')
            raise
        except Exception as exc:
            row.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        finally:
            save_summary()
        if result.get('termination_reason')=='USER_INTERRUPTED':
            break
    return batch,pd.DataFrame(rows)


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
        best=result.get('best')
        if best:
            d=best['design']
            pd.DataFrame([{**best['economics'],**best['metrics'],**best['scoring'],
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
