"""Target-slack energy workflow using unchanged native Parallel EGO.

Separate entry points preserve process-tree deadlines and saved-dispatch recovery.
Only worker/adapter bindings and new EMS diagnostics differ from the graded runner.
"""
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from uuid import uuid4

from hydesign.sizing_native_hpp_sizing_base_max_reliability import (
    ROOT, NATIVE_SETTINGS, HISTORY_COLUMNS, read_configuration, physical_inputs,
    _output_path, _frontier_row as _original_frontier_row, _frontier_report,
    independently_validate,
    prepare_configuration as _prepare_configuration,
    write_evaluation_history as _write_evaluation_history,
)

from hydesign.process_control_graded_max_reliability import run_contained, Deadline

EMS_DEFAULTS = {
    'ems_beta_energy_target_slack_energy': .25,
    'ems_kappa_battery_target_slack_energy': .001,
    'ems_mip_gap_target_slack_energy': 1e-4,
    'ems_seed_target_slack_energy': 0,
    'ems_time_limit_seconds': 30., 'ems_threads': 1,
}

CONTROL_DEFAULTS = {
    "hpp_sizing_cleanup_seconds": 10.,
    "hpp_sizing_recovery_time_limit_seconds": 60.,
    "hpp_sizing_reporting_time_limit_seconds": 30.,
    "hpp_sizing_heartbeat_seconds": 10.,
    "hpp_sizing_prevent_sleep": True,
}

COST_AWARE_DEFAULTS = {
    "hpp_sizing_cost_aware_lambda_ref": 10.0,  # EUR/MWh per excess hour at 99.9%
    "hpp_sizing_cost_aware_lambda_min": 0.5,   # EUR/MWh per excess hour, loose-target limit
    "hpp_sizing_cost_aware_smoothing_hours": 50.0,
}

PENALTY_SETTINGS = (
    "hpp_sizing_rejection_mode", "hpp_sizing_lcoe_ref", "hpp_sizing_penalty_gamma",
    *COST_AWARE_DEFAULTS,
)
SCORING_COLUMNS = (
    "cost_aware_lambda_ref", "cost_aware_lambda_min", "cost_aware_smoothing_hours",
    "cost_aware_reference_max_failed_hours", "lcoe_exceeds_reference_objective_bound",
    "rejection_mode", "max_failed_hours", "excess_failed_hours", "penalty_hours_scale",
    "lcoe_ref_eur_per_mwh", "penalty_gamma", "penalty_coefficient", "penalty_contribution",
    "base_rejection_objective", "invalid_rejection_objective", "ego_objective_score",
    "reliability_assessment", "exact_service_target_excluded", "reported_target_excluded",
    "reported_target_feasibility_unresolved",
)


def candidate_summary(record):
    """Small notebook/report view; full status and hourly archives remain saved."""
    status = record.get('solver_status') or {}
    certificate = status.get('certificate') or {}
    metrics = record.get('metrics') or {}
    stage = (status.get('stages') or [{}])[0]
    return dict(reliability_target=status.get('reliability_target'),
        achieved_reliability=metrics.get('firm_reliability'), LOLH=metrics.get('LOLH'),
        binary_failed_hours=status.get('binary_failed_hours'),
        reliability_slack=status.get('reliability_slack'),
        served_energy_mwh=status.get('delivered_energy_mwh'),
        battery_throughput_mwh=status.get('battery_throughput_mwh'),
        lcoe_hpp_eur_per_mwh=record.get('lcoe_hpp_eur_per_mwh'),
        ego_objective_score=record.get('ego_objective_score'),
        ems_status=status.get('termination'), ems_seconds=stage.get('seconds'),
        ems_relative_gap=stage.get('relative_gap'),
        minimum_slack_lower_bound=certificate.get('minimum_slack_lower_bound'),
        minimum_slack_upper_bound=certificate.get('minimum_slack_upper_bound'),
        minimum_reliability_violation_certified=certificate.get('minimum_reliability_violation_certified'),
        physically_validated=status.get('physically_validated', False),
        reliability_assessment=record.get('reliability_assessment'),
        error=record.get('error'))


def _frontier_row(result, target):
    row = _original_frontier_row(result, target)
    best = result.get('best') or {}
    info = candidate_summary(best)
    for key in ('reliability_slack', 'binary_failed_hours', 'minimum_slack_lower_bound',
                'minimum_slack_upper_bound', 'minimum_reliability_violation_certified',
                'ems_status', 'ems_relative_gap', 'ems_seconds'):
        row[key] = info[key]
    # Final recovery accepts only target-feasible designs, so score is actual LCoE.
    row['ego_objective_score'] = best.get('lcoe_hpp_eur_per_mwh')
    return row


def load_lcoe_reliability_experiment(experiment_dir, save_figures=False):
    """Reuse the existing plot and include slack diagnostics in the loaded table."""
    import pandas as pd
    directory = _output_path(experiment_dir)
    original_table, figure = _frontier_report(directory)
    manifest = json.loads((directory/'experiment_config.json').read_text())
    rows = []
    for case in manifest['cases']:
        path = directory/case['folder']/'result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        rows.append(_frontier_row(result, case['reliability_target']))
    table = pd.DataFrame(rows)
    table['nondominated'] = original_table['nondominated']
    if save_figures:
        report = _output_path(directory/(
            datetime.now(timezone.utc).strftime('reports_%Y%m%d_%H%M%SZ_')
            + uuid4().hex[:8] + '_target_slack_energy'))
        report.mkdir(exist_ok=False)
        table.to_csv(report/'summary.csv', index=False)
        figure.savefig(report/'pareto_frontier.png', dpi=200)
        figure.savefig(report/'pareto_frontier.pdf')
        print('Regenerated reports:', report)
    return table, figure


def run_fixed_design(config_path, config, design, output_dir):
    """Use the same bounded evaluation/checkpoint path, with all sizes fixed."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import write_json_atomic
    effective = deepcopy(config)
    for name, variable in effective['variables'].items():
        variable['var_type'] = 'fixed'
        if name in design: variable['value'] = design[name]
    effective['hpp_sizing_output_dir'] = str(output_dir)
    result = run_native_hpp_sizing(config_path, effective)
    records, errors = read_saved_evaluations(
        Path(result['run_directory'])/'evaluations', result['termination_reason'])
    completed = [r for r in records if 'completed_at' in r]
    record = completed[-1] if completed else dict(error=str(errors or result['final_status']))
    summary = candidate_summary(record)
    summary['run_directory'] = result['run_directory']
    write_json_atomic(Path(result['run_directory'])/'fixed_summary_target_slack_energy.json', summary)
    return summary


def prepare_configuration(config_path, overrides=None):
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_target_slack_energy import penalty_configuration
    requested = deepcopy(overrides or {})
    # New keys are adapter-only; never require changes in the shared base
    # configuration loader or YAML format.
    cost_aware_requested = {k: requested.pop(k) for k in COST_AWARE_DEFAULTS if k in requested}
    requested.setdefault("hpp_sizing_output_dir", str(ROOT/"hpp_sizing_runs/lcoe_reliability_frontier_target_slack_energy"))
    for key, default in EMS_DEFAULTS.items():
        requested.setdefault(key, default)
    requested.setdefault('reliability_target', .70)
    requested.setdefault('facility_load_mw', 16.)
    requested.setdefault('pue', 1.15)
    requested.setdefault('hpp_sizing_time_limit_seconds', 300.)
    cfg = _prepare_configuration(config_path, requested)
    cfg['dispatch_policy'] = 'target_slack_energy'
    for key in list(cfg):
        if key.endswith('_max_reliability'): cfg.pop(key)
    from hydesign.ems.ems_simple_datacenter_target_slack_energy import calibrate_penalties
    calibrate_penalties(8760, cfg['facility_load_mw'], 1., cfg['battery_charge_efficiency'],
        cfg['ems_beta_energy_target_slack_energy'], cfg['ems_kappa_battery_target_slack_energy'])
    if cfg.get('hpp_sizing_rejection_mode', 'cost_aware_rational') != 'cost_aware_rational':
        raise ValueError('This workflow requires cost_aware_rational scoring')
    cfg.setdefault("hpp_sizing_rejection_mode", "cost_aware_rational")
    cfg.setdefault("hpp_sizing_lcoe_ref", 100.0)
    cfg.setdefault("hpp_sizing_penalty_gamma", 1.0)
    for key, default in COST_AWARE_DEFAULTS.items():
        cfg[key] = cost_aware_requested.get(key, cfg.get(key, default))
    penalty_configuration(
        cfg["reliability_target"], 1.0, cfg["hpp_sizing_lcoe_ref"],
        cfg["hpp_sizing_penalty_gamma"], cfg["hpp_sizing_rejection_mode"],
        cfg["hpp_sizing_cost_aware_lambda_ref"],
        cfg["hpp_sizing_cost_aware_lambda_min"],
        cfg["hpp_sizing_cost_aware_smoothing_hours"])

    for key, value in CONTROL_DEFAULTS.items():
        cfg.setdefault(key, value)
    for key in ("hpp_sizing_time_limit_seconds", "hpp_sizing_cleanup_seconds",
                "hpp_sizing_recovery_time_limit_seconds", "hpp_sizing_reporting_time_limit_seconds",
                "hpp_sizing_heartbeat_seconds"):
        Deadline(cfg[key])
    if not isinstance(cfg["hpp_sizing_prevent_sleep"], bool):
        raise ValueError("hpp_sizing_prevent_sleep must be boolean")
    return cfg


def write_evaluation_history(records, path, started_at, ended_at,
                             objective="minimize_lcoe", reliability_target=None):
    """Extend the established history; physical values stay empty on failures."""
    _write_evaluation_history(records, path, started_at, ended_at, objective, reliability_target)
    with Path(path).open(encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    ordered = sorted(records, key=lambda r:(r.get("completed_at",ended_at),r["started_at"]))
    for row, record in zip(rows, ordered):
        row.update({key:record.get(key) for key in SCORING_COLUMNS})
        if "completed_at" not in record:
            row["reliability_assessment"] = "INTERRUPTED_NO_VALID_DISPATCH"
    # Do not export the old maximum-availability certificate under a new meaning.
    obsolete = {'maximum_availability_certified', 'failed_hours_lower_bound',
                'failed_hours_upper_bound', 'exact_reliability_lower_bound', 'exact_reliability_upper_bound'}
    columns = [key for key in columns if key not in obsolete]
    diagnostics = ('reliability_slack', 'observed_excess_binary_failed_hours',
                   'reported_excess_failed_hours', 'slack_matches_observed_excess',
                   'secondary_objective', 'observed_shortfall_hours_1e_7')
    certificates = ('minimum_slack_lower_bound', 'minimum_slack_upper_bound',
                    'minimum_reliability_violation_certified')
    for row, record in zip(rows, ordered):
        for key in obsolete: row.pop(key, None)
        status = record.get('solver_status') or {}
        row.update({key:status.get(key) for key in diagnostics})
        row.update({key:(status.get('certificate') or {}).get(key) for key in certificates})
    columns += list(diagnostics) + list(certificates)
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w",encoding="utf8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=columns+list(SCORING_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary,path)


def _worker(request_path, gate_path):
    # Deliberately no scientific imports before the supervisor assigns the job.
    while not Path(gate_path).exists():
        time.sleep(.02)
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_target_slack_energy import (
        hpp_model_simple_datacenter_hpp_sizing_target_slack_energy, VARIABLE_NAMES, write_json_atomic, objective_name,
    )
    request = json.loads(Path(request_path).read_text(encoding="utf8"))
    directory = Path(request["run_directory"])
    config = request["configuration"]
    inputs = dict(request["physical_inputs"], variables=config["variables"], num_batteries=1,
                  hpp_sizing_objective=config["hpp_sizing_objective"],
                  baseline_log_dir=str(directory / "evaluations"),
                  **{key:config[key] for key in PENALTY_SETTINGS})
    try:
        if any(v["var_type"] == "design" for v in config["variables"].values()):
            inputs.update({k: config[k] for k in NATIVE_SETTINGS})
            inputs.update(hpp_model=hpp_model_simple_datacenter_hpp_sizing_target_slack_energy,
                          opt_var=objective_name(config["hpp_sizing_objective"]),
                          final_design_fn=str(directory / "native_proposal_hpp_sizing_base.csv"))
            ego = EfficientGlobalOptimizationDriver(**inputs)
            ego.run()
            write_json_atomic(directory / "native_recorder_hpp_sizing_base.json", ego.recorder)
            result = {"status": "NATIVE_COMPLETED"}
        else:
            model = hpp_model_simple_datacenter_hpp_sizing_target_slack_energy(**inputs)
            try:
                model.evaluate(*[config["variables"][name]["value"] for name in VARIABLE_NAMES])
            finally:
                model.prob.cleanup()
            result = {"status": "FIXED_DESIGN_EVALUATED"}
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "NATIVE_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(directory / "completion_hpp_sizing_base.json", result)


def _search_command(request_path, gate_path):
    return [sys.executable, '-B', '-u', '-m', __name__ if __name__ != '__main__' else
            'hydesign.sizing_native_hpp_sizing_target_slack_energy', str(request_path), str(gate_path)]


def _control_options(config):
    return dict(cleanup_seconds=config['hpp_sizing_cleanup_seconds'],
                heartbeat_seconds=config['hpp_sizing_heartbeat_seconds'],
                prevent_sleep=config['hpp_sizing_prevent_sleep'])


def _run_phase(phase, request, config, directory):
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import write_json_atomic
    token = uuid4().hex[:8]
    path = directory/f'{phase}_request_{token}_target_slack_energy.json'
    gate = directory/f'{phase}_start_{token}_target_slack_energy.flag'
    write_json_atomic(path,request)
    command = [sys.executable,'-B','-u','-m','hydesign.sizing_native_hpp_sizing_target_slack_energy',
               '--'+phase,str(path),str(gate)]
    limit = config['hpp_sizing_recovery_time_limit_seconds' if phase=='recover'
                   else 'hpp_sizing_reporting_time_limit_seconds']
    outcome = run_contained(command,directory,limit,gate,phase=phase,**_control_options(config))
    write_json_atomic(directory/f'{phase}_status_{token}_target_slack_energy.json',outcome)
    return outcome


def read_saved_evaluations(directory, termination_reason):
    """Read completed atomic checkpoints, reporting corruption; never edit originals."""
    import math
    records, errors = [], []
    for path in Path(directory).glob('*_hpp_sizing_base.json'):
        try:
            record = json.loads(path.read_text(encoding='utf8'))
            if (not isinstance(record,dict) or len(record['design']) != 11 or
                    not all(math.isfinite(v) for v in record['design']) or
                    not math.isfinite(record['started_at']) or not isinstance(record['status'],str)):
                raise ValueError('Invalid checkpoint schema')
            if record['status']=='RUNNING':
                record.update(status='TIME_LIMIT_INTERRUPTED' if termination_reason=='TIME_LIMIT' else 'INTERRUPTED',
                              error=f'Evaluation interrupted: {termination_reason}')
            records.append(record)
        except Exception as exc:
            errors.append(dict(path=str(path),error=f'{type(exc).__name__}: {exc}'))
    return records, errors


def _recover_worker(request_path, gate_path):
    while not Path(gate_path).exists():
        time.sleep(.02)
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import write_json_atomic
    request = json.loads(Path(request_path).read_text(encoding='utf8'))
    directory = Path(request['run_directory'])
    config, inputs, outcome = request['configuration'],request['physical_inputs'],request['outcome']
    if outcome['termination_reason']=='NATIVE_COMPLETED':
        log=(directory/'native_ego_hpp_sizing_base.log').read_text(encoding='utf8',errors='replace')
        outcome['termination_reason']='NATIVE_CONVERGENCE' if 'Surrogate based optimization is converged.' in log else 'MAX_ITERATIONS'
    source = Path(request.get('candidate_directory',directory/'evaluations'))
    started = time.monotonic()
    records, errors = read_saved_evaluations(source,outcome['termination_reason'])
    history_path = directory/'history_hpp_sizing_base.csv'
    result = dict(outcome,configuration=config,run_directory=str(directory),history_path=None,
        best=None,final_status='RECOVERY_INCOMPLETE',recovery_complete=False,
        evaluated_designs=sum('completed_at' in r for r in records),attempted_evaluations=len(records),
        unique_designs=len({tuple(r['design']) for r in records}),
        feasible_evaluations=sum(r['status']=='FEASIBLE' for r in records),
        infeasible_evaluations=0,below_target_evaluations=sum(r['status']=='BELOW_TARGET' for r in records),
        unresolved_evaluations=sum(r['status'] not in ('FEASIBLE','BELOW_TARGET') for r in records),
        checkpoint_errors=errors,globally_optimal=False,validation_errors=[])
    result_path = directory/'result_hpp_sizing_base.json'
    write_json_atomic(result_path,result)
    try:
        write_evaluation_history(records,history_path,outcome['started_at'],outcome['ended_at'],
                                 config['hpp_sizing_objective'],config['reliability_target'])
        result['history_path']=str(history_path)
    except Exception as exc:
        result['history_error']=f'{type(exc).__name__}: {exc}'
    write_json_atomic(result_path,result)
    # Original validator checks checksum, dispatch physics, reliability, native
    # costs and actual LCoE. It skips corrupt archives and tries the next LCoE.
    best, validation_errors, seconds = independently_validate(records,inputs,directory,config['hpp_sizing_objective'])
    result.update(best=best,validation_errors=validation_errors,final_validation_seconds=seconds,
                  recovery_seconds=time.monotonic()-started,recovery_complete=True,
                  final_status='VALIDATED_FEASIBLE_INCUMBENT' if best else 'NO_VALIDATED_FEASIBLE_SOLUTION_FOUND',
                  time_limited_incumbent=best is not None and outcome['termination_reason']=='TIME_LIMIT')
    write_json_atomic(result_path,result)


def run_native_hpp_sizing(config_path, overrides=None):
    """Independent deadlines for native EGO, cleanup and saved-dispatch recovery."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import write_json_atomic
    config = prepare_configuration(config_path,overrides)
    run_dir = _output_path(Path(config['hpp_sizing_output_dir'])/(
        datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid4().hex[:8]+'_hpp_sizing_target_slack_energy'))
    (run_dir/'evaluations').mkdir(parents=True)
    inputs = physical_inputs(config_path,config,run_dir)
    digest = hashlib.sha256((ROOT/'hydesign/Parallel_EGO.py').read_bytes()).hexdigest()
    request = dict(configuration=config,physical_inputs=inputs,run_directory=str(run_dir),native_ego_sha256=digest)
    path = run_dir/'configuration_hpp_sizing_base.json'
    write_json_atomic(path,request)
    gate = run_dir/'start_hpp_sizing_base.flag'
    print('Native EGO results:',run_dir,flush=True)
    print('Outer wall-clock limit [s]:',config['hpp_sizing_time_limit_seconds'],flush=True)
    outcome = run_contained(_search_command(path,gate),run_dir,config['hpp_sizing_time_limit_seconds'],
                            gate,**_control_options(config))
    completion_path = run_dir/'completion_hpp_sizing_base.json'
    try:
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    except (ValueError,OSError) as exc:
        completion = dict(error=f'Invalid completion file: {exc}')
    if outcome['termination_reason']=='NORMAL_EXIT':
        outcome['termination_reason']=completion.get('status','PROCESS_ERROR')
        if completion.get('status')=='NATIVE_COMPLETED':
            # Reporting-only: reading the log is done in the recovery subprocess.
            outcome['termination_reason']='NATIVE_COMPLETED'
    outcome['error']=outcome.get('error') or completion.get('error')
    write_json_atomic(run_dir/'search_status_target_slack_energy.json',outcome)
    print('Search stopped:',outcome['termination_reason'],'-- bounded independent validation.',flush=True)
    request['outcome']=outcome
    recovery = _run_phase('recover',request,config,run_dir)
    result_path = run_dir/'result_hpp_sizing_base.json'
    try:
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
    except (OSError,ValueError):
        result={}
    if not result:
        result=dict(outcome,configuration=config,run_directory=str(run_dir),best=None,history_path=None,
                    evaluated_designs=None,final_status='RECOVERY_INCOMPLETE',recovery_complete=False,
                    globally_optimal=False)
    result['recovery_process']=recovery
    if recovery['termination_reason']=='USER_INTERRUPTED':
        result['search_termination_reason']=result['termination_reason']
        result['termination_reason']='USER_INTERRUPTED'
    result['recovery_complete']=bool(result.get('recovery_complete'))
    if not result['recovery_complete']:
        result['final_status']='RECOVERY_TIME_LIMIT' if recovery['termination_reason']=='TIME_LIMIT' else 'RECOVERY_ERROR'
    result['target_ended_at']=time.time()
    result['total_target_seconds']=result['target_ended_at']-outcome['started_at']
    write_json_atomic(result_path,result)
    return result


def _write_summary(directory):
    """Save the table before plotting; rendering cannot prevent a completed row."""
    manifest=json.loads((directory/'experiment_config.json').read_text())
    rows=[]
    for case in manifest['cases']:
        path=directory/case['folder']/'result.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        rows.append(_frontier_row(result,case['reliability_target']))
    usable=[r for r in rows if r['physical_validation_passed'] and r['lcoe_hpp_eur_per_mwh'] is not None]
    for row in rows:
        row['nondominated']=row in usable and not any(
            other['achieved_reliability_percent']>=row['achieved_reliability_percent'] and
            other['lcoe_hpp_eur_per_mwh']<=row['lcoe_hpp_eur_per_mwh'] and
            (other['achieved_reliability_percent']>row['achieved_reliability_percent'] or
             other['lcoe_hpp_eur_per_mwh']<row['lcoe_hpp_eur_per_mwh']) for other in usable)
    temporary=directory/'summary.tmp'
    with temporary.open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary,directory/'summary.csv')


def _report_worker(request_path,gate_path):
    while not Path(gate_path).exists():
        time.sleep(.02)
    request=json.loads(Path(request_path).read_text())
    directory=Path(request['directory'])
    def copy_atomic(source,destination):
        temporary=destination.with_name(destination.name+'.tmp')
        shutil.copyfile(source,temporary)
        os.replace(temporary,destination)
    if request.get('copy_case'):
        source=Path(request['copy_case']['source'])
        destination=directory/request['copy_case']['folder']
        destination.mkdir(exist_ok=False)
        # Publish result last: a completed result implies all available exports
        # were copied successfully, even if rendering is subsequently killed.
        for name in ('configuration.json','evaluations.csv','result.json'):
            if (source/name).exists(): copy_atomic(source/name,destination/name)
    if request.get('history_path'):
        copy_atomic(request['history_path'],directory/request['case_folder']/'evaluations.csv')
    _write_summary(directory)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Render separately and publish only complete figures. Previous reports
    # survive a renderer crash/timeout without being truncated.
    render=directory/('render_'+uuid4().hex[:8]+'_target_slack_energy')
    render.mkdir()
    _,fig=_frontier_report(directory,render)
    plt.close(fig)
    for name in ('pareto_frontier.png','pareto_frontier.pdf'):
        os.replace(render/name,directory/name)


def _completed_case(path):
    try:
        result=json.loads((path/'result.json').read_text())
        return (result.get('final_status') in ('VALIDATED_FEASIBLE_INCUMBENT','NO_VALIDATED_FEASIBLE_SOLUTION_FOUND')
                and result.get('termination_reason') not in ('USER_INTERRUPTED','ERROR','PROCESS_ERROR','NATIVE_ERROR')
                and result.get('recovery_complete',True))
    except (ValueError,OSError):
        return False


def run_lcoe_reliability_experiment(config_path,overrides,reliability_targets,output_dir,*,resume_from=None):
    """Bound each search/recovery/report independently; never modify a prior experiment."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import (
        CAPACITY_VARIABLES,LCOE_OBJECTIVE,objective_bound,write_json_atomic)
    config=prepare_configuration(config_path,overrides)
    if resume_from is not None:
        saved=json.loads((_output_path(resume_from)/'experiment_config.json').read_text())['configuration']
        execution_keys=set(CONTROL_DEFAULTS)|{'hpp_sizing_time_limit_seconds','hpp_sizing_output_dir'}
        changed=[key for key,value in saved.items() if key not in execution_keys and config.get(key)!=value]
        if changed:
            raise ValueError(f'Resume would change saved model/search assumptions: {changed}')
    if config['facility_load_mw']!=16 or config['pue']!=1.15:
        raise ValueError('This experiment requires 16 MW facility load and PUE 1.15')
    if any(config['variables'][name]['var_type']!='design' for name in CAPACITY_VARIABLES):
        raise ValueError('Activate all four HPP capacity variables for the frontier')
    cases=[]
    for target in reliability_targets:
        target=float(target)
        objective_bound(dict(config,reliability_target=target),1.,LCOE_OBJECTIVE)
        label=f'{100*target:07.3f}'.rstrip('0').rstrip('.').replace('.','p')
        cases.append(dict(reliability_target=target,folder='reliability_'+label))
    if not cases or len({c['folder'] for c in cases})!=len(cases):
        raise ValueError('Provide nonempty distinct targets')
    directory=_output_path(Path(output_dir)/(
        datetime.now(timezone.utc).strftime('experiment_%Y%m%d_%H%M%SZ_')+uuid4().hex[:8]+'_target_slack_energy'))
    directory.mkdir(parents=True,exist_ok=False)
    physical=physical_inputs(config_path,config,directory)
    manifest=dict(configuration=config,cases=cases,config_path=str(Path(config_path).resolve()),
        physical_inputs=physical,timestamp_timezone='UTC',resume_from=str(resume_from) if resume_from else None,
        native_ego_sha256=hashlib.sha256((ROOT/'hydesign/Parallel_EGO.py').read_bytes()).hexdigest(),
        weather_sha256=hashlib.sha256(Path(physical['input_ts_fn']).read_bytes()).hexdigest())
    write_json_atomic(directory/'experiment_config.json',manifest)
    print('LCoE experiment:',directory,flush=True)
    # A stalled renderer can no longer prevent the first search from starting.
    report=_run_phase('report',dict(directory=str(directory)),config,directory)
    if report['termination_reason']=='USER_INTERRUPTED': return directory
    for case in cases:
        case_dir=_output_path(directory/case['folder'])
        source=_output_path(Path(resume_from)/case['folder']) if resume_from else None
        if source is not None and _completed_case(source):
            outcome=_run_phase('report',dict(directory=str(directory),
                copy_case=dict(source=str(source),folder=case['folder'])),config,directory)
            if outcome['termination_reason']=='USER_INTERRUPTED': return directory
            if _completed_case(case_dir):
                print('Preserved completed target:',case['reliability_target'],flush=True)
                continue
            # Never rerun a completed target just because copying/reporting failed.
            raise RuntimeError(f'Could not preserve completed target {source}; source remains untouched: {outcome}')
        case_dir.mkdir(exist_ok=False)
        effective=deepcopy(config)
        effective.update(reliability_target=case['reliability_target'],hpp_sizing_output_dir=str(case_dir))
        resolved=physical_inputs(config_path,effective,case_dir)
        effective.update({k:resolved[k] for k in ('input_ts_fn','latitude','longitude','altitude')})
        write_json_atomic(case_dir/'configuration.json',effective)
        started=time.time()
        print('Target start UTC:',datetime.now(timezone.utc).isoformat(),case['reliability_target'],flush=True)
        try:
            result=run_native_hpp_sizing(config_path,effective)
        except KeyboardInterrupt:
            result=dict(best=None,final_status='INTERRUPTED',termination_reason='USER_INTERRUPTED')
        except Exception as exc:
            result=dict(best=None,final_status='RUN_ERROR',termination_reason='ERROR',error=f'{type(exc).__name__}: {exc}')
        result.setdefault('configuration',effective)
        result.update(_frontier_row(result,case['reliability_target']))
        result.update(target_started_at=started,target_ended_at=time.time())
        write_json_atomic(case_dir/'result.json',result)
        report=_run_phase('report',dict(directory=str(directory),history_path=result.get('history_path'),
                                       case_folder=case['folder']),config,directory)
        write_json_atomic(case_dir/'report_status_target_slack_energy.json',report)
        print('Target end UTC:',datetime.now(timezone.utc).isoformat(),
              'status:',result['termination_reason'],'evaluations:',result.get('evaluated_designs'),
              'best validated LCoE:',result.get('lcoe_hpp_eur_per_mwh'),flush=True)
        if result['termination_reason']=='USER_INTERRUPTED' or report['termination_reason']=='USER_INTERRUPTED': break
        if (result.get('process_cleanup',{}).get('verified') is False or
                result.get('recovery_process',{}).get('process_cleanup',{}).get('verified') is False or
                report['process_cleanup'].get('verified') is False):
            print('Stopping sweep: process-tree cleanup could not be verified.',flush=True)
            break
    return directory


def resume_lcoe_reliability_experiment(experiment_dir,output_dir=None,execution_overrides=None):
    """New experiment with completed targets copied, incomplete searches restarted.

    Saved physics/economics/penalties/targets are reused. Only execution controls
    can be overridden; this is not a resume of the native EGO surrogate state.
    """
    source=_output_path(experiment_dir)
    manifest=json.loads((source/'experiment_config.json').read_text())
    overrides=execution_overrides or {}
    allowed=set(CONTROL_DEFAULTS)|{'hpp_sizing_time_limit_seconds'}
    if set(overrides)-allowed:
        raise ValueError(f'Resume accepts execution controls only: {sorted(allowed)}')
    cfg=deepcopy(manifest['configuration'])
    cfg.update(overrides)
    for key in ('input_ts_fn','latitude','longitude','altitude'):
        if key in manifest.get('physical_inputs',{}): cfg[key]=manifest['physical_inputs'][key]
    if manifest.get('weather_sha256') and hashlib.sha256(Path(cfg['input_ts_fn']).read_bytes()).hexdigest()!=manifest['weather_sha256']:
        raise ValueError('Saved weather file changed; start a fresh experiment')
    if manifest.get('native_ego_sha256') and hashlib.sha256((ROOT/'hydesign/Parallel_EGO.py').read_bytes()).hexdigest()!=manifest['native_ego_sha256']:
        raise ValueError('Native EGO changed; start a fresh experiment')
    return run_lcoe_reliability_experiment(manifest['config_path'],cfg,
        [case['reliability_target'] for case in manifest['cases']],output_dir or source.parent,resume_from=source)


if __name__=='__main__':
    if sys.argv[1]=='--recover': _recover_worker(*sys.argv[2:])
    elif sys.argv[1]=='--report': _report_worker(*sys.argv[2:])
    else: _worker(*sys.argv[1:])


