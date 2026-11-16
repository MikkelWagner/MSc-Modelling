"""Native Parallel EGO with the separate maximum-availability EMS.

Imports the existing process containment, configuration, history and plotting
helpers unchanged. The worker, adapter, recovery and experiment entry points
are explicitly separate; no original module bindings are replaced.
"""
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from uuid import uuid4

from hydesign.sizing_native_hpp_sizing_base import (
    ROOT, NATIVE_SETTINGS, HISTORY_COLUMNS, read_configuration,
    prepare_configuration as _prepare_configuration, physical_inputs,
    run_contained, read_evaluations, _output_path,
    write_evaluation_history as _write_evaluation_history,
    _frontier_row, _frontier_report, load_lcoe_reliability_experiment)


def prepare_configuration(config_path, overrides=None):
    requested = deepcopy(overrides or {})
    requested.setdefault("hpp_sizing_objective", "minimize_lcoe")
    if requested["hpp_sizing_objective"] != "minimize_lcoe":
        raise ValueError("This separate workflow minimizes LCoE")
    requested.setdefault("ems_time_limit_seconds", 30)
    requested.setdefault("ems_threads", 1)
    requested.setdefault("ems_beta_max_reliability", .25)
    requested.setdefault("ems_kappa_max_reliability", .10)
    requested.setdefault("ems_mip_gap_max_reliability", 1e-4)
    requested.setdefault("ems_seed_max_reliability", 0)
    requested.setdefault("hpp_sizing_output_dir", str(ROOT/"hpp_sizing_runs/lcoe_reliability_frontier_max_reliability"))
    cfg = _prepare_configuration(config_path, requested)
    cfg.pop("battery_power_fluctuation_penalty", None)
    cfg["dispatch_policy"] = "maximum_availability"
    return cfg


def write_evaluation_history(records, path, started_at, ended_at,
                             objective="minimize_lcoe", reliability_target=None):
    """Keep the established CSV layout and retain below-target economics/physics."""
    _write_evaluation_history(records, path, started_at, ended_at, objective, reliability_target)
    with Path(path).open(encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    extra = ("physical_validation_passed", "outer_feasible", "dispatch_path",
             "exact_reliability_lower_bound", "exact_reliability_upper_bound",
             "maximum_availability_certified", "binary_failed_hours",
             "binary_lolh_mismatch_hours", "ems_termination", "curtailment_mwh",
             "battery_throughput_mwh", "ems_relative_gap", "ems_seconds",
             "failed_hours_lower_bound", "failed_hours_upper_bound", "certificate_scope")
    ordered = sorted(records, key=lambda r:(r.get("completed_at",ended_at),r["started_at"]))
    for row, record in zip(rows, ordered):
        status = record.get("solver_status") or {}
        certificate = status.get("certificate") or {}
        economics, metrics = record.get('economics') or {}, record.get('metrics') or {}
        row.update({key:value for key,value in economics.items() if key in columns})
        row.update(EENS=metrics.get('EENS'),reliability=metrics.get('firm_reliability'),
                   failed_hours=metrics.get('LOLH'),curtailment_mwh=metrics.get('curtailment_mwh'),
                   battery_throughput_mwh=metrics.get('battery_throughput_mwh'),
                   ems_relative_gap=(status.get('stages') or [{}])[0].get('relative_gap'),
                   ems_seconds=(status.get('stages') or [{}])[0].get('seconds'),
                   **{k:certificate.get(k) for k in ('failed_hours_lower_bound','failed_hours_upper_bound','certificate_scope')})
        row.update(annualized_cost_hpp=record.get("annualized_cost_hpp"),
                   lcoe_hpp_eur_per_mwh=record.get("lcoe_hpp_eur_per_mwh"),
                   physical_validation_passed=record.get("physical_validation_passed",False),
                   outer_feasible=record.get("outer_feasible",False),
                   dispatch_path=record.get("dispatch_path"),
                   ems_termination=status.get("termination"),
                   **{k:certificate.get(k) for k in extra[3:6]},
                   **{k:status.get(k) for k in extra[6:8]})
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w",encoding="utf8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=columns+list(extra))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary,path)


def annual_cost_from_dispatch(cfg, design, wind):
    """Independently recalculate installed HPP costs through unchanged native classes."""
    import numpy as np
    from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
    from hydesign.costs.cost_simple_hpp_with_hpp_cost import hpp_cost_annualized
    from hydesign.wind.wind import get_rotor_d
    clearance, sp, rated, count, density, solar, tilt, azimuth, ratio, power, duration = design
    diameter = get_rotor_d(rated*1e6/sp)
    def params(*keys):
        return {key:cfg[key] for key in keys}
    cw, ow = wpp_cost(N_time=len(wind), **params(
        'wind_turbine_cost','wind_civil_works_cost','wind_fixed_onm_cost',
        'wind_variable_onm_cost','d_ref','hh_ref','p_rated_ref')).compute(
            count, clearance+diameter/2, diameter, rated, wind)
    cs, osolar = pvp_cost(**params('solar_PV_cost','solar_hardware_installation_cost',
        'solar_inverter_cost','solar_fixed_onm_cost')).compute(solar,ratio)
    cb, ob = battery_cost(life_y=1, battery_price_reduction_per_year=0, **params(
        'battery_energy_cost','battery_power_cost','battery_BOP_installation_commissioning_cost',
        'battery_control_system_cost','battery_energy_onm_cost')).compute(
            power*duration,power,np.ones(len(wind)))
    csh, osh, *_ = shared_cost(**params('hpp_BOS_soft_cost','hpp_grid_connection_cost','land_cost')).compute(
        cfg['facility_load_mw']+power, count*rated/density, solar*cfg['land_use_per_solar_MW'])
    annual = hpp_cost_annualized(**params('wind_WACC','solar_WACC','battery_WACC',
        'hpp_economic_lifetime_years','battery_replacement_year','battery_replacement_cost_fraction'))
    return float(annual.compute(cw,ow,cs,osolar,cb,ob,csh,osh)[-1])


def independently_validate(records, inputs, run_dir, objective='minimize_lcoe'):
    """Validate the actual scored incumbent, never resample a time-limited EMS.

    Check archive hash, all physical balances, threshold, energy, native costs
    and LCoE independently. Reject corrupt checkpoints and try the next LCoE.
    """
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base_max_reliability import validate_result
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import objective_value
    if objective != 'minimize_lcoe':
        raise ValueError('Expected LCoE objective')
    start = time.monotonic()
    ranked, errors = [], []
    for record in records:
        if record['status'] != 'FEASIBLE':
            continue
        try:
            ranked.append((objective_value(record['economics'],objective),record))
        except Exception as exc:
            errors.append(dict(design=record['design'],error=str(exc)))
    for value, record in sorted(ranked,key=lambda item:item[0]):
        try:
            path = Path(record['dispatch_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['dispatch_sha256']:
                raise ValueError('Dispatch archive checksum mismatch')
            with np.load(path,allow_pickle=False) as archive:
                hourly = {k:archive[k].copy() for k in archive.files if k!='soc_mwh'}
                soc = archive['soc_mwh'].copy()
            model = SimpleNamespace(sim_pars=inputs,hourly=hourly,soc_mwh=soc,
                                    solver_status=record['solver_status'],economics=record['economics'])
            checks = validate_result(model,record['design'])
            if not checks['outer_feasible']:
                raise ValueError('Saved dispatch is below the requested target')
            np.testing.assert_allclose(value,record['lcoe_hpp_eur_per_mwh'],rtol=1e-10,atol=1e-8)
            np.testing.assert_allclose(-value,record['score'],rtol=1e-10,atol=1e-8)
            native_cost = annual_cost_from_dispatch(inputs,record['design'],hourly['wind_mw'])
            np.testing.assert_allclose(native_cost,record['economics']['annualized_cost_hpp'],rtol=1e-9,atol=1e-6)
            pd.DataFrame(hourly).to_csv(Path(run_dir)/'hourly_hpp_sizing_base.csv',index=False)
            np.save(Path(run_dir)/'soc_hpp_sizing_base.npy',soc)
            best = dict(design=record['design'],metrics=checks['metrics'],economics=record['economics'],
                        solver_status=record['solver_status'],validation=checks,lcoe_hpp_eur_per_mwh=value,
                        dispatch_path=record['dispatch_path'],dispatch_sha256=record['dispatch_sha256'],
                        recovery_method='independent validation of the scored saved dispatch')
            return best, errors, time.monotonic()-start
        except Exception as exc:
            errors.append(dict(design=record['design'],error=f'{type(exc).__name__}: {exc}'))
    return None,errors,time.monotonic()-start


def _worker(request_path, gate_path):
    # Deliberately no scientific imports before the supervisor assigns the job.
    while not Path(gate_path).exists():
        time.sleep(.02)
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base_max_reliability import (
        hpp_model_simple_datacenter_hpp_sizing_base_max_reliability, VARIABLE_NAMES, write_json_atomic, objective_name,
    )
    request = json.loads(Path(request_path).read_text(encoding="utf8"))
    directory = Path(request["run_directory"])
    config = request["configuration"]
    inputs = dict(request["physical_inputs"], variables=config["variables"], num_batteries=1,
                  hpp_sizing_objective=config["hpp_sizing_objective"],
                  baseline_log_dir=str(directory / "evaluations"))
    try:
        if any(v["var_type"] == "design" for v in config["variables"].values()):
            inputs.update({k: config[k] for k in NATIVE_SETTINGS})
            inputs.update(hpp_model=hpp_model_simple_datacenter_hpp_sizing_base_max_reliability,
                          opt_var=objective_name(config["hpp_sizing_objective"]),
                          final_design_fn=str(directory / "native_proposal_hpp_sizing_base.csv"))
            ego = EfficientGlobalOptimizationDriver(**inputs)
            ego.run()
            write_json_atomic(directory / "native_recorder_hpp_sizing_base.json", ego.recorder)
            result = {"status": "NATIVE_COMPLETED"}
        else:
            model = hpp_model_simple_datacenter_hpp_sizing_base_max_reliability(**inputs)
            try:
                model.evaluate(*[config["variables"][name]["value"] for name in VARIABLE_NAMES])
            finally:
                model.prob.cleanup()
            result = {"status": "FIXED_DESIGN_EVALUATED"}
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "NATIVE_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(directory / "completion_hpp_sizing_base.json", result)


def run_native_hpp_sizing(config_path, overrides=None):
    """Notebook entry point: configure, supervise native EGO, recover and validate."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import write_json_atomic
    config = prepare_configuration(config_path, overrides)
    run_dir = _output_path(Path(config["hpp_sizing_output_dir"]) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8] + "_hpp_sizing_base_max_reliability"))
    (run_dir / "evaluations").mkdir(parents=True)
    inputs = physical_inputs(config_path, config, run_dir)
    native_path = ROOT / "hydesign/Parallel_EGO.py"
    digest = hashlib.sha256(native_path.read_bytes()).hexdigest()
    request_path = run_dir / "configuration_hpp_sizing_base.json"
    write_json_atomic(request_path, {"configuration": config, "physical_inputs": inputs,
                                    "run_directory": str(run_dir), "native_ego_sha256": digest})
    gate_path = run_dir / "start_hpp_sizing_base.flag"
    print("Native EGO results:", run_dir, flush=True)
    print("Outer limit [s]:", config["hpp_sizing_time_limit_seconds"], flush=True)
    command = [sys.executable, "-B", "-u", "-m", "hydesign.sizing_native_hpp_sizing_base_max_reliability",
               str(request_path), str(gate_path)]
    outcome = run_contained(command, run_dir, config["hpp_sizing_time_limit_seconds"], gate_path)
    completion_path = run_dir / "completion_hpp_sizing_base.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    if outcome["termination_reason"] == "NORMAL_EXIT":
        outcome["termination_reason"] = completion.get("status", "PROCESS_ERROR")
        if completion.get("status") == "NATIVE_COMPLETED":
            log = (run_dir / "native_ego_hpp_sizing_base.log").read_text(encoding="utf8", errors="replace")
            outcome["termination_reason"] = "NATIVE_CONVERGENCE" if "Surrogate based optimization is converged." in log else "MAX_ITERATIONS"
    records = read_evaluations(run_dir / "evaluations", outcome["termination_reason"])
    history_path = run_dir / "history_hpp_sizing_base.csv"
    write_evaluation_history(records, history_path, outcome["started_at"],
                             outcome["started_at"] + outcome["optimization_seconds"],
                             config["hpp_sizing_objective"], config["reliability_target"])
    print("Search stopped:", outcome["termination_reason"], "-- independently validating saved candidates.", flush=True)
    best, validation_errors, validation_seconds = independently_validate(
        records, inputs, run_dir, config["hpp_sizing_objective"])
    result = dict(outcome, configuration=config, run_directory=str(run_dir), history_path=str(history_path),
                  best=best, final_validation_seconds=validation_seconds, validation_errors=validation_errors,
                  error=completion.get("error"), evaluated_designs=sum("completed_at" in r for r in records),
                  attempted_evaluations=len(records), unique_designs=len({tuple(r["design"]) for r in records}),
                  feasible_evaluations=sum(r["status"] == "FEASIBLE" for r in records),
                  infeasible_evaluations=0,
                  below_target_evaluations=sum(r["status"] == "BELOW_TARGET" for r in records),
                  unresolved_evaluations=sum(r["status"] not in ("FEASIBLE", "BELOW_TARGET") for r in records),
                  final_status="VALIDATED_FEASIBLE_INCUMBENT" if best else "NO_VALIDATED_FEASIBLE_SOLUTION_FOUND",
                  globally_optimal=False)
    result["time_limited_incumbent"] = best is not None and outcome["termination_reason"] == "TIME_LIMIT"
    if hashlib.sha256(native_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Native Parallel_EGO.py changed during the run")
    write_json_atomic(run_dir / "result_hpp_sizing_base.json", result)
    return result


def run_lcoe_reliability_experiment(config_path, overrides, reliability_targets, output_dir):
    """Sequential native EGO runs; retain native logs and checkpoints per target.

    Only the active new experiment's summary/figures are updated. Earlier
    experiments are never reused, overwritten or deleted. All times are UTC.
    """
    import matplotlib.pyplot as plt
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import (
        CAPACITY_VARIABLES, LCOE_OBJECTIVE, objective_bound, write_json_atomic,
    )
    config = prepare_configuration(config_path, overrides)
    config["hpp_sizing_objective"] = LCOE_OBJECTIVE
    if config["facility_load_mw"] != 16 or config["pue"] != 1.15:
        raise ValueError("This experiment requires 16 MW facility load and PUE 1.15")
    if any(config["variables"][name]["var_type"] != "design" for name in CAPACITY_VARIABLES):
        raise ValueError("Activate all four HPP capacity variables for the frontier")
    cases = []
    for target in reliability_targets:
        target = float(target)
        objective_bound(dict(config, reliability_target=target), 1., LCOE_OBJECTIVE)
        label = f"{100*target:07.3f}".rstrip("0").rstrip(".").replace(".", "p")
        cases.append({"reliability_target": target, "folder": "reliability_" + label})
    if not cases or len({c["folder"] for c in cases}) != len(cases):
        raise ValueError("Provide nonempty distinct targets (at least 0.001 percentage point apart)")
    directory = _output_path(Path(output_dir) / (
        datetime.now(timezone.utc).strftime("experiment_%Y%m%d_%H%M%SZ_") + uuid4().hex[:8] + "_max_reliability"))
    directory.mkdir(parents=True, exist_ok=False)
    physical = physical_inputs(config_path, config, directory)
    manifest = {"configuration": config, "cases": cases, "config_path": str(Path(config_path).resolve()),
                "physical_inputs": physical, "timestamp_timezone": "UTC",
                "native_ego_sha256": hashlib.sha256((ROOT / "hydesign/Parallel_EGO.py").read_bytes()).hexdigest(),
                "weather_sha256": hashlib.sha256(Path(physical["input_ts_fn"]).read_bytes()).hexdigest()}
    write_json_atomic(directory / "experiment_config.json", manifest)
    _, fig = _frontier_report(directory, directory)
    plt.close(fig)
    print("LCoE experiment:", directory, flush=True)
    for case in cases:
        target = case["reliability_target"]
        case_dir = _output_path(directory / case["folder"])
        case_dir.mkdir(exist_ok=False)
        effective = deepcopy(config)
        effective.update(reliability_target=target, hpp_sizing_output_dir=str(case_dir))
        # Record the exact resolved weather/location as well as all economic/search assumptions.
        resolved = physical_inputs(config_path, effective, case_dir)
        effective.update({k: resolved[k] for k in ("input_ts_fn", "latitude", "longitude", "altitude")})
        write_json_atomic(case_dir / "configuration.json", effective)
        try:
            result = run_native_hpp_sizing(config_path, overrides=effective)
        except KeyboardInterrupt:
            result = {"best": None, "final_status": "INTERRUPTED", "termination_reason": "USER_INTERRUPTED"}
        except Exception as exc:
            result = {"best": None, "final_status": "RUN_ERROR", "termination_reason": "ERROR",
                      "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("configuration", effective)
        result.update(_frontier_row(result, target))
        write_json_atomic(case_dir / "result.json", result)
        if result.get("history_path"):
            shutil.copyfile(result["history_path"], case_dir / "evaluations.csv")
        else:
            with (case_dir / "evaluations.csv").open("w", newline="", encoding="utf8") as stream:
                csv.writer(stream).writerow(HISTORY_COLUMNS)
        _, fig = _frontier_report(directory, directory)
        plt.close(fig)
        print(f"Target {100*target:g}%: {result['feasibility_status']}; "
              f"LCoE={result['lcoe_hpp_eur_per_mwh']}", flush=True)
        if result["termination_reason"] == "USER_INTERRUPTED":
            break
    return directory



if __name__ == "__main__":
    _worker(*sys.argv[1:])
