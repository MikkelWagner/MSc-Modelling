"""Run original Parallel EGO in a contained process; recover validated checkpoints.

This module controls execution, not candidate selection. Parallel_EGO and its
evaluator are imported unchanged. The outer budget includes child startup, DOE,
surrogates and dispatch. Process cleanup and independent final validation have
separate timings. Windows requires pywin32 (present in hydesign_clean).
"""

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4


NATIVE_SETTINGS = ("n_procs", "n_doe", "n_clusters", "npred", "n_seed",
                   "max_iter", "tol", "min_conv_iter")
ROOT = Path(__file__).resolve().parent.parent
HISTORY_COLUMNS = (
    "evaluation_index", "elapsed_seconds", "status", "accepted", "Nwt", "wind_MW",
    "solar_MW", "b_P", "b_E_h", "b_E", "score", "annualized_cost_hpp", "EENS",
    "reliability", "failed_hours", "Revenue_dc", "annualized_cost_dc",
    "net_annual_value_system", "net_annual_value_dc_only", "best_feasible_cost",
    "best_feasible_design", "error", "solver_observations", "optimization_method",
    "proposal_id", "proposed_elapsed_seconds", "attempt", "retry", "selection_reason",
    "search_iteration", "predicted_annualized_cost_hpp", "predicted_cost_std_eur",
    "predicted_feasibility_probability", "expected_improvement_eur", "acquisition_score",
    "surrogate_diagnostics",
    "G_proxy_MW", "CAPEX_sh", "CAPEX_sh_fixed", "CAPEX_sh_land",
    "annualized_cost_shared", "CAPEX_hpp",
    "CAPEX_b", "battery_replacement_cost", "discounted_battery_replacement_cost",
    "annualized_CAPEX_battery", "annualized_cost_battery",
    "objective", "lcoe_hpp_eur_per_mwh", "served_facility_energy_mwh",
    "best_feasible_lcoe", "reliability_target", "solver_status",
)


def read_configuration(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf8"))


def _output_path(path):
    """Allow nested experiment checkpoints beyond Windows' legacy 260-character limit."""
    path = Path(path).resolve()
    value = str(path)
    if os.name == "nt" and len(value) > 180 and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def prepare_configuration(config_path, overrides=None):
    """Resolve YAML plus explicit overrides, without selecting sizing flags for users."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import (
        CAPACITY_VARIABLES, VARIABLE_NAMES, COST_OBJECTIVE, objective_bound,
    )
    config = read_configuration(config_path)
    config.update(deepcopy(overrides or {}))
    objective = config.setdefault("hpp_sizing_objective", COST_OBJECTIVE)
    objective_bound(config, 1.0, objective)  # Validate the objective before spawning workers.
    limit = config.get("hpp_sizing_time_limit_seconds")
    if limit is not None and (isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0):
        raise ValueError("hpp_sizing_time_limit_seconds must be positive, or None to disable")
    variables = config["variables"]
    if set(variables) != set(VARIABLE_NAMES):
        raise ValueError("Variable names must match the fixed-load assembly")
    for name, var in variables.items():
        if var["var_type"] not in ("fixed", "design"):
            raise ValueError(f"Invalid var_type: {name}")
        if var["var_type"] == "design":
            if name not in CAPACITY_VARIABLES:
                raise ValueError(f"Only the four HPP capacity variables can be optimized: {name}")
            lo, hi = var["limits"]
            if not all(math.isfinite(x) for x in (lo, hi)) or not 0 <= lo < hi:
                raise ValueError(f"Invalid design bounds: {name}")
            if var["types"] != ("int" if name == "Nwt" else "float"):
                raise ValueError(f"Invalid native variable type: {name}")
            if name == "Nwt" and any(x != round(x) for x in (lo, hi)):
                raise ValueError("Turbine-count limits must be integers")
        else:
            value = var["value"]
            if not math.isfinite(value) or value < 0 or (name == "Nwt" and value != round(value)):
                raise ValueError(f"Invalid fixed value: {name}")
    for name in NATIVE_SETTINGS:
        value = config[name]
        minimum = 0 if name in ("max_iter", "n_seed") else 1
        if name == "tol":
            if not math.isfinite(value) or value <= 0:
                raise ValueError("tol must be positive and finite")
        elif isinstance(value, bool) or not math.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"Invalid native setting: {name}")
        else:
            config[name] = int(value)
    # Method switches and custom search controls have no role in this baseline.
    return {k: v for k, v in config.items()
            if k != "optimization_method" and not k.startswith("fa_")}


def physical_inputs(config_path, config, run_dir):
    """Forward ALL effective model assumptions, including notebook economic overrides."""
    import pandas as pd
    sites = pd.read_csv(ROOT / "hydesign/examples/examples_sites.csv", sep=";")
    site = sites.loc[sites["name"] == config["hpp_sizing_site"]].iloc[0]
    inputs = {k: v for k, v in config.items()
              if k not in NATIVE_SETTINGS and k != "variables" and not k.startswith("hpp_sizing_")}
    inputs.update({
        "sim_pars_fn": str(Path(config_path).resolve()),
        "input_ts_fn": str(Path(config.get("input_ts_fn") or
                               ROOT / "hydesign/examples" / site["input_ts_fn"]).resolve()),
        "latitude": float(config.get("latitude", site["latitude"])),
        "longitude": float(config.get("longitude", site["longitude"])),
        "altitude": float(config.get("altitude", site["altitude"])),
        "name": config["hpp_sizing_site"], "work_dir": str(run_dir) + os.sep,
    })
    inputs.pop("verbose", None)  # Native model_evaluation supplies verbose=False.
    return inputs


def run_contained(command, run_dir, limit_seconds, gate_path):
    """Launch gated child; kill the entire job at deadline and verify it is empty.

    The child waits for gate_path before importing/running HyDesign. It cannot
    spawn Pool workers until job assignment succeeds. KILL_ON_JOB_CLOSE also
    contains children if the notebook process itself exits unexpectedly.
    """
    job = None
    if os.name == "nt":
        import win32api
        import win32con
        import win32job
        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    env = os.environ.copy()
    env.update(OPENMDAO_REPORTS="0", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    process = None
    start = time.monotonic()
    started_at = time.time()
    reason = "NORMAL_EXIT"
    cleanup = {}
    try:
        with (Path(run_dir) / "native_ego_hpp_sizing_base.log").open("w", encoding="utf8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            if job is not None:
                handle = win32api.OpenProcess(
                    win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, process.pid)
                try:
                    win32job.AssignProcessToJobObject(job, handle)
                finally:
                    handle.Close()
            Path(gate_path).touch()  # No child work before containment succeeds.
            remaining = None if limit_seconds is None else max(0, limit_seconds - (time.monotonic() - start))
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                reason = "TIME_LIMIT"
            except KeyboardInterrupt:
                reason = "USER_INTERRUPTED"
    finally:
        optimization_seconds = time.monotonic() - start
        cleanup_start = time.monotonic()
        try:
            if job is not None:
                # Also clean up any descendants left after a normal driver exit.
                win32job.TerminateJobObject(job, 124 if reason == "TIME_LIMIT" else 1)
                while True:
                    info = win32job.QueryInformationJobObject(job, win32job.JobObjectBasicAccountingInformation)
                    cleanup = {"mechanism": "Windows Job Object", "total_processes": info["TotalProcesses"],
                               "active_processes_after_stop": info["ActiveProcesses"]}
                    if info["ActiveProcesses"] == 0:
                        break
                    if time.monotonic() - cleanup_start > 10:
                        raise RuntimeError("Windows job did not become empty after termination")
                    time.sleep(.02)
            elif process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                cleanup = {"mechanism": "POSIX process group"}
        finally:
            if job is not None:
                job.Close()
            if process is not None:
                # Covers assignment failure too: the unassigned process is still gated.
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
    return {"termination_reason": reason, "optimization_seconds": optimization_seconds,
            "started_at": started_at, "cleanup_seconds": time.monotonic() - cleanup_start,
            "process_cleanup": cleanup, "returncode": process.returncode}


def _worker(request_path, gate_path):
    # Deliberately no scientific imports before the supervisor assigns the job.
    while not Path(gate_path).exists():
        time.sleep(.02)
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import (
        hpp_model_simple_datacenter_hpp_sizing_base, VARIABLE_NAMES, write_json_atomic, objective_name,
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
            inputs.update(hpp_model=hpp_model_simple_datacenter_hpp_sizing_base,
                          opt_var=objective_name(config["hpp_sizing_objective"]),
                          final_design_fn=str(directory / "native_proposal_hpp_sizing_base.csv"))
            ego = EfficientGlobalOptimizationDriver(**inputs)
            ego.run()
            write_json_atomic(directory / "native_recorder_hpp_sizing_base.json", ego.recorder)
            result = {"status": "NATIVE_COMPLETED"}
        else:
            model = hpp_model_simple_datacenter_hpp_sizing_base(**inputs)
            try:
                model.evaluate(*[config["variables"][name]["value"] for name in VARIABLE_NAMES])
            finally:
                model.prob.cleanup()
            result = {"status": "FIXED_DESIGN_EVALUATED"}
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "NATIVE_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(directory / "completion_hpp_sizing_base.json", result)


def read_evaluations(directory, termination_reason):
    """Read only atomically published JSON; unfinished calls are never infeasible."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import write_json_atomic
    records = []
    for path in Path(directory).glob("*_hpp_sizing_base.json"):
        record = json.loads(path.read_text(encoding="utf8"))
        if record["status"] == "RUNNING":
            record.update(status="TIME_LIMIT_INTERRUPTED" if termination_reason == "TIME_LIMIT" else "INTERRUPTED",
                          error=f"Evaluation interrupted: {termination_reason}")
            write_json_atomic(path, record)
        records.append(record)
    return sorted(records, key=lambda r: (r.get("completed_at", float("inf")), r["started_at"]))


def write_evaluation_history(records, path, started_at, ended_at,
                             objective="minimize_annualized_cost", reliability_target=None):
    """Match the earlier HPP-sizing CSV layout without importing its search code.

    Completion order defines the running incumbent. Rejected/interrupted calls
    have no accepted economics. Native EGO does not expose proposal/prediction
    metadata to this adapter, so those columns remain blank (diagnostics = {}).
    Older checkpoints lacking economics can still be exported with blank cells.
    """
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    records = sorted(records, key=lambda r: (r.get("completed_at", ended_at), r["started_at"]))
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import objective_value, LCOE_OBJECTIVE
    best_cost = best_design = best_value = best_lcoe = None
    with temporary.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            design = record["design"]
            cost = record.get("annualized_cost_hpp")
            accepted = (record["status"] == "FEASIBLE" and record.get("validation") is not None
                        and cost is not None and math.isfinite(cost) and cost >= 0)
            economics = (record.get("economics") or {}) if accepted else {}
            metrics = (record.get("metrics") or record["validation"].get("metrics", {})) if accepted else {}
            lcoe = (objective_value(economics, LCOE_OBJECTIVE)
                    if economics.get("served_facility_energy_mwh", 0) > 0 else None)
            value = lcoe if objective == LCOE_OBJECTIVE else cost
            accepted = accepted and value is not None and math.isfinite(value)
            if accepted and (best_value is None or value < best_value):
                best_value, best_cost, best_design, best_lcoe = value, cost, design, lcoe
            row = {
                "evaluation_index": index,
                "elapsed_seconds": record.get("completed_at", ended_at) - started_at,
                "status": record["status"], "accepted": accepted,
                "Nwt": int(design[3]) if float(design[3]).is_integer() else design[3],
                "wind_MW": design[3] * design[2], "solar_MW": design[5],
                "b_P": design[-2], "b_E_h": design[-1], "b_E": design[-2] * design[-1],
                "score": record.get("score", record.get("score_meur_per_year")),
                "objective": objective, "lcoe_hpp_eur_per_mwh": lcoe,
                "served_facility_energy_mwh": economics.get("served_facility_energy_mwh"),
                "best_feasible_lcoe": best_lcoe, "reliability_target": reliability_target,
                "solver_status": json.dumps(record.get("solver_status")),
                "annualized_cost_hpp": cost if accepted else None,
                "EENS": metrics.get("EENS"), "reliability": metrics.get("firm_reliability"),
                "failed_hours": metrics.get("LOLH"),
                "Revenue_dc": economics.get("Revenue_dc"),
                "annualized_cost_dc": economics.get("annualized_cost_dc"),
                "net_annual_value_system": economics.get("net_annual_value_system"),
                "net_annual_value_dc_only": economics.get("net_annual_value_dc_only"),
                "best_feasible_cost": best_cost,
                "best_feasible_design": None if best_design is None else json.dumps(best_design),
                "error": record.get("error"),
                "solver_observations": json.dumps(record.get("solver", [])),
                "optimization_method": "native_ego", "surrogate_diagnostics": "{}",
                **{k: economics.get(k) for k in ("G_proxy_MW", "CAPEX_sh", "CAPEX_sh_fixed",
                                                "CAPEX_sh_land", "annualized_cost_shared", "CAPEX_hpp",
                                                "CAPEX_b", "battery_replacement_cost", "discounted_battery_replacement_cost",
                                                "annualized_CAPEX_battery", "annualized_cost_battery")},
            }
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def independently_validate(records, inputs, run_dir, objective="minimize_annualized_cost"):
    """Recover in selected-objective order, independently rechecking physics and energy."""
    import numpy as np
    import pandas as pd
    from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost import hpp_model_simple_datacenter_with_hpp_cost
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import (
        validate_result, objective_value, LCOE_OBJECTIVE,
    )
    eligible = [r for r in records if r["status"] == "FEASIBLE"
                and r.get("validation") is not None and r.get("annualized_cost_hpp") is not None
                and math.isfinite(r["annualized_cost_hpp"]) and r["annualized_cost_hpp"] >= 0]
    attempts, seen = [], set()
    ranked = []
    for record in eligible:
        try:
            economics = record.get("economics") or {"annualized_cost_hpp": record["annualized_cost_hpp"]}
            ranked.append((objective_value(economics, objective), record))
        except (KeyError, ValueError, TypeError) as exc:
            attempts.append({"design": record["design"], "error": str(exc)})
    start = time.monotonic()
    for expected_objective, record in sorted(ranked, key=lambda item: item[0]):
        design = record["design"]
        if tuple(design) in seen:
            continue
        seen.add(tuple(design))
        model = None
        try:
            model = hpp_model_simple_datacenter_with_hpp_cost(**inputs, verbose=False)
            model.evaluate(*design)
            checks = validate_result(model, design)
            np.testing.assert_allclose(model.economics["annualized_cost_hpp"], record["annualized_cost_hpp"],
                                       rtol=1e-9, atol=1e-6)
            if objective == LCOE_OBJECTIVE:
                energy = model.economics["served_facility_energy_mwh"]
                np.testing.assert_allclose(energy, model.hourly["served_facility_mw"].sum(), rtol=1e-10, atol=1e-6)
                np.testing.assert_allclose(energy, model.sim_pars["facility_load_mw"]*8760 - checks["metrics"]["EENS"],
                                           rtol=1e-10, atol=1e-5)
                np.testing.assert_allclose(objective_value(model.economics, objective), expected_objective,
                                           rtol=1e-6, atol=1e-6)
            best = {"design": design, "metrics": checks["metrics"], "economics": model.economics,
                    "solver_status": model.solver_status, "validation": checks,
                    "lcoe_hpp_eur_per_mwh": (objective_value(model.economics, LCOE_OBJECTIVE)
                        if model.economics.get("served_facility_energy_mwh", 0) > 0 else None)}
            pd.DataFrame(model.hourly).to_csv(Path(run_dir) / "hourly_hpp_sizing_base.csv", index=False)
            np.save(Path(run_dir) / "soc_hpp_sizing_base.npy", model.soc_mwh)
            return best, attempts, time.monotonic() - start
        except Exception as exc:
            attempts.append({"design": design, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if model is not None:
                model.prob.cleanup()
    return None, attempts, time.monotonic() - start


def run_native_hpp_sizing(config_path, overrides=None):
    """Notebook entry point: configure, supervise native EGO, recover and validate."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_base import write_json_atomic
    config = prepare_configuration(config_path, overrides)
    run_dir = _output_path(Path(config["hpp_sizing_output_dir"]) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8] + "_hpp_sizing_base"))
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
    command = [sys.executable, "-B", "-u", "-m", "hydesign.sizing_native_hpp_sizing_base",
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
                  infeasible_evaluations=sum(r["status"] == "PROVEN_INFEASIBLE" for r in records),
                  unresolved_evaluations=sum(r["status"] not in ("FEASIBLE", "PROVEN_INFEASIBLE") for r in records),
                  final_status="VALIDATED_FEASIBLE_INCUMBENT" if best else "NO_VALIDATED_FEASIBLE_SOLUTION_FOUND",
                  globally_optimal=False)
    result["time_limited_incumbent"] = best is not None and outcome["termination_reason"] == "TIME_LIMIT"
    if hashlib.sha256(native_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Native Parallel_EGO.py changed during the run")
    write_json_atomic(run_dir / "result_hpp_sizing_base.json", result)
    return result


def _frontier_row(result, target):
    """One summary row; a missing/failed run never masquerades as a feasible point."""
    best = result.get("best") or {}
    metrics, economics = best.get("metrics", {}), best.get("economics", {})
    design = best.get("design")
    valid = bool(best and best.get("validation") is not None)
    return {
        "reliability_target": target, "target_reliability_percent": 100*target,
        "achieved_reliability_percent": 100*metrics["firm_reliability"] if valid else None,
        "lcoe_hpp_eur_per_mwh": best.get("lcoe_hpp_eur_per_mwh") if valid else None,
        "annualized_cost_hpp": economics.get("annualized_cost_hpp") if valid else None,
        "served_facility_energy_mwh": economics.get("served_facility_energy_mwh") if valid else None,
        "EENS": metrics.get("EENS") if valid else None,
        "LOLH": metrics.get("LOLH") if valid else None,
        "Nwt": design[3] if valid else None,
        "wind_MW": design[2]*design[3] if valid else None,
        "solar_MW": design[5] if valid else None,
        "b_P": design[-2] if valid else None,
        "b_E_h": design[-1] if valid else None,
        "b_E": design[-2]*design[-1] if valid else None,
        "optimization_seconds": result.get("optimization_seconds"),
        "final_validation_seconds": result.get("final_validation_seconds"),
        "evaluated_designs": result.get("evaluated_designs"),
        "attempted_evaluations": result.get("attempted_evaluations"),
        "unique_designs": result.get("unique_designs"),
        "feasible_evaluations": result.get("feasible_evaluations"),
        "termination_reason": result.get("termination_reason", "PENDING"),
        "feasibility_status": result.get("final_status", "PENDING"),
        "physical_validation_passed": valid,
        "run_directory": result.get("run_directory"), "error": result.get("error"),
    }


def _frontier_report(experiment_dir, report_dir=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    experiment_dir = _output_path(experiment_dir)
    manifest = json.loads((experiment_dir / "experiment_config.json").read_text(encoding="utf8"))
    rows = []
    for case in manifest["cases"]:
        path = experiment_dir / case["folder"] / "result.json"
        result = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
        rows.append(_frontier_row(result, case["reliability_target"]))
    table = pd.DataFrame(rows)
    usable = table[table["physical_validation_passed"] & table["lcoe_hpp_eur_per_mwh"].notna()]
    table["nondominated"] = False
    for index, row in usable.iterrows():
        dominates = ((usable["achieved_reliability_percent"] >= row["achieved_reliability_percent"])
                     & (usable["lcoe_hpp_eur_per_mwh"] <= row["lcoe_hpp_eur_per_mwh"])
                     & ((usable["achieved_reliability_percent"] > row["achieved_reliability_percent"])
                        | (usable["lcoe_hpp_eur_per_mwh"] < row["lcoe_hpp_eur_per_mwh"])))
        table.loc[index, "nondominated"] = not dominates.any()
    frontier = table[table["nondominated"]].sort_values("achieved_reliability_percent").drop_duplicates(
        ["achieved_reliability_percent", "lcoe_hpp_eur_per_mwh"])
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    if not usable.empty:
        ax.scatter(usable["achieved_reliability_percent"], usable["lcoe_hpp_eur_per_mwh"],
                   color="0.55", s=45, label="Best validated design per target", zorder=3)
        ax.plot(frontier["achieved_reliability_percent"], frontier["lcoe_hpp_eur_per_mwh"],
                "o--", color="#146482", label="Approximate observed Pareto frontier", zorder=4)
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "No validated feasible results yet", transform=ax.transAxes, ha="center")
    ax.set(xlabel="Achieved firm reliability (%)", ylabel="HPP LCoE (EUR/MWh)",
           title="Fixed 16 MW facility demand", xlim=(0, 100.5))
    ax.grid(alpha=.2)
    if report_dir is not None:
        report_dir = Path(report_dir)
        temporary = report_dir / "summary.tmp"
        table.to_csv(temporary, index=False)
        os.replace(temporary, report_dir / "summary.csv")
        fig.savefig(report_dir / "pareto_frontier.png", dpi=200)
        fig.savefig(report_dir / "pareto_frontier.pdf")
    return table, fig


def load_lcoe_reliability_experiment(experiment_dir, save_figures=False):
    """Rebuild from saved JSON only. Optional exports use a NEW reports folder."""
    report_dir = None
    if save_figures:
        report_dir = _output_path(Path(experiment_dir) / (
            datetime.now(timezone.utc).strftime("reports_%Y%m%d_%H%M%SZ_") + uuid4().hex[:8]))
        report_dir.mkdir(exist_ok=False)
    table, fig = _frontier_report(experiment_dir, report_dir)
    if report_dir is not None:
        print("Regenerated reports:", report_dir)
    return table, fig


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
    if config.get("battery_power_fluctuation_penalty") != .0001:
        raise ValueError("Keep the experiment's battery fluctuation penalty at 0.0001")
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
        datetime.now(timezone.utc).strftime("experiment_%Y%m%d_%H%M%SZ_") + uuid4().hex[:8]))
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
