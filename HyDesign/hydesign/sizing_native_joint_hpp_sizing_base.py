"""Joint HPP/DC native Parallel EGO: configuration, containment and reporting.

Search and dispatch are unchanged. The supervisor reuses baseline process
control and physical validation; recovery ranks completed candidates by value.
"""
from copy import deepcopy
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

from hydesign.sizing_native_hpp_sizing_base import (
    ROOT, NATIVE_SETTINGS, HISTORY_COLUMNS, read_configuration,
    prepare_configuration as prepare_hpp_configuration, physical_inputs,
    run_contained, read_evaluations, independently_validate as validate_fixed_candidate,
)


def prepare_configuration(config_path, overrides=None):
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_sizing_base import (
        IT_NAME, validate_it_variable,
    )
    config = read_configuration(config_path)
    config.update(deepcopy(overrides or {}))
    pue = config["pue"]
    if not math.isfinite(pue) or pue < 1:
        raise ValueError("PUE must be finite and at least one")
    variables = config["variables"]
    it_variable = variables.pop(IT_NAME, {
        "var_type": "design", "types": "float",
        "limits": [config["facility_load_min_mw"] / pue, config["facility_load_max_mw"] / pue],
        "value": config["facility_load_mw"] / pue,
    })
    validate_it_variable(it_variable)
    config = prepare_hpp_configuration(config_path, config)
    config["variables"][IT_NAME] = it_variable
    return config


def _worker(request_path, gate_path):
    # Only standard-library imports before Windows Job Object assignment.
    while not Path(gate_path).exists():
        time.sleep(.02)
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_sizing_base import (
        hpp_model_simple_datacenter_joint_hpp_sizing_base, SCORE_NAME, VARIABLE_NAMES,
        write_json_atomic,
    )
    request = json.loads(Path(request_path).read_text(encoding="utf8"))
    directory = Path(request["run_directory"])
    config = request["configuration"]
    inputs = dict(request["physical_inputs"], variables=config["variables"], num_batteries=1,
                  baseline_log_dir=str(directory / "evaluations"))
    try:
        if any(v["var_type"] == "design" for v in config["variables"].values()):
            inputs.update({k: config[k] for k in NATIVE_SETTINGS})
            inputs.update(hpp_model=hpp_model_simple_datacenter_joint_hpp_sizing_base,
                          opt_var=SCORE_NAME,
                          final_design_fn=str(directory / "native_proposal_joint_hpp_sizing_base.csv"))
            ego = EfficientGlobalOptimizationDriver(**inputs)
            ego.run()
            write_json_atomic(directory / "native_recorder_joint_hpp_sizing_base.json", ego.recorder)
            result = {"status": "NATIVE_COMPLETED"}
        else:
            model = hpp_model_simple_datacenter_joint_hpp_sizing_base(**inputs)
            try:
                model.evaluate(*[config["variables"][name]["value"] for name in VARIABLE_NAMES])
            finally:
                model.prob.cleanup()
            result = {"status": "FIXED_DESIGN_EVALUATED"}
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "NATIVE_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(directory / "completion_joint_hpp_sizing_base.json", result)


def accepted_record(record):
    economics = record.get("economics") or {}
    value = economics.get("net_annual_value_system")
    return (record["status"] == "FEASIBLE" and record.get("validation") is not None
            and value is not None and math.isfinite(value)
            and record.get("annualized_cost_hpp") is not None
            and math.isfinite(record["annualized_cost_hpp"]) and record["annualized_cost_hpp"] >= 0)


def write_evaluation_history(records, path, started_at, ended_at):
    """Baseline CSV columns plus DC size and the value-ranked incumbent.

    best_feasible_cost is the HPP cost of the VALUE incumbent, not the minimum
    cost seen. Rejections have blank economics, never fictitious zero revenue.
    """
    columns = (*HISTORY_COLUMNS, "it_capacity_mw", "facility_load_mw",
               "best_feasible_net_annual_value_system", "best_feasible_it_capacity_mw")
    temporary = Path(path).with_suffix(".tmp")
    best = None
    with temporary.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, record in enumerate(sorted(records, key=lambda r: r.get("completed_at", ended_at)), 1):
            d = record["design"]
            accepted = accepted_record(record)
            e = record["economics"] if accepted else {}
            m = (record.get("metrics") or record["validation"]["metrics"]) if accepted else {}
            if accepted and (best is None or e["net_annual_value_system"] > best["economics"]["net_annual_value_system"]):
                best = record
            writer.writerow({
                "evaluation_index": index,
                "elapsed_seconds": record.get("completed_at", ended_at) - started_at,
                "status": record["status"], "accepted": accepted,
                "Nwt": d[3], "wind_MW": d[3]*d[2], "solar_MW": d[5],
                "b_P": d[9], "b_E_h": d[10], "b_E": d[9]*d[10],
                "it_capacity_mw": d[11], "facility_load_mw": record.get("facility_load_mw"),
                "score": record.get("score_meur_per_year"),
                "annualized_cost_hpp": e.get("annualized_cost_hpp"),
                "EENS": m.get("EENS"), "reliability": m.get("firm_reliability"), "failed_hours": m.get("LOLH"),
                **{k: e.get(k) for k in ("Revenue_dc", "annualized_cost_dc",
                                        "net_annual_value_system", "net_annual_value_dc_only",
                                        "G_proxy_MW", "CAPEX_sh", "CAPEX_sh_fixed", "CAPEX_sh_land",
                                        "annualized_cost_shared", "CAPEX_hpp",
                                        "CAPEX_b", "battery_replacement_cost", "discounted_battery_replacement_cost",
                                        "annualized_CAPEX_battery", "annualized_cost_battery")},
                "best_feasible_cost": None if best is None else best["annualized_cost_hpp"],
                "best_feasible_design": None if best is None else json.dumps(best["design"]),
                "best_feasible_net_annual_value_system": None if best is None else best["economics"]["net_annual_value_system"],
                "best_feasible_it_capacity_mw": None if best is None else best["design"][-1],
                "error": record.get("error"), "solver_observations": json.dumps(record.get("solver", [])),
                "optimization_method": "native_ego", "surrogate_diagnostics": "{}",
            })
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def independently_validate(records, inputs, run_dir):
    """Re-evaluate in descending system value with each candidate's actual load.

    Pass ONE eleven-variable candidate to the unchanged physical validator.
    Its cost sorting therefore cannot override joint system-value ordering.
    """
    import numpy as np
    import pandas as pd
    attempts, seen = [], set()
    start = time.monotonic()
    eligible = sorted(filter(accepted_record, records),
                      key=lambda r: r["economics"]["net_annual_value_system"], reverse=True)
    for record in eligible:
        design = record["design"]
        if tuple(design) in seen:
            continue
        seen.add(tuple(design))
        candidate_inputs = dict(inputs, facility_load_mw=design[-1] * inputs["pue"])
        fixed_record = dict(record, design=design[:-1])
        best, errors, _ = validate_fixed_candidate([fixed_record], candidate_inputs, run_dir)
        attempts.extend(dict(error, design=design) for error in errors)
        if best is None:
            continue
        try:
            np.testing.assert_allclose(best["economics"]["it_capacity_mw"], design[-1], rtol=0, atol=1e-8)
            np.testing.assert_allclose(
                best["economics"]["net_annual_value_system"], record["economics"]["net_annual_value_system"],
                rtol=1e-6, atol=1.0,
            )
        except AssertionError as exc:
            attempts.append({"design": design, "error": str(exc)})
            continue
        best.update(design=design, checkpoint_completed_at=record["completed_at"])
        hourly_path = Path(run_dir) / "hourly_hpp_sizing_base.csv"
        hourly = pd.read_csv(hourly_path)
        hourly["it_capacity_mw"] = design[-1]
        hourly["pue"] = inputs["pue"]
        hourly.to_csv(Path(run_dir) / "hourly_joint_hpp_sizing_base.csv", index=False)
        hourly_path.unlink()
        soc_path = Path(run_dir) / "soc_hpp_sizing_base.npy"
        if soc_path.exists():
            soc_path.replace(Path(run_dir) / "soc_joint_hpp_sizing_base.npy")
        return best, attempts, time.monotonic() - start
    # Do not leave an apparently accepted hourly export after failed recovery.
    for filename in ("hourly_hpp_sizing_base.csv", "soc_hpp_sizing_base.npy"):
        (Path(run_dir) / filename).unlink(missing_ok=True)
    return None, attempts, time.monotonic() - start


def plot_history(result):
    """Save compact history and capacity plots; return the figure for notebooks."""
    import matplotlib.pyplot as plt
    import pandas as pd
    history = pd.read_csv(result["history_path"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    axes[0].step(history["elapsed_seconds"], history["best_feasible_net_annual_value_system"]/1e6, where="post")
    axes[0].set(xlabel="Elapsed optimization time [s]", ylabel="Best system value [MEUR/year]")
    axes[1].step(history["evaluation_index"], history["best_feasible_net_annual_value_system"]/1e6, where="post")
    axes[1].set(xlabel="Evaluation (completion order)", ylabel="Best system value [MEUR/year]")
    feasible = history["accepted"].astype(str).str.lower() == "true"
    axes[2].scatter(history.loc[feasible, "it_capacity_mw"],
                    history.loc[feasible, "net_annual_value_system"]/1e6, s=18, label="Feasible")
    # Invalid candidates have no economic result: show them on an axes-relative strip.
    rejected = history[~feasible]
    for status, group in rejected.groupby("status"):
        axes[2].plot(group["it_capacity_mw"], [0.02]*len(group), "|",
                     transform=axes[2].get_xaxis_transform(), label=status)
    axes[2].set(xlabel="Installed IT capacity [MW IT]", ylabel="System value [MEUR/year]")
    if len(history):
        axes[2].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(Path(result["run_directory"]) / "history_joint_hpp_sizing_base.png", dpi=160)
    return fig


def print_results(result):
    """Keep notebook presentation short; all values here are unshifted economics."""
    cfg, best = result["configuration"], result["best"]
    print("Final status:", result["final_status"], "| Search:", result["termination_reason"])
    if best is not None:
        d, e, m = best["design"], best["economics"], best["metrics"]
        print("\nHPP")
        print(f"Turbines: {d[3]:g} | Wind: {d[3]*d[2]:.3f} MW | PV: {d[5]:.3f} MW AC")
        print(f"Battery: {d[9]:.3f} MW / {d[9]*d[10]:.3f} MWh / {d[10]:.3f} h")
        print("\nData center")
        print(f"IT: {d[11]:.4f} MW IT | Facility: {e['facility_load_mw']:.4f} MW | PUE: {cfg['pue']:g}")
        print(f"Annual IT demand: {d[11]*8760:,.3f} MWh | Served: {e['served_it_energy_mwh']:,.3f} MWh")
        print(f"Facility EENS: {m['EENS']:,.3f} MWh | LOLH: {m['LOLH']} h")
        print(f"Reliability: {100*m['firm_reliability']:.5f}% | Target: {100*cfg['reliability_target']:g}%")
        print("IT capacity bounds:", result["it_bound_position"])
        print("\nEconomics")
        print(f"Initial battery CAPEX: {e['CAPEX_b']:,.2f} EUR")
        print(f"Battery replacement (year {cfg['battery_replacement_year']:g}): {e['battery_replacement_cost']:,.2f} EUR")
        print(f"Discounted battery replacement: {e['discounted_battery_replacement_cost']:,.2f} EUR at year 0")
        print(f"Annualized battery CAPEX / cost: {e['annualized_CAPEX_battery']:,.2f} / {e['annualized_cost_battery']:,.2f} EUR/year")
        print(f"Shared capacity proxy (facility + rated battery): {e['G_proxy_MW']:.3f} MW")
        for key in ("CAPEX_sh_fixed", "CAPEX_sh_land", "CAPEX_sh"):
            print(f"{key}: {e[key]:,.2f} EUR")
        print(f"annualized_cost_shared: {e['annualized_cost_shared']:,.2f} EUR/year")
        for key in ("Revenue_dc", "annualized_cost_dc", "annualized_cost_hpp"):
            print(f"{key}: {e[key]:,.2f} EUR/year ({e[key]/1e6:,.4f} MEUR/year)")
        print(f"HPP CAPEX: {e['CAPEX_hpp']/1e6:,.3f} MEUR | DC CAPEX: {e['CAPEX_dc_total']/1e6:,.3f} MEUR")
        print(f"NET ANNUAL SYSTEM VALUE: {e['net_annual_value_system']/1e6:,.4f} MEUR/year")
        stages = best["solver_status"].get("stages", [])
        print("Independent validation: passed | CPLEX:",
              "; ".join(str(stage.get("status")) for stage in stages))
    print("\nOptimization")
    print(f"Evaluations: {result['attempted_evaluations']} | Feasible: {result['feasible_evaluations']} | "
          f"Proven infeasible: {result['infeasible_evaluations']} | Other/interrupted: {result['unresolved_evaluations']}")
    print("Time to recovered best [s]:", result["time_to_best_seconds"])
    print(f"Optimization: {result['optimization_seconds']:.2f} s | Final validation: {result['final_validation_seconds']:.2f} s | "
          f"Total: {result['total_seconds']:.2f} s")
    print(f"Seed: {cfg['n_seed']} | DOE: {cfg['n_doe']} | Workers: {cfg['n_procs']} | Clusters: {cfg['n_clusters']}")
    if result.get("error"):
        print("Native driver error:", result["error"])
    print("Saved results:", result["run_directory"])
    print("A feasible incumbent is not a certificate of global optimality.")


def run_native_joint_hpp_sizing(config_path, overrides=None):
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_sizing_base import IT_NAME, write_json_atomic
    start = time.monotonic()
    config = prepare_configuration(config_path, overrides)
    run_dir = Path(config["hpp_sizing_output_dir"]).resolve() / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8] + "_joint_hpp_sizing_base")
    (run_dir / "evaluations").mkdir(parents=True)
    inputs = physical_inputs(config_path, config, run_dir)
    native_path = ROOT / "hydesign/Parallel_EGO.py"
    digest = hashlib.sha256(native_path.read_bytes()).hexdigest()
    request_path = run_dir / "configuration_joint_hpp_sizing_base.json"
    write_json_atomic(request_path, {"configuration": config, "physical_inputs": inputs,
                                    "run_directory": str(run_dir), "native_ego_sha256": digest})
    gate_path = run_dir / "start_joint_hpp_sizing_base.flag"
    print("Joint native EGO results:", run_dir, flush=True)
    print("Outer limit [s]:", config["hpp_sizing_time_limit_seconds"], flush=True)
    command = [sys.executable, "-B", "-u", "-m", "hydesign.sizing_native_joint_hpp_sizing_base",
               str(request_path), str(gate_path)]
    outcome = run_contained(command, run_dir, config["hpp_sizing_time_limit_seconds"], gate_path)
    completion_path = run_dir / "completion_joint_hpp_sizing_base.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    if outcome["termination_reason"] == "NORMAL_EXIT":
        outcome["termination_reason"] = completion.get("status", "PROCESS_ERROR")
        if completion.get("status") == "NATIVE_COMPLETED":
            log = (run_dir / "native_ego_hpp_sizing_base.log").read_text(encoding="utf8", errors="replace")
            outcome["termination_reason"] = "NATIVE_CONVERGENCE" if "Surrogate based optimization is converged." in log else "MAX_ITERATIONS"
    records = read_evaluations(run_dir / "evaluations", outcome["termination_reason"])
    history_path = run_dir / "history_joint_hpp_sizing_base.csv"
    write_evaluation_history(records, history_path, outcome["started_at"],
                             outcome["started_at"] + outcome["optimization_seconds"])
    print("Search stopped:", outcome["termination_reason"], "-- independently validating value-ranked candidates.", flush=True)
    best, errors, seconds = independently_validate(records, inputs, run_dir)
    statuses = Counter(r["status"] for r in records)
    result = dict(outcome, configuration=config, run_directory=str(run_dir), history_path=str(history_path),
                  best=best, final_validation_seconds=seconds, validation_errors=errors, error=completion.get("error"),
                  attempted_evaluations=len(records), evaluated_designs=sum("completed_at" in r for r in records),
                  unique_designs=len({tuple(r["design"]) for r in records}),
                  feasible_evaluations=sum(accepted_record(r) for r in records),
                  infeasible_evaluations=statuses["PROVEN_INFEASIBLE"], status_counts=dict(statuses),
                  final_status="VALIDATED_FEASIBLE_INCUMBENT" if best else "NO_VALIDATED_FEASIBLE_SOLUTION_FOUND",
                  globally_optimal=False, it_bound_position=None, time_to_best_seconds=None)
    result["unresolved_evaluations"] = len(records)-result["feasible_evaluations"]-result["infeasible_evaluations"]
    result["time_limited_incumbent"] = best is not None and outcome["termination_reason"] == "TIME_LIMIT"
    if best:
        result["time_to_best_seconds"] = best["checkpoint_completed_at"] - outcome["started_at"]
        var, capacity = config["variables"][IT_NAME], best["design"][-1]
        if var["var_type"] == "fixed":
            position = "fixed"
        else:
            lo, hi = var["limits"]
            tolerance = max(1e-6, .01*(hi-lo))
            position = "near lower bound (within 1% of range)" if capacity-lo <= tolerance else (
                "near upper bound (within 1% of range)" if hi-capacity <= tolerance else "interior")
        result["it_bound_position"] = position
    import matplotlib.pyplot as plt
    plt.close(plot_history(result))
    result["total_seconds"] = time.monotonic()-start
    if hashlib.sha256(native_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Native Parallel_EGO.py changed during the run")
    write_json_atomic(run_dir / "result_joint_hpp_sizing_base.json", result)
    return result


if __name__ == "__main__":
    _worker(*sys.argv[1:])

