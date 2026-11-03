"""Selectable HPP searches with shared fixed-DC validation and a time budget.

The native EGO search is unchanged. A scoped ParallelEvaluator replacement adds
deadline checks, seeds, a per-run cache and single-writer recording. Worker tasks
use Windows spawn. The deadline is best effort: no new candidate starts after it;
running CPLEX calls are capped by remaining time, but model construction, native
surrogate work, solver cleanup and final validation can overrun it.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd
import yaml

from hydesign import Parallel_EGO as native_ego
from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_sizing import (
    hpp_model_simple_datacenter_with_hpp_sizing, normalize_turbines, SCORE_NAME,
)
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.costs.cost_simple_hpp_with_hpp_cost import hpp_cost_annualized
from hydesign.examples import examples_filepath
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch
from hydesign.search_simple_datacenter_with_hpp_sizing import (
    UNKNOWN_STATUSES, run_feasibility_aware, search_settings,
)
from hydesign.wind.wind import get_rotor_d


VARIABLE_NAMES = (
    "clearance [m]", "sp [W/m2]", "p_rated [MW]", "Nwt",
    "wind_MW_per_km2 [MW/km2]", "solar_MW [MW AC]", "surface_tilt [deg]",
    "surface_azimuth [deg]", "DC_AC_ratio", "b_P [MW]", "b_E_h [h]",
)
SIZABLE = ("Nwt", "solar_MW [MW AC]", "b_P [MW]", "b_E_h [h]")
RUN_LOCK = threading.Lock()  # Scoped native evaluator binding cannot overlap in one process.


class BudgetExpired(RuntimeError):
    pass


class EvaluationLimitReached(RuntimeError):
    pass


def unique_evaluation_limit(config):
    limit = config.get("hpp_sizing_max_unique_evaluations")
    if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, (int, float))
            or not np.isfinite(limit) or int(limit) != limit or limit < 1):
        raise ValueError("hpp_sizing_max_unique_evaluations must be null or a positive integer")
    return None if limit is None else int(limit)


def read_configuration(path):
    with Path(path).open(encoding="utf8") as stream:
        return yaml.safe_load(stream)


def prepare_variables(config):
    variables = config["variables"]
    if set(variables) != set(VARIABLE_NAMES):
        raise ValueError("Variable names must match exactly: " + ", ".join(VARIABLE_NAMES))
    for name in VARIABLE_NAMES:
        v = variables[name]
        if v["var_type"] not in ("fixed", "design"):
            raise ValueError(f"Invalid var_type for {name}")
        if name not in SIZABLE and v["var_type"] != "fixed":
            raise ValueError(f"Geometry/technology must remain fixed: {name}")
        value = float(v["value"])
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite initial/fixed value: {name}")
        if name in SIZABLE:
            lower, upper = v["limits"]
            if not np.isfinite([lower, upper]).all() or not 0 <= lower < upper:
                raise ValueError(f"Invalid bounds for {name}")
            if not lower <= value <= upper:
                raise ValueError(f"Initial/fixed value outside bounds for {name}")
            required = "int" if name == "Nwt" else "float"
            if v["types"] != required:
                raise ValueError(f"{name} requires types={required}")
        if name == "Nwt":
            normalize_turbines(value)
            for bound in v["limits"]:
                normalize_turbines(bound)
    for name in ("sp [W/m2]", "p_rated [MW]", "wind_MW_per_km2 [MW/km2]", "DC_AC_ratio"):
        if variables[name]["value"] <= 0:
            raise ValueError(f"{name} must be positive")
    return {name: dict(variables[name]) for name in VARIABLE_NAMES}


def native_cost_bound(config, variables):
    """Conservative native cost bound, not a feasibility relaxation.

    Fixed geometry makes wind CAPEX linear in Nwt. Wind variable OPEX is bounded
    using continuous nameplate generation for all 8760 hours (no wake losses).
    PV and battery use their largest configured capacities. All coefficients must
    be nonnegative. Accepted candidate costs are checked against this bound.
    """
    def maximum(name):
        v = variables[name]
        return v["limits"][1] if v["var_type"] == "design" else v["value"]
    def costs(keys):
        values = {k: config[k] for k in keys}
        if any(not np.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("HPP cost coefficients must be finite and nonnegative")
        return values
    rated = maximum("p_rated [MW]")
    diameter = get_rotor_d(rated * 1e6 / maximum("sp [W/m2]"))
    count = maximum("Nwt")
    wind = wpp_cost(**costs(("wind_turbine_cost", "wind_civil_works_cost", "wind_fixed_onm_cost",
                            "wind_variable_onm_cost", "d_ref", "hh_ref", "p_rated_ref")), N_time=8760)
    cw, ow = wind.compute(count, diameter / 2 + maximum("clearance [m]"), diameter,
                          rated, np.full(8760, count * rated * max(1., config["wpp_efficiency"])))
    pv = pvp_cost(**costs(("solar_PV_cost", "solar_hardware_installation_cost",
                          "solar_inverter_cost", "solar_fixed_onm_cost")))
    cs, osolar = pv.compute(maximum("solar_MW [MW AC]"), maximum("DC_AC_ratio"))
    battery = battery_cost(**costs(("battery_energy_cost", "battery_power_cost",
                                   "battery_BOP_installation_commissioning_cost",
                                   "battery_control_system_cost", "battery_energy_onm_cost")),
                           life_y=1, battery_price_reduction_per_year=0)
    cb, ob = battery.compute(maximum("b_P [MW]") * maximum("b_E_h [h]"),
                             maximum("b_P [MW]"), np.ones(8760))
    annual = hpp_cost_annualized(config["wind_WACC"], config["solar_WACC"], config["battery_WACC"],
                                 config["hpp_economic_lifetime_years"],
                                 battery_replacement_year=config["battery_replacement_year"],
                                 battery_replacement_cost_fraction=config["battery_replacement_cost_fraction"])
    sh = shared_cost(**costs(("hpp_BOS_soft_cost", "hpp_grid_connection_cost", "land_cost")))
    csh, osh, *_ = sh.compute(
        config["facility_load_mw"] + maximum("b_P [MW]"),
        count * rated / maximum("wind_MW_per_km2 [MW/km2]"),
        maximum("solar_MW [MW AC]") * config["land_use_per_solar_MW"],
    )
    return float(annual.compute(cw, ow, cs, osolar, cb, ob, csh, osh)[-1])


def evaluate_candidate(task):
    """Importable process worker. No worker writes to shared output files."""
    design, kwargs = task
    model = None
    result = {"design": list(design), "status": "CONFIGURATION_OR_PROGRAMMING_ERROR",
              "accepted": False, "score": kwargs["hpp_rejection_score"], "error": None}
    try:
        if time.monotonic() >= kwargs["hpp_deadline"]:
            result.update(status="TIME_LIMIT_UNRESOLVED", error="Deadline before model construction")
            return result
        model = hpp_model_simple_datacenter_with_hpp_sizing(**kwargs)
        outputs = model.evaluate(*design)
        result.update(status=model.candidate_status, error=model.candidate_error,
                      score=float(outputs[-1]), solver_observations=model.solve_observations)
        if model.candidate_status == "FEASIBLE":
            result.update(accepted=True, economics=model.economics, metrics=model.metrics,
                          hourly=model.hourly, soc_mwh=model.soc_mwh,
                          solver_status=model.solver_status, validation=model.validation)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            model.prob.cleanup()
    return result


def validate_selected(result, config):
    """Independent validation of saved dispatch; no extra optimization after deadline."""
    b_P, hours = result["design"][-2:]
    residuals = validate_dispatch(result["hourly"], result["soc_mwh"], result["solver_status"],
                                  b_P, b_P * hours, config["battery_depth_of_discharge"],
                                  config["battery_charge_efficiency"], config["battery_initial_soc"], config["pue"])
    metrics = firm_load_metrics(result["hourly"]["unserved_facility_mw"], epsilon_mw=config["epsilon_mw"])
    if len(result["hourly"]["wind_mw"]) != 8760:
        raise ValueError("Selected dispatch is not a full 8760-hour year")
    allowed = int(np.floor((1 - config["reliability_target"]) * 8760 + 1e-9))
    if metrics["LOLH"] > allowed or metrics["firm_reliability"] < config["reliability_target"] - 1e-12:
        raise ValueError("Selected design violates hourly reliability")
    np.testing.assert_allclose(result["hourly"]["facility_load_mw"], config["facility_load_mw"], rtol=0, atol=1e-8)
    return {"physical_residuals": residuals, "metrics": metrics, "max_failed_hours": allowed}


def json_write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + "\n", encoding="utf8")
    temporary.replace(path)


def public_result(result):
    return {k: v for k, v in result.items() if k not in ("hourly", "soc_mwh")}


class RunController:
    def __init__(self, config, variables, kwargs, directory, start, executor):
        self.config, self.variables, self.kwargs = config, variables, kwargs
        self.directory, self.start, self.executor = directory, start, executor
        self.cache, self.history, self.best = {}, [], None
        self.attempts, self.search_summary = {}, {}
        self.seed_designs = []
        self.max_unique_evaluations = unique_evaluation_limit(config)

    def check_time(self):
        if time.monotonic() >= self.kwargs["hpp_deadline"]:
            raise BudgetExpired("Outer time budget reached")

    def check_evaluation_limit(self, reserved=0):
        if (self.max_unique_evaluations is not None
                and len(self.cache) + reserved >= self.max_unique_evaluations):
            raise EvaluationLimitReached("Unique physical evaluation limit reached")

    def record(self, result):
        result["evaluation_index"] = len(self.history) + 1
        result["elapsed_seconds"] = time.monotonic() - self.start
        is_new = tuple(result["design"]) not in self.cache
        improved = False
        if result["accepted"]:
            validate_selected(result, self.config)
            if self.best is None or result["score"] > self.best["score"]:
                improved = True
                self.best = result
                json_write(self.directory / "best_with_hpp_sizing.json", public_result(result))
                np.savez_compressed(self.directory / "dispatch_with_hpp_sizing.npz",
                                    soc_mwh=result["soc_mwh"], **result["hourly"])
        self.cache[tuple(result["design"])] = public_result(result)
        key = tuple(result["design"])
        self.attempts[key] = self.attempts.get(key, 0) + 1
        row = {"evaluation_index": result["evaluation_index"], "elapsed_seconds": result["elapsed_seconds"],
               "status": result["status"], "accepted": result["accepted"], "Nwt": result["design"][3],
               "wind_MW": result["design"][3] * result["design"][2], "solar_MW": result["design"][5],
               "b_P": result["design"][-2], "b_E_h": result["design"][-1],
               "b_E": result["design"][-2] * result["design"][-1], "score": result["score"],
               "annualized_cost_hpp": result.get("economics", {}).get("annualized_cost_hpp"),
               "EENS": result.get("metrics", {}).get("EENS"),
               "reliability": result.get("metrics", {}).get("firm_reliability"),
               "failed_hours": result.get("metrics", {}).get("LOLH"),
               "Revenue_dc": result.get("economics", {}).get("Revenue_dc"),
               "annualized_cost_dc": result.get("economics", {}).get("annualized_cost_dc"),
               "net_annual_value_system": result.get("economics", {}).get("net_annual_value_system"),
               "net_annual_value_dc_only": result.get("economics", {}).get("net_annual_value_dc_only"),
               "best_feasible_cost": None if self.best is None else self.best["economics"]["annualized_cost_hpp"],
               "best_feasible_design": None if self.best is None else json.dumps(self.best["design"]),
               "error": result.get("error"), "solver_observations": json.dumps(result.get("solver_observations", [])),
               "improved_incumbent": improved}
        proposal = result.get("proposal", {})
        for name in ("optimization_method", "proposal_id", "proposed_elapsed_seconds", "attempt",
                     "retry", "selection_reason", "search_iteration", "predicted_annualized_cost_hpp",
                     "predicted_cost_std_eur", "predicted_feasibility_probability",
                     "expected_improvement_eur", "acquisition_score", "refinement_event",
                     "refinement_center_cost_eur", "refinement_center_evaluation_index",
                     "refinement_center_changed", "refinement_coordinate", "refinement_step"):
            row[name] = proposal.get(name)
        row["refinement_center_design"] = json.dumps(proposal.get("refinement_center_design"))
        submitted = proposal.get("proposed_elapsed_seconds")
        row["evaluation_elapsed_seconds"] = None if submitted is None else result["elapsed_seconds"] - submitted
        row["surrogate_diagnostics"] = json.dumps(proposal.get("surrogate_diagnostics", {}))
        path = self.directory / "history_with_hpp_sizing.csv"
        with path.open("a", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not self.history:
                writer.writeheader()
            writer.writerow(row)
        self.history.append(row)
        if proposal.get("selection_reason", "").startswith("native_refinement_"):
            stats = self.search_summary.get("native_refinement")
            if stats is not None:
                stats["unique_evaluations"] += int(is_new)
                stats["incumbent_improvements"] += int(improved)
                if result["accepted"]:
                    stats["feasible_evaluations"] += 1
                elif result["status"] == "PROVEN_INFEASIBLE":
                    stats["proven_infeasible_evaluations"] += 1
                elif result["status"] in UNKNOWN_STATUSES:
                    stats["unknown_evaluations"] += 1
                else:
                    stats["error_evaluations"] += 1

    def evaluate(self, designs, proposals=None):
        if proposals is not None and len(proposals) != len(designs):
            raise ValueError("One proposal description is required for each design")
        answers = [None] * len(designs)
        pending = []
        queued, duplicates = {}, {}
        for i, design in enumerate(designs):
            design = list(map(float, design))
            design[3] = normalize_turbines(design[3])
            key = tuple(design)
            proposal = dict(proposals[i]) if proposals is not None else {}
            retry = bool(proposal.get("retry", False))
            if retry and (key not in self.cache or self.cache[key]["status"] not in UNKNOWN_STATUSES
                          or self.attempts.get(key, 0) > self.config.get("fa_max_retries", 1)):
                raise ValueError("Retry is only allowed for an unknown result within the retry limit")
            if key in self.cache and not retry:
                answers[i] = self.cache[key]["score"]
            elif proposals is not None and key in queued:
                duplicates[i] = queued[key]
            else:
                queued[key] = i
                pending.append((i, design, proposal))
        # Submit only a worker-sized batch: no long queue beyond the deadline.
        for offset in range(0, len(pending), self.config["n_procs"]):
            self.check_time()
            self.check_evaluation_limit()
            batch = pending[offset:offset + self.config["n_procs"]]
            futures = {}
            reserved = 0
            for i, design, proposal in batch:
                if futures and time.monotonic() >= self.kwargs["hpp_deadline"]:
                    break  # Record already-started workers before propagating the deadline.
                self.check_time()
                if (self.max_unique_evaluations is not None
                        and len(self.cache) + reserved >= self.max_unique_evaluations):
                    break  # Complete this bounded batch without exceeding the count cap.
                proposal.update(proposal_id=uuid.uuid4().hex, design=design,
                                optimization_method=self.config.get("optimization_method", "native_ego"),
                                proposed_elapsed_seconds=time.monotonic() - self.start,
                                attempt=self.attempts.get(tuple(design), 0) + 1)
                proposal.setdefault("selection_reason", "native_ego")
                proposal.setdefault("retry", False)
                # Persist predictions before starting the worker, including any
                # proposal whose evaluation is interrupted or exceeds the budget.
                with (self.directory / "proposals_with_hpp_sizing.jsonl").open("a", encoding="utf8") as stream:
                    stream.write(json.dumps(proposal) + "\n")
                futures[self.executor.submit(evaluate_candidate, (design, self.kwargs))] = (i, proposal)
                reserved += int(tuple(design) not in self.cache)
            fatal = None
            recorded = set()
            try:
                for future in as_completed(futures):
                    result = future.result()
                    result["proposal"] = futures[future][1]
                    self.record(result)
                    recorded.add(future)
                    answers[futures[future][0]] = result["score"]
                    if result["status"] == "CONFIGURATION_OR_PROGRAMMING_ERROR":
                        fatal = result["error"]
            except KeyboardInterrupt:
                # Stop submission, then drain the bounded in-flight batch so a
                # completed feasible incumbent is not lost. This may overrun.
                for future in futures:
                    if future not in recorded and not future.cancel():
                        result = future.result()
                        result["proposal"] = futures[future][1]
                        self.record(result)
                raise
            if fatal:
                raise RuntimeError(fatal)
            self.check_time()
            self.check_evaluation_limit()
        for i, original in duplicates.items():
            answers[i] = answers[original]
        return np.array(answers, dtype=float).reshape(-1, 1)


class BudgetedEvaluator(native_ego.ParallelEvaluator):
    def __init__(self, controller, n_procs):
        super().__init__(n_procs)
        self.controller, self.first_doe = controller, True

    def run_ydoe(self, fun, x, **kwargs):
        self.controller.check_time()
        if self.first_doe:
            # Mutate the actual native DOE in place, so xdoe/ydoe stay aligned.
            design_names = kwargs["design_vars"]
            for i, seed in enumerate(self.controller.seed_designs[:len(x)]):
                selected = [seed[VARIABLE_NAMES.index(name)] for name in design_names]
                x[i] = kwargs["scaler"].transform([selected])[0]
            self.first_doe = False
        physical = kwargs["scaler"].inverse_transform(x)
        designs = native_ego.expand_x_for_model_eval(physical, kwargs)
        return kwargs["opt_sign"] * self.controller.evaluate(designs)

    def run_both(self, fun, i, **kwargs):
        self.controller.check_time()
        result = super().run_both(fun, i, **kwargs)
        self.controller.check_time()
        return result


def run_hpp_sizing(config_path, overrides=None):
    """Return the best validated feasible candidate, never a global-optimum claim."""
    if not RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("Run only one HPP sizing search per Python process")
    try:
        return _run_hpp_sizing(config_path, overrides)
    finally:
        RUN_LOCK.release()


def _run_hpp_sizing(config_path, overrides):
    start = time.monotonic()
    config = read_configuration(config_path)
    config.update(overrides or {})
    method = config.setdefault("optimization_method", "native_ego")
    if method not in ("native_ego", "feasibility_aware"):
        raise ValueError("optimization_method must be native_ego or feasibility_aware")
    if method == "feasibility_aware":
        config.update(search_settings(config))
    unique_evaluation_limit(config)
    variables = prepare_variables(config)
    for key in ("hpp_sizing_time_limit_seconds", "ems_time_limit_seconds"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 <= config["reliability_target"] <= 1 or config["facility_load_mw"] <= 0:
        raise ValueError("Invalid reliability target or fixed facility load")
    for key in ("n_procs", "n_doe", "n_clusters", "max_iter", "npred", "min_conv_iter"):
        if int(config[key]) != config[key] or config[key] < (0 if key == "max_iter" else 1):
            raise ValueError(f"Invalid native EGO setting: {key}")
    dimension = sum(v["var_type"] == "design" for v in variables.values())
    if method == "native_ego" and dimension and config["n_doe"] < max(3, dimension + 1):
        raise ValueError("n_doe must be at least max(3, number of design variables + 1)")
    if method == "native_ego" and config["npred"] * config["n_procs"] * .01 < config["n_clusters"] + 1:
        raise ValueError("npred is too small for native EGO's top-1% clustering")
    os.environ["OPENMDAO_REPORTS"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    deadline = start + config["hpp_sizing_time_limit_seconds"]
    directory = Path(config["hpp_sizing_output_dir"]).expanduser().resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8] + "_with_hpp_sizing")
    directory.mkdir(parents=True, exist_ok=False)
    json_write(directory / "configuration_with_hpp_sizing.json", config)
    sites = pd.read_csv(Path(examples_filepath) / "examples_sites.csv", sep=";")
    site = sites.loc[sites.name == config["hpp_sizing_site"]].iloc[0]
    bound = native_cost_bound(config, variables)
    rejection = -(bound + max(1., .01 * bound)) / 1e6
    kwargs = dict(config)
    kwargs.update(sim_pars_fn=str(Path(config_path).resolve()), latitude=float(site.latitude),
                  longitude=float(site.longitude), altitude=float(site.altitude),
                  input_ts_fn=str(Path(examples_filepath) / site.input_ts_fn), verbose=False,
                  hpp_deadline=deadline, hpp_cost_upper_bound=bound, hpp_rejection_score=rejection)
    reason, error = "completed", None
    with ProcessPoolExecutor(max_workers=config["n_procs"], mp_context=multiprocessing.get_context("spawn")) as executor:
        control = RunController(config, variables, kwargs, directory, start, executor)
        try:
            reference = [variables[name]["value"] for name in VARIABLE_NAMES]
            active = [name for name in VARIABLE_NAMES if variables[name]["var_type"] == "design"]
            upper = [variables[name]["limits"][1] if name in active else variables[name]["value"] for name in VARIABLE_NAMES]
            # Feasibility scouts are actual constrained evaluations, never assumed feasible.
            scouts = [reference]
            if active:
                scouts += [upper]
            control.evaluate(scouts, proposals=[{"selection_reason": reason, "search_iteration": -1}
                             for reason in ("reference_scout", "upper_bound_scout")[:len(scouts)]]
                             if method == "feasibility_aware" else None)
            if method == "native_ego" and active and control.best is None:
                # Small deterministic exploration before attempting a surrogate with flat scores.
                mix = native_ego.get_mixint_context(variables, seed=config["n_seed"])
                sample = native_ego.get_sampling(mix, seed=config["n_seed"])(config["n_doe"])
                physical = np.asarray(mix.design_space.decode_values(sample), dtype=float)
                control.evaluate(native_ego.expand_x_for_model_eval(physical, {
                    "variables": variables, "list_vars": VARIABLE_NAMES, "design_vars": active,
                    "fixed_vars": [n for n in VARIABLE_NAMES if n not in active]}))
            if method == "feasibility_aware":
                reason = run_feasibility_aware(control, config, variables, VARIABLE_NAMES)
            elif not active:
                reason = "all_variables_fixed"
            elif control.best is None:
                reason = "no_feasible_design_found_in_initial_exploration"
            else:
                control.seed_designs = [control.best["design"]]
                # Include a different completed scout so initial objectives need not be flat.
                control.seed_designs += [list(d) for d in control.cache if list(d) != control.best["design"]][:1]
                inputs = dict(kwargs)
                inputs.pop("verbose", None)  # Native model_evaluation also supplies this keyword.
                inputs.update(name=config["hpp_sizing_site"], hpp_model=hpp_model_simple_datacenter_with_hpp_sizing,
                              variables=variables, opt_var=SCORE_NAME, num_batteries=1, work_dir=str(directory) + os.sep,
                              final_design_fn=str(directory / "native_final_with_hpp_sizing.csv"))
                # Only evaluation scheduling is replaced; the native EGO run/search is called verbatim.
                with patch.object(native_ego, "ParallelEvaluator", lambda n_procs: BudgetedEvaluator(control, n_procs)):
                    driver = native_ego.EfficientGlobalOptimizationDriver(**inputs)
                    driver.run()
                reason = "native_ego_stopped_no_global_optimality_certificate"
        except EvaluationLimitReached:
            reason = "unique_evaluation_limit"
        except BudgetExpired:
            reason = "outer_time_limit"
        except KeyboardInterrupt:
            reason = "user_interrupt"
        except Exception as exc:
            reason, error = "error", f"{type(exc).__name__}: {exc}"
        # Executor waits for active workers; solve budgets were capped before each solve.
    if time.monotonic() >= deadline and reason not in ("error", "user_interrupt"):
        reason = "outer_time_limit"
    best = control.best
    final_validation = validate_selected(best, config) if best is not None else None
    result = {"best": None if best is None else public_result(best), "termination_reason": reason,
              "optimization_method": method, "search_summary": control.search_summary,
              "error": error, "globally_optimal": False, "elapsed_seconds": time.monotonic() - start,
              "deadline_semantics": "best_effort; active solver/construction/surrogate/cleanup may overrun",
              "time_limit_seconds": config["hpp_sizing_time_limit_seconds"],
              "max_unique_evaluations": control.max_unique_evaluations,
              "native_refinement_enabled": method == "feasibility_aware" and config["fa_native_refinement_enabled"],
              "evaluated_designs": len(control.history), "feasible_designs": sum(r["accepted"] for r in control.history),
              "unique_designs": len(control.cache),
              "proposals_path": str(directory / "proposals_with_hpp_sizing.jsonl"),
              "history_path": str(directory / "history_with_hpp_sizing.csv"), "run_directory": str(directory),
              "cost_upper_bound_eur_year": bound, "rejection_score": rejection,
              "final_validation": final_validation, "configuration": config,
              "bound_hits": {} if best is None else {n: {
                  "lower": bool(np.isclose(best["design"][VARIABLE_NAMES.index(n)], variables[n]["limits"][0], rtol=0, atol=1e-8)),
                  "upper": bool(np.isclose(best["design"][VARIABLE_NAMES.index(n)], variables[n]["limits"][1], rtol=0, atol=1e-8))}
                  for n in SIZABLE if variables[n]["var_type"] == "design"}}
    json_write(directory / "result_with_hpp_sizing.json", result)
    return result


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    summary = run_hpp_sizing(args.config)
    print(json.dumps(summary, indent=2))
