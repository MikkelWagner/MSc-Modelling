"""Separate genuine cost regression from dispatch feasibility classification.

Global proposals use native mixed-integer LHS samples. Optional refinement calls
the native coordinate helpers around the current verified feasible incumbent.
Predictions only select global evaluations; the unchanged EMS and physical
validation decide acceptance. This module never reads previous run files.
"""

import time
import warnings

import numpy as np
from scipy.special import ndtr
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from hydesign import Parallel_EGO as native_ego


UNKNOWN_STATUSES = frozenset((
    "TIME_LIMIT_WITH_FEASIBLE_DISPATCH", "TIME_LIMIT_UNRESOLVED", "NUMERICAL_ERROR",
))
DEFAULTS = {
    "fa_batch_size": 2,
    "fa_exploration_interval": 5,
    "fa_max_retries": 1,
    "fa_retry_interval": 5,
    "fa_retry_cooldown_iterations": 3,
    "fa_cost_theta_bounds": [1e-6, 20.0],
    "fa_classifier_length_scale": 0.2,
    "fa_classifier_optimize": True,
    "fa_classifier_max_iter": 100,
    "fa_native_refinement_enabled": False,
    "fa_native_refinement_interval": 5,
}


def search_settings(config):
    settings = {name: config.get(name, default) for name, default in DEFAULTS.items()}
    for name in ("fa_batch_size", "fa_exploration_interval", "fa_max_retries",
                 "fa_retry_interval", "fa_retry_cooldown_iterations", "fa_classifier_max_iter"):
        value = settings[name]
        if not np.isfinite(value) or int(value) != value or value < (0 if name == "fa_max_retries" else 1):
            raise ValueError(f"Invalid search setting: {name}")
    bounds = settings["fa_cost_theta_bounds"]
    if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
        raise ValueError("fa_cost_theta_bounds must be finite positive increasing bounds")
    length = settings["fa_classifier_length_scale"]
    if not np.isfinite(length) or length <= 0:
        raise ValueError("fa_classifier_length_scale must be positive and finite")
    if not isinstance(settings["fa_classifier_optimize"], bool):
        raise ValueError("fa_classifier_optimize must be a boolean")
    if not isinstance(settings["fa_native_refinement_enabled"], bool):
        raise ValueError("fa_native_refinement_enabled must be a boolean")
    interval = settings["fa_native_refinement_interval"]
    if (isinstance(interval, bool) or not isinstance(interval, (int, float))
            or not np.isfinite(interval) or int(interval) != interval or interval < 1):
        raise ValueError("fa_native_refinement_interval must be a positive integer")
    if settings["fa_native_refinement_enabled"]:
        tol = config.get("tol", 1e-6)
        if (isinstance(tol, bool) or not isinstance(tol, (int, float))
                or not np.isfinite(tol) or tol < 0):
            raise ValueError("tol must be finite and nonnegative for native refinement")
    if int(config["n_seed"]) != config["n_seed"] or config["n_seed"] < 0:
        raise ValueError("n_seed must be a nonnegative integer")
    return settings


class SearchSpace:
    """Encode only active variables; retain the original full evaluation vector."""

    def __init__(self, variables, names):
        self.variables, self.names = variables, tuple(names)
        self.active = [name for name in names if variables[name]["var_type"] == "design"]
        self.indices = [self.names.index(name) for name in self.active]
        self.lower = np.array([variables[name]["limits"][0] for name in self.active], dtype=float)
        self.upper = np.array([variables[name]["limits"][1] for name in self.active], dtype=float)
        self.dimension = len(self.active)
        self.fixed = [name for name in names if name not in self.active]

    def encode(self, designs):
        designs = np.asarray(designs, dtype=float).reshape(-1, len(self.names))
        return (designs[:, self.indices] - self.lower) / (self.upper - self.lower)

    def sample(self, count, seed):
        if not self.active:
            return np.array([[self.variables[name]["value"] for name in self.names]], dtype=float)
        context = native_ego.get_mixint_context(self.variables, seed=int(seed))
        sample = native_ego.get_sampling(context, seed=int(seed))(int(count))
        physical = np.asarray(context.design_space.decode_values(sample), dtype=float)
        # Use the native casting/expansion, not a second HPP design convention.
        physical = native_ego.cast_to_mixint(physical, self.variables)
        return native_ego.expand_x_for_model_eval(physical, dict(
            variables=self.variables, list_vars=self.names,
            design_vars=self.active, fixed_vars=self.fixed))


def training_data(records, space):
    """Return costs in EUR/year and labels, never an optimizer rejection score."""
    cost_designs, costs, labelled_designs, labels = [], [], [], []
    for record in records:
        status = record["status"]
        if status == "FEASIBLE":
            if not record["accepted"]:
                raise ValueError("FEASIBLE observation has not passed acceptance validation")
            cost = float(record["economics"]["annualized_cost_hpp"])
            if not np.isfinite(cost) or cost < 0:
                raise ValueError("Genuine annualized HPP cost must be finite and nonnegative")
            cost_designs.append(record["design"])
            costs.append(cost)
            labelled_designs.append(record["design"])
            labels.append(1)
        elif status == "PROVEN_INFEASIBLE":
            if record["accepted"]:
                raise ValueError("An infeasible observation cannot be accepted")
            labelled_designs.append(record["design"])
            labels.append(0)
        elif status in UNKNOWN_STATUSES:
            if record["accepted"]:
                raise ValueError("An unresolved observation cannot be accepted")
        else:
            raise ValueError(f"Cannot train on evaluation error: {status}")
    return (space.encode(cost_designs), np.asarray(costs),
            space.encode(labelled_designs), np.asarray(labels, dtype=int))


def expected_improvement(best, mean, std):
    """Nonnegative EI for minimization, including deterministic predictions."""
    mean, std = np.broadcast_arrays(np.asarray(mean, dtype=float), np.asarray(std, dtype=float))
    result = np.zeros_like(mean)
    valid = np.isfinite(mean) & np.isfinite(std) & (std >= 0)
    if not np.isfinite(best):
        return result
    deterministic = valid & (std <= 1e-12)
    result[deterministic] = np.maximum(best - mean[deterministic], 0.0)
    uncertain = valid & ~deterministic
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        improvement = best - mean[uncertain]
        z = improvement / std[uncertain]
        result[uncertain] = improvement * ndtr(z) + std[uncertain] * np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)
    return np.maximum(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


class FeasibilitySurrogate:
    def __init__(self, space, config):
        self.space, self.config = space, config
        self.settings = search_settings(config)
        self.cost_model = self.classifier = None
        self.fingerprint = None
        self.diagnostics = {}
        self.fit_count = 0

    def fit(self, records):
        xc, yc, xf, labels = training_data(records, self.space)
        fingerprint = (xc.tobytes(), yc.tobytes(), xf.tobytes(), labels.tobytes())
        if fingerprint == self.fingerprint:
            return  # An unknown result does not change either training dataset.
        self.fingerprint = fingerprint
        self.fit_count += 1
        self.cost_model = self.classifier = None
        self.costs = yc.copy()
        self.labels = labels.copy()
        self.prior = float((labels.sum() + 1) / (len(labels) + 2))
        self.diagnostics = {"cost_training_count": len(yc), "feasibility_training_count": len(labels),
                            "feasible_labels": int(labels.sum()), "infeasible_labels": int((labels == 0).sum()),
                            "fit_count": self.fit_count, "warnings": []}
        # Native KPLSK needs enough rows for its active dimensions. A constant
        # target is handled directly, without an ill-conditioned regression.
        self.constant_cost = None
        if len(yc) >= max(3, self.space.dimension + 1):
            if np.ptp(yc) <= 1e-8:
                self.constant_cost = float(yc[0])
            else:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        np.random.seed(int(self.config["n_seed"]))
                        self.cost_model = native_ego.get_sm(
                            xc, yc[:, None] / 1e6, n_comp=min(self.space.dimension, 4),
                            theta_bounds=self.settings["fa_cost_theta_bounds"])
                    self.diagnostics["warnings"] += [str(w.message) for w in caught]
                except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    self.diagnostics["cost_model_error"] = f"{type(exc).__name__}: {exc}"
        if len(np.unique(labels)) == 2:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.classifier = GaussianProcessClassifier(
                        kernel=RBF(np.full(self.space.dimension, self.settings["fa_classifier_length_scale"])),
                        optimizer="fmin_l_bfgs_b" if self.settings["fa_classifier_optimize"] else None,
                        max_iter_predict=self.settings["fa_classifier_max_iter"],
                        random_state=int(self.config["n_seed"]), n_restarts_optimizer=0).fit(xf, labels)
                self.diagnostics["warnings"] += [str(w.message) for w in caught]
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                self.classifier = None
                self.diagnostics["classifier_error"] = f"{type(exc).__name__}: {exc}"

    def predict(self, designs):
        x = self.space.encode(designs)
        mean, std = np.full(len(x), np.nan), np.full(len(x), np.nan)
        probability = np.full(len(x), self.prior)
        try:
            if self.constant_cost is not None:
                mean[:], std[:] = self.constant_cost, 0.0
            elif self.cost_model is not None:
                mean = self.cost_model.predict_values(x).ravel() * 1e6
                variance = self.cost_model.predict_variances(x).ravel()
                std = np.sqrt(np.maximum(variance, 0.0)) * 1e6
                std[variance < -1e-8] = np.nan
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            self.diagnostics["cost_prediction_error"] = f"{type(exc).__name__}: {exc}"
        try:
            if self.classifier is not None:
                probability = self.classifier.predict_proba(x)[:, list(self.classifier.classes_).index(1)]
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            self.diagnostics["feasibility_prediction_error"] = f"{type(exc).__name__}: {exc}"
        probability = np.clip(np.nan_to_num(probability, nan=self.prior, posinf=self.prior, neginf=self.prior), 0., 1.)
        best = float(self.costs.min()) if len(self.costs) else np.inf
        ei = expected_improvement(best, mean, std)
        return mean, std, probability, ei, ei * probability


class NativeRefinement:
    """Schedule native coordinate candidates; never rank them by a surrogate.

    Every interval iterations, add at most fa_batch_size evaluations. Compare
    cost with the previous refinement event using native tol: stagnation selects
    native perturbations, otherwise native coordinate extremes. Rotate through
    opposing coordinate pairs so the cap does not always select the first axes.
    There is no pending candidate queue: every event uses the latest incumbent.
    """

    def __init__(self, space, config):
        self.space, self.config = space, config
        self.settings = search_settings(config)
        self.previous_cost = None
        self.previous_center = None
        self.cursor = 0
        self.diagnostics = dict(
            events=0, generated_candidates=0, proposed_candidates=0,
            skipped_cached_or_duplicate=0, center_updates=0,
            unique_evaluations=0, feasible_evaluations=0,
            proven_infeasible_evaluations=0, unknown_evaluations=0,
            error_evaluations=0, incumbent_improvements=0,
            generation_seconds=0.0, evaluation_wall_seconds=0.0,
        )

    def propose(self, best, records, iteration):
        if (not self.settings["fa_native_refinement_enabled"] or not self.space.active
                or (iteration + 1) % self.settings["fa_native_refinement_interval"]):
            return [], []
        # Only RunController.best has passed the shared acceptance validation.
        if best is None or best["status"] != "FEASIBLE" or not best["accepted"]:
            return [], []
        cost = float(best["economics"]["annualized_cost_hpp"])
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("Refinement requires a genuine finite nonnegative cost")
        started = time.perf_counter()
        center = tuple(best["design"])
        changed = self.previous_center is not None and center != self.previous_center
        self.diagnostics["center_updates"] += int(changed)
        self.diagnostics["events"] += 1
        x = self.space.encode([center])
        if not np.isfinite(x).all() or np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
            raise ValueError("Refinement incumbent is outside the configured bounds")
        x = np.clip(x, 0., 1.)
        if self.previous_cost is None:
            change = np.inf
        elif cost == 0:
            change = 0. if self.previous_cost == 0 else np.inf
        else:
            change = abs(1. - self.previous_cost / cost)
        step = None
        if change < self.config.get("tol", 1e-6):
            # Same native seed and 5%-25% step, with a private RNG so the
            # existing global proposal/random-number sequence is not changed.
            rng = np.random.RandomState(int(self.config["n_seed"]) * 100 + iteration)
            step = float(rng.uniform(.05, .25))
            points = native_ego.perturbe_around_point(x, step=step)
            reason = "native_refinement_perturbation"
        else:
            points = native_ego.extreme_around_point(x)
            reason = "native_refinement_extreme"
        self.previous_cost, self.previous_center = cost, center
        physical = self.space.lower + points * (self.space.upper - self.space.lower)
        physical = native_ego.cast_to_mixint(physical, self.space.variables)
        # Preserve untouched coordinates bit-for-bit: an encode/decode round trip
        # must not turn a cached design into a new configuration through roundoff.
        unchanged = points == x
        original = np.broadcast_to(np.asarray(center)[self.space.indices], physical.shape)
        physical[unchanged] = original[unchanged]
        designs = native_ego.expand_x_for_model_eval(physical, dict(
            variables=self.space.variables, list_vars=self.space.names,
            design_vars=self.space.active, fixed_vars=self.space.fixed))
        self.diagnostics["generated_candidates"] += len(designs)
        # The helpers return one direction for all axes, then the opposite
        # direction. Interleave pairs only for scheduling; coordinates are native.
        d = self.space.dimension
        order = [i for axis in range(d) for i in (axis, axis + d)]
        start_cursor = self.cursor
        seen = {center}
        selected, proposals = [], []
        for offset in range(len(order)):
            position = (start_cursor + offset) % len(order)
            index = order[position]
            self.cursor = (position + 1) % len(order)
            key = tuple(designs[index])
            if key in records or key in seen:
                self.diagnostics["skipped_cached_or_duplicate"] += 1
                continue
            seen.add(key)
            selected.append(designs[index].tolist())
            proposals.append(dict(
                selection_reason=reason, search_iteration=iteration, retry=False,
                refinement_event=self.diagnostics["events"],
                refinement_center_design=list(center),
                refinement_center_cost_eur=cost,
                refinement_center_evaluation_index=best.get("evaluation_index"),
                refinement_center_changed=changed,
                refinement_coordinate=self.space.active[index % d],
                refinement_step=step,
            ))
            if len(selected) >= self.settings["fa_batch_size"]:
                break
        self.diagnostics["proposed_candidates"] += len(selected)
        self.diagnostics["generation_seconds"] += time.perf_counter() - started
        return selected, proposals


class CandidatePolicy:
    def __init__(self, space, config):
        self.space, self.config = space, config
        self.settings = search_settings(config)
        self.surrogate = FeasibilitySurrogate(space, config)
        self.proposed_count = 0

    def propose(self, records, attempts, iteration):
        """Rank globally sampled unevaluated designs; bounded retries are explicit."""
        self.surrogate.fit(list(records.values()))
        pool = np.vstack([self.space.sample(self.config["npred"],
                          (worker + iteration * self.config["n_procs"]) * 100 + self.config["n_seed"])
                          for worker in range(self.config["n_procs"])])
        candidates = {tuple(row): row for row in pool if tuple(row) not in records}
        retry_keys = set()
        if (iteration + 1) % self.settings["fa_retry_interval"] == 0:
            for key, record in records.items():
                last_iteration = record.get("proposal", {}).get("search_iteration", -1)
                if (record["status"] in UNKNOWN_STATUSES
                        and attempts.get(key, 1) <= self.settings["fa_max_retries"]
                        and iteration - last_iteration >= self.settings["fa_retry_cooldown_iterations"]):
                    candidates[key] = np.asarray(key)
                    retry_keys.add(key)
        if not candidates:
            return [], []
        designs = np.asarray(list(candidates.values()))
        mean, std, probability, ei, acquisition = self.surrogate.predict(designs)
        x = self.space.encode(designs)
        seen = self.space.encode([r["design"] for r in records.values()])
        distance = cdist(x, seen).min(axis=1) if len(seen) else np.ones(len(x))
        retry = np.array([tuple(row) in retry_keys for row in designs])
        available = np.ones(len(designs), dtype=bool)
        selected, proposals = [], []
        for _ in range(min(self.settings["fa_batch_size"], len(designs))):
            if not available.any():
                break
            scheduled_exploration = (self.proposed_count + 1) % self.settings["fa_exploration_interval"] == 0
            useful_acquisition = available & (acquisition > 0)
            if not scheduled_exploration and useful_acquisition.any():
                index = int(np.argmax(np.where(available, acquisition, -np.inf)))
                reason = "constrained_expected_improvement"
            else:
                # Coverage never uses a feasibility probability cut-off. Do not
                # spend an exploration slot on a previously attempted design.
                choices = available & ~retry
                if not choices.any():
                    break
                index = int(np.argmax(np.where(choices, distance, -np.inf)))
                reason = "global_exploration" if scheduled_exploration else "sparse_or_flat_model_exploration"
            key = tuple(designs[index])
            is_retry = key in retry_keys
            if is_retry:
                reason = "retry_" + reason
                available[retry] = False  # At most one retry in an eligible batch.
            selected.append(designs[index].tolist())
            proposals.append({
                "selection_reason": reason, "search_iteration": iteration, "retry": is_retry,
                "predicted_annualized_cost_hpp": float(mean[index]) if np.isfinite(mean[index]) else None,
                "predicted_cost_std_eur": float(std[index]) if np.isfinite(std[index]) else None,
                "predicted_feasibility_probability": float(probability[index]),
                "expected_improvement_eur": float(ei[index]), "acquisition_score": float(acquisition[index]),
                "surrogate_diagnostics": dict(self.surrogate.diagnostics),
            })
            available[index] = False
            distance = np.minimum(distance, cdist(x, x[[index]]).ravel())
            self.proposed_count += 1
        return selected, proposals


def run_feasibility_aware(control, config, variables, names):
    """Use the shared controller; the cost predictions never decide acceptance."""
    space = SearchSpace(variables, names)
    refinement = NativeRefinement(space, config)
    enabled = refinement.settings["fa_native_refinement_enabled"]
    control.search_summary = {"iterations": 0, "surrogate_fit_count": 0,
                              "local_search": enabled, "native_refinement_enabled": enabled,
                              "native_refinement": refinement.diagnostics}
    if not space.active:
        return "all_variables_fixed"
    control.check_time()
    initial = space.sample(config["n_doe"], config["n_seed"])
    # Match native EGO's insertion of completed scouts in the initial DOE.
    seeds = [control.best["design"]] if control.best is not None else []
    seeds += [list(key) for key in control.cache if list(key) not in seeds]
    for i, design in enumerate(seeds[:min(2, len(initial))]):
        initial[i] = design
    initial = list({tuple(row): row.tolist() for row in initial}.values())
    control.evaluate(initial, proposals=[{"selection_reason": "initial_doe", "search_iteration": -1} for _ in initial])
    policy = CandidatePolicy(space, config)
    empty_pools = 0
    for iteration in range(config["max_iter"]):
        control.check_time()
        designs, proposals = policy.propose(control.cache, control.attempts, iteration)
        control.search_summary.update(iterations=iteration + 1, surrogate_fit_count=policy.surrogate.fit_count,
                                      training=policy.surrogate.diagnostics)
        control.check_time()  # Training/sampling may have consumed the remaining budget.
        if designs:
            control.evaluate(designs, proposals=proposals)
        refined = []
        if enabled:
            control.check_time()
            refined, metadata = refinement.propose(control.best, control.cache, iteration)
            control.check_time()
            if refined:
                started = time.perf_counter()
                try:
                    control.evaluate(refined, proposals=metadata)
                finally:
                    refinement.diagnostics["evaluation_wall_seconds"] += time.perf_counter() - started
        if not designs and not refined:
            empty_pools += 1
            if empty_pools >= max(3, policy.settings["fa_retry_interval"]):
                return "no_new_candidates_sampled"  # Not a claim of global infeasibility.
            continue
        empty_pools = 0
    return "feasibility_aware_iteration_limit"
