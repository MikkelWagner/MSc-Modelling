# Feasibility-aware HPP sizing implementation

Implemented on 2026-10-26. The methodological change is separate cost regression
and feasibility classification. The physical evaluator and annualized economics
are shared with the existing native EGO route. No 600-second benchmark was run.

## A. Files

Modified only these three existing thesis files:

- `hydesign/sizing_simple_datacenter_with_hpp_sizing.py`: method selection, shared
  scheduling, bounded retry accounting, proposal diagnostics and result metadata.
- `hydesign/examples/thesis_simple_datacenter/simple_datacenter_with_hpp_sizing.yml`:
  method selector and configurable surrogate/search settings.
- `docs/notebooks/simple_datacenter_with_hpp_sizing.ipynb`: selector, diagnostic
  display and restored YAML defaults. Removed the active fixed-19-turbine override
  and cleared old cell outputs.

New files:

- `hydesign/search_simple_datacenter_with_hpp_sizing.py`: search space, training
  datasets, two surrogates, acquisition, global candidate sampling and search loop.
- `hydesign/tests/test_search_simple_datacenter_with_hpp_sizing.py`: focused tests.
- `scripts/validate_feasibility_with_hpp_sizing.py`: explicitly short integration checks.
- `docs/benchmarks/feasibility_validation_with_hpp_sizing.json`: measured results.
- This report, `docs/benchmarks/feasibility_implementation_with_hpp_sizing.md`.

New run artifacts use the existing unique directories under `hpp_sizing_runs`.
Existing results were not overwritten. Proposal records are additionally saved as
`proposals_with_hpp_sizing.jsonl`, before launching their evaluations.

## B. Original files remain unchanged

A SHA-256 comparison against the starting snapshot checked 536 existing repository
files: 533 were unchanged; the only three changes are listed above. In particular,
`hydesign/Parallel_EGO.py`, both existing HPP cost/sizing assemblies, the EMS,
metrics, wind/PV generation, native cost classes, HPP annualization and DC cost
files are unchanged. No dependency was installed or upgraded.

## C. Cost surrogate

`training_data()` reads `economics['annualized_cost_hpp']` only from accepted,
validated `FEASIBLE` results. It never reads the adapter's rejection score. Neither
infeasible nor unresolved evaluations enter cost training. Real costs are retained
in EUR/year; regression uses MEUR/year for conditioning and predictions are
converted back to EUR/year.

The model reuses native `Parallel_EGO.get_sm()` / SMT KPLSK, including its linear
trend and squared-exponential correlation. Active design inputs are normalized
to their configured bounds. At least `max(3, dimension + 1)` feasible observations
are needed. Sparse or failed fits trigger global coverage sampling; a constant
cost dataset is handled without fitting a singular regression. Both models are
retrained only when their training data change. Unknown outcomes alone do not
trigger retraining.

Obtaining costs for infeasible designs is deferred: the existing assembly runs
the EMS before costs, and native wind OPEX also requires the generation profile.
There is no duplicated cost equation or speculative fabricated cost observation.

## D. Feasibility model

A scikit-learn GaussianProcessClassifier with an RBF kernel estimates the chance
of dispatch feasibility. Positive labels require accepted, physically validated
feasible dispatch; negative labels require proven infeasibility of the primary
dispatch problem. One-class/empty data use the Laplace-smoothed prior
`(positive_count + 1) / (label_count + 2)`. With both classes available, the RBF
length scales are fitted by default. Their initial value, fitting toggle and
iteration limit are configurable. Numerical fit failures fall back to the prior
and are recorded in surrogate diagnostics.

This prediction is not the plant's physical reliability. Only the unchanged CPLEX
constraints and independent physical validation determine acceptance.

## E. Candidate selection

Each iteration draws global native mixed-integer Latin-hypercube pools over the
full configured bounds, using the configured seed. It removes previously
evaluated designs and duplicate samples, apart from explicit bounded retries.
Candidates are ranked by `EI_cost * P(feasible)`, where EI uses the current best
genuine feasible cost. Zero variance, nonfinite predictions and sparse models
are handled explicitly.

Every fifth adaptive proposal, by default, maximizes normalized distance from
evaluated designs and the already selected batch points. This global coverage
step has no feasibility-probability cutoff. It also provides the fallback when
there is no useful acquisition or not enough feasible cost data. Batch size is
configurable. EI-ranked members share the model fitted at the start of that batch.

Initial evaluations retain the existing reference and upper-bound scouts, both
requiring real constrained evaluation. The completed scouts occupy the first
initial DOE positions as in the native route. No historical result is loaded.
Unlike the native wrapper, the new loop can continue exploration when the initial
DOE contains no feasible design. Failure to find one does not prove the domain
infeasible.

## F. Reused libraries

- HyDesign: native mixed-integer design space, LHS, casting, expansion, KPLSK helper,
  generation, assemblies, CPLEX dispatch, validation and all economic calculations.
- SMT 2.9.2: native LHS and KPLSK regression.
- scikit-learn 1.9.1: Gaussian-process classification.
- SciPy 1.17.1 / NumPy: normal CDF for EI, distances and array operations.
- Existing standard-library Windows spawn/process pool, YAML and pandas support.

## G. Evaluation statuses and retries

| Status | Training treatment | Acceptance |
|---|---|---|
| `FEASIBLE` with successful validation | Genuine cost and positive label | Yes |
| `PROVEN_INFEASIBLE` | Negative label only | No |
| `TIME_LIMIT_WITH_FEASIBLE_DISPATCH` | Unknown; excluded from both models | No |
| `TIME_LIMIT_UNRESOLVED` | Unknown; excluded from both models | No |
| `NUMERICAL_ERROR` | Unknown; excluded from both models | No |
| `CONFIGURATION_OR_PROGRAMMING_ERROR` | Abort and report the error | No |

The existing conservative requirement remains: both lexicographic EMS stages
and physical validation must finish before accepting a candidate. Cleanup failure
after a feasible primary solve does not prove physical infeasibility.

Every fifth batch may consider an unknown candidate again, after a three-iteration
cooldown. It must win on constrained acquisition. At most one retry is selected
in that batch, and at most one additional attempt per design is allowed by default.
Retries use the same solver settings and remaining outer budget. Proven infeasible
and accepted designs are cached; they are not retried. Cached and within-batch
duplicates do not start another new-method evaluation.

## H. Design variables

The existing configuration remains authoritative: `Nwt` is integer, while
`solar_MW [MW AC]`, `b_P [MW]` and `b_E_h [h]` are continuous. Any of the 16
fixed/design combinations is supported; all fixed performs one evaluation.
Fixed geometry and technology inputs remain fixed. Battery energy remains
`b_P * b_E_h` in the unchanged assembly.

The search encodes an arbitrary active-variable list. Future variable DC load
would still require extending the existing runner/assembly interface; it is not
enabled by this implementation.

## I. Time budget and termination

The monotonic outer deadline starts before configuration, initialization, scouts,
sampling, surrogate work and evaluation. Existing worker caps apply to each CPLEX
solve: the lesser of its configured limit and remaining outer time. This is a
best-effort deadline: Python model construction, surrogate training, solver
cleanup and final validation cannot be interrupted exactly at 600 seconds.
The controller checks again after model training and before submitting work.
Only worker-sized batches are in flight; completed results are recorded before
propagating a deadline. The best validated feasible result is preserved.

The new loop stops at its deadline, `max_iter`, interruption, a diagnosed error,
or repeated empty new-candidate pools. The latter is a sampling-exhaustion reason,
not a proof of global infeasibility. `n_clusters`, `tol` and `min_conv_iter` remain
native-EGO-only settings. Neither method certifies global optimality.

## J. No local search

No incumbent-centred sampling, neighbour enumeration, turbine decrement rule,
special 21-turbine handling, narrowed bounds or monotonicity assumption was added.
All new designs originate in global LHS pools. Neighbouring integer designs can
still independently rank highest in those pools; this is not an added neighbour
search procedure. The native baseline algorithm remains unchanged.

## K. Tests

28 unittest tests passed, including the existing 11 tests and 17 new ones:
genuine cost preservation; excluded penalties/unknowns; label validation;
one-class, sparse, constant and failed fits; EI numerical edges; real native
regression plus classification in one and four dimensions; all 16 active-variable
subsets; integer/fixed preservation; global-only proposals; probability-independent
exploration; bounded retries; duplicate suppression; proposal logging before
evaluation; errors; deadlines during evaluation and surrogate work; retained best
results; native evaluator sign/DOE alignment; real short CPLEX reliability checks;
and notebook syntax/default configuration.

Command in `hydesign_clean`, from the repository root:

```text
python -B -m unittest hydesign.tests.test_search_simple_datacenter_with_hpp_sizing hydesign.tests.test_simple_datacenter_with_hpp_sizing -q
```

## L. Short validation results

Both methods used the same original broad wind-only case: initial 20 turbines,
bounds 0-400, PV 180 MW AC, battery 25 MW / 200 MWh, 16 MW load and 99% reliability.
Only runtime/search-size overrides were used: 90-second outer cap, 5-second limit
per CPLEX solve, six DOE points, two adaptive iterations and 1,000 predictions.

| Method | Seconds | Evaluations | Feasible / proven infeasible / unknown | Best turbines | Annual HPP cost |
|---|---:|---:|---|---:|---:|
| Native EGO | 78.938 | 12 | 10 / 1 / 1 | 36 | EUR 22,273,586.9794 |
| Feasibility-aware | 54.985 | 10 | 5 / 4 / 1 | 36 | EUR 22,273,586.9794 |

Both selected designs had 59 failed hours, reliability 99.326484% and EENS
607.606514 MWh. The 20-turbine scout timed out and was correctly excluded from
new-method training. Four adaptive EI proposals were made. The last fitted cost
model contained five genuine costs; its classifier contained five positive and
two negative labels. Two further negative labels were collected after that final
fit; final history contains all four. The adaptive proposals were 0, 1, 2 and 3
turbines, chosen by ranking the global pool. They were proven infeasible. No
special boundary information or known 21-turbine observation was supplied.

The selected new-method design was then evaluated directly with the unchanged
`hpp_model_simple_datacenter_with_hpp_cost`. Every hourly array and SOC value
matched exactly (maximum difference zero); all 33 economic fields and 11 metrics
passed the comparison. Unique proposal/evaluation records and independent final
validation were also checked. Windows spawn completed successfully.

This verifies execution and separation of the datasets, not superior convergence.
Both runs stopped at their deliberately short iteration limit. The short test
did not discover the independently known 21-turbine design. Full results are in
`feasibility_validation_with_hpp_sizing.json`; its run-directory fields locate
the original histories, proposals, dispatch arrays and configuration snapshots.

## M. First real ten-minute comparison

The committed YAML already has these settings:

```yaml
optimization_method: feasibility_aware
facility_load_mw: 16
pue: 1.15
reliability_target: 0.99
hpp_sizing_time_limit_seconds: 600
ems_time_limit_seconds: 60
ems_threads: 4
n_procs: 1
n_doe: 16
max_iter: 100
n_clusters: 2
npred: 2000
n_seed: 0
tol: 0.000001
min_conv_iter: 100
fa_batch_size: 2
fa_exploration_interval: 5
fa_max_retries: 1
fa_retry_interval: 5
fa_retry_cooldown_iterations: 3
fa_cost_theta_bounds: [0.000001, 20.0]
fa_classifier_length_scale: 0.2
fa_classifier_optimize: true
fa_classifier_max_iter: 100
```

Retain the existing `Nwt` design bounds `[0, 400]`, initial value 20, fixed PV
180 MW AC and battery 25 MW times 8 h. The YAML economics and physical assumptions
are unchanged. Use these settings first rather than tuning against the known answer.

## N. Comparing methods

Restart the notebook kernel to load the updated runner, then run
`docs/notebooks/simple_datacenter_with_hpp_sizing.ipynb` using `hydesign_clean`.
The default is `feasibility_aware`. For the comparison, change only
`config['optimization_method'] = 'native_ego'` in the configuration cell and rerun
the scenario/run/result cells. Use the same seed, bounds, physics, economics,
solver limits and 600-second budget. Each call creates a separate directory.

Compare best validated cost against elapsed seconds, accepted design/reliability,
unique evaluations, solver statuses, and the actual proposed designs. New-method
CSV columns separately report predicted cost, estimated feasibility, EI and
acquisition; these are never substituted for actual results. Do not compare
rejection scores as if they were real costs. Equal iteration counts need not mean
equal evaluation counts because native clustering and configurable new-method
batches differ. An elapsed-time comparison is the primary benchmark.

Repeat with matching alternative seeds later if needed; one run cannot establish
a consistent improvement. The independently evaluated 21-turbine result is a
reference for judging the outcome, never a search input.

## O. Remaining limitations

- Improvement over native EGO is unproven by this smoke test. Cost regression may
  extrapolate poorly and the classifier may be uncertain near a narrow feasible
  boundary. A true full-budget comparison remains necessary.
- Unknowns supply no training label; difficult cases may consume 60 seconds per
  solve, and an accepted dispatch requires both existing EMS solves. One bounded
  retry may still remain unresolved. No constraint has been relaxed to avoid this.
- Classification probabilities are heuristic predictions, not certified feasibility
  bounds. Early data may contain only feasible plants and give optimistic priors.
- Global pool ranking is approximate; it is not an exact acquisition maximizer.
  EI members of a batch are not conditioned on imagined outcomes of other members.
  No local refinement was introduced to compensate for these limitations.
- Only accepted feasible costs are available to the regressor. Economically cheap
  but infeasible designs add labels without cost data.
- Numerical issues are recorded with documented fallbacks. The installed native
  SMT sampling helper emits a `random_state` deprecation warning; native code was
  left untouched rather than changing that unrelated interface.
- The time cap is best effort. Multiple workers are supported, but CPU competition
  with CPLEX threads and concurrent completion order can affect runtime and the
  designs evaluated before the deadline. Start the comparison with one worker.
