# Optional native refinement for feasibility-aware HPP sizing

Implemented on 2026-10-27. This is an isolated search extension. No full-year HPP
sizing benchmark or one-hour optimization was run as part of implementation.

## A. Implementation and native reuse

`hydesign/search_simple_datacenter_with_hpp_sizing.py` adds `NativeRefinement`.
It calls these existing functions from `hydesign/Parallel_EGO.py` directly:

- `extreme_around_point(x)`
- `perturbe_around_point(x, step=step)`
- `cast_to_mixint(physical, variables)`
- `expand_x_for_model_eval(physical, kwargs)`

The centre is `RunController.best`, which has passed the existing dispatch and
hourly reliability validation. Status must be `FEASIBLE`, acceptance must be true,
and the annualized cost must be finite and nonnegative. The latest centre is read
at every event; no candidates from an old centre remain in a queue.

Only active coordinates are normalized to [0, 1]. The native helpers create the
points; native casting restores integer turbine counts, and native expansion
inserts fixed inputs. Untouched continuous coordinates retain their original
bits to prevent encode/decode roundoff from defeating the exact-design cache.
Clipped duplicates, the centre itself and all cached designs are skipped. This
includes cached unresolved designs; only the existing global retry policy can
retry those designs.

Refinement never consults or filters on classifier predictions. It calls the
same `RunController.evaluate()` as global proposals. CPLEX limits, physical
validation, reliability acceptance and status classification are unchanged.

The published basis is HyDesign Section 3 / Algorithm 1:
https://wes.copernicus.org/articles/9/759/2024/
The scheduling below is a budget-limited adaptation of the native driver, not a
claim that the complete native EGO algorithm has been copied unchanged.

## B. Scheduling

Default refinement settings:

```yaml
fa_native_refinement_enabled: false
fa_native_refinement_interval: 5
```

When enabled, an event is considered after every fifth outer iteration's normal
global batch. No feasible incumbent means no refinement. Each event generates
up to 2d native points, where d is the number of active variables, and selects at
most the existing `fa_batch_size` (normally two) for evaluation.

At the first eligible event, use native coordinate extremes. At later events,
compare incumbent cost against its value at the previous event, before that
event's refinement evaluations. Use the native relative-change test
`abs(1 - previous_cost / current_cost) < tol`: stagnation selects perturbations;
otherwise select extremes. Zero cost has an explicit division-safe treatment.
Any improved incumbent becomes the new centre, even if its improvement is below
`tol`; `tol` only selects the candidate-generation mode. It does not stop search.

Perturbations retain native steps: uniform between 0.05 and 0.25 in normalized
coordinates. A private RandomState uses the native seed `n_seed * 100 + iteration`
without changing the global sampling/classifier random-number state.

The native helpers list one direction for all coordinates followed by the
opposite direction. For the batch cap, their output indices are interleaved into
opposing coordinate pairs and visited with a rotating cursor. The cursor persists
across events/centre updates, so later coordinates are not systematically omitted.
No candidate coordinates or technology-specific rules are invented by this ordering.

With full global batches of two and interval five, there are ten global proposals
for at most two extra refinement proposals. Thus at most about one sixth of
adaptive evaluations are refinement under the defaults, before cache exclusions
and incomplete batches. This is an allocation of evaluation counts, not a promise
about wall time: any candidate can take longer to evaluate. Global exploration
still occurs at every fifth GLOBAL adaptive proposal. Refinement does not advance
the existing global proposal counter. Native helper steps can be large on broad
search bounds; they are not an exhaustive integer-neighbour search.

## C. Budgets and compatibility

`max_iter` still counts outer global-search iterations. The monotonic outer
deadline and the existing per-CPLEX-solve cap have unchanged meanings. Refinement
is checked against the same deadline before and after generation and before
evaluation. Completed candidates remain recorded if a batch reaches a deadline.

An optional shared-controller limit makes equal-count comparisons executable:

```yaml
hpp_sizing_max_unique_evaluations: null  # no extra count limit
```

Set this to 216 for the controlled comparison. Initial scouts, DOE, global and
refinement candidates all count. New in-flight designs reserve count slots, so a
parallel batch cannot overshoot the limit. Cache reads cost no slots. Retrying an
unresolved design is another actual evaluation and consumes time, but not another
UNIQUE slot; both total attempts and unique designs remain reported. Termination
is `unique_evaluation_limit`. Other existing termination causes can occur first.

The time budget is still best effort: model construction, solver return and
cleanup can overrun it. The count cap cannot turn an unresolved result into a
negative feasibility label or an accepted result.

With refinement disabled and the count limit null/missing, the old sampling,
surrogate, acquisition, exploration and stopping paths remain in effect. Native
EGO remains available and never calls the refinement extension. It can also use
the explicitly requested controller count cap; with null it behaves as before.

## D. Files

Modified:

1. `hydesign/search_simple_datacenter_with_hpp_sizing.py`: optional native helper
   integration, conservative event scheduling and refinement diagnostics.
2. `hydesign/sizing_simple_datacenter_with_hpp_sizing.py`: optional unique-count
   cap, refinement history fields and result accounting, using the same evaluator.
3. `hydesign/examples/thesis_simple_datacenter/simple_datacenter_with_hpp_sizing.yml`:
   disabled-by-default toggle, interval and null count cap.

New:

4. `hydesign/tests/test_native_refinement_with_hpp_sizing.py`: unit, controller and
   synthetic complete-run smoke tests.
5. This report, `docs/benchmarks/native_refinement_implementation_with_hpp_sizing.md`.

No notebook, assembly, original optimizer, EMS, classifier kernel/hyperparameter
optimizer, acquisition equation, generation model, cost equation or economic
assumption was changed. No new package was installed. All new filenames contain
`_with_hpp_sizing`.

## E. Diagnostics

Existing history columns remain. New/refinement-specific columns include:

- `selection_reason`: `native_refinement_extreme` or `native_refinement_perturbation`.
- `refinement_event`, `refinement_coordinate`, `refinement_step` (null for extremes).
- `refinement_center_design`, `refinement_center_cost_eur`,
  `refinement_center_evaluation_index`, `refinement_center_changed`.
- `improved_incumbent`: whether this validated evaluation improved the best cost.
- `evaluation_elapsed_seconds`: parent-observed completion minus proposal time,
  including submission/return overhead; not CPLEX-only runtime.

Proposal JSONL records include centre metadata BEFORE evaluation. Prediction
fields are absent/null for refinement because predictions do not select these points.

`result['search_summary']['native_refinement']` reports events, generated native
points, selected proposals, skipped cache/duplicate points, centre changes,
completed unique evaluations, feasible/proven-infeasible/unknown/error counts,
incumbent improvements, generation overhead and controller evaluation wall time.
Selected proposals can exceed completed evaluations if a deadline stops submission.
`center_updates` counts changes between centres actually used at refinement events;
the first centre is initialization, not an update. `refinement_step` refers to the
normalized pre-rounding step, not the final physical change after clipping/rounding.

Both `result['native_refinement_enabled']` and the search summary identify the mode.
All existing best-design, dispatch, history and result output files are retained.

## F. Tests

Run from the repository root in `hydesign_clean`:

```powershell
python -B -m unittest hydesign.tests.test_search_simple_datacenter_with_hpp_sizing hydesign.tests.test_simple_datacenter_with_hpp_sizing hydesign.tests.test_native_refinement_with_hpp_sizing -v
```

The new suite uses analytic synthetic outcomes. Its end-to-end smoke tests exercise
the actual runner, search, surrogates, controller, annual physical/reliability
validation, output writing and termination, while substituting the expensive HPP
worker. The invented dispatch is explicitly not an HPP performance result.

The existing regression suite includes tiny two-hour CPLEX checks. No full-year
CPLEX evaluation is necessary to validate the refinement plumbing.

Final verification: **47 tests passed in 16.603 seconds** (28 existing regression
tests and 19 new tests). This includes all 16 active/fixed variable subsets,
integer/bound checks, cache and duplicate handling, centre changes, RNG isolation,
unknown-status handling, actual rejection of a purported feasible dispatch with
88 failed hours, deadline interruption, disabled-mode compatibility and the count
cap with a four-worker-sized submission batch. The worker is synchronous in the
synthetic tests; actual parallel CPLEX performance was not benchmarked.

| Synthetic complete-run smoke | Unique evaluations | Refinement evaluations | Feasible refinement | Refinement incumbent improvements | Runtime |
|---|---:|---:|---:|---:|---:|
| Wind only | 24 | 2 | 2 | 1 | 3.016 s |
| Four variables | 24 | 6 | 5 | 2 | 4.453 s |

Both terminated with `unique_evaluation_limit`, recorded proposal/history/result
files, continued global search after refinement and passed real annual dispatch
validation on synthetic hourly arrays. These tests used interval two to exercise
multiple events quickly. They are plumbing tests, not evidence of better HPP costs.
Generation overhead was about one millisecond per complete synthetic run. CPLEX
was only exercised by the existing tiny regression cases, not these smoke tests.

A SHA-256 audit of 251 existing source/configuration/notebook/report files found
only the three intended existing-file modifications listed above; the other 248
were unchanged. In particular, the original optimizer, all assemblies, EMS,
generation, metrics, costs and notebooks match the starting snapshot. The two
new files are the tests and this report. Documented Python snippets compile, and
the saved comparison configuration passes the current variable/settings checks.

## G. Exact controlled comparison

Use `docs/notebooks/simple_datacenter_with_hpp_sizing.ipynb`. Restart its kernel
after the code change. Keep its import/setup cell and result-display cells.

For the first comparison, use the historical run's CONFIGURATION only. This freezes
the same weather/economics/bounds and four active variables. It does not read prior
candidate histories, costs or winning designs as search inputs.

In the notebook's configuration cell, use:

```python
import json

baseline_config = (
    repo / 'hpp_sizing_runs'
    / '20261026T180012Z_ab67171d_with_hpp_sizing'
    / 'configuration_with_hpp_sizing.json'
)
config = json.loads(baseline_config.read_text())
config.update(
    optimization_method='feasibility_aware',
    facility_load_mw=16,
    reliability_target=0.99,
    hpp_sizing_output_dir=str(repo / 'hpp_sizing_runs'),
    hpp_sizing_time_limit_seconds=3600,
    hpp_sizing_max_unique_evaluations=216,
    max_iter=500,
    fa_native_refinement_enabled=False,  # Run A; change ONLY this to True for run B
    fa_native_refinement_interval=5,
)
variables = config['variables']
for name in ('Nwt', 'solar_MW [MW AC]', 'b_P [MW]', 'b_E_h [h]'):
    assert variables[name]['var_type'] == 'design'
assert config['n_doe'] == 16 and config['n_seed'] == 0
assert config['fa_batch_size'] == 2 and config['fa_exploration_interval'] == 5
assert config['ems_time_limit_seconds'] == 60 and config['n_procs'] == 1
print('Refinement:', config['fa_native_refinement_enabled'])
```

Then run the existing execution cell:

```python
result = run_hpp_sizing(config_path, overrides=config)
print(result['run_directory'])
```

Run A with False and B with True, in fresh runs. Retain both generated directories.
The historical reference design remains 20 turbines / 180 MW PV / 25 MW and 8 h;
it is only the existing feasibility scout, never assumed feasible. No known optimum
is injected. Do not change DOE, classifier, acquisition, bounds, CPLEX settings or
exploration settings between A and B.

Equivalent additional YAML settings for the four-variable comparison are:

```yaml
optimization_method: feasibility_aware
facility_load_mw: 16
reliability_target: 0.99
hpp_sizing_time_limit_seconds: 3600
hpp_sizing_max_unique_evaluations: 216
max_iter: 500
fa_native_refinement_enabled: false  # true only for B
fa_native_refinement_interval: 5
fa_batch_size: 2
```

Also set all four capacity entries' `var_type: design` if using the live YAML
instead of the saved configuration; preserve their existing values/types/bounds.
The live YAML still defaults to wind-only and a 600-second budget, so merely
turning on refinement is NOT the four-variable comparison.

Compare the minimum validated annualized cost after 216 unique evaluations and
the whole best-cost trajectory against elapsed time. Also inspect feasible hits
and incumbent improvements grouped by selection reason. Report actual attempts
alongside unique counts if retries occurred. A run stopped by time, error or
iteration limit before 216 is incomplete for the equal-count endpoint; compare
only its common completed prefix instead of claiming a 216-evaluation result.

To plot unique count explicitly from a history DataFrame:

```python
keys = ['Nwt', 'solar_MW', 'b_P', 'b_E_h']
history['unique_evaluations'] = (~history.duplicated(keys)).cumsum()
history.plot(x='unique_evaluations', y='best_feasible_cost')
history.plot(x='elapsed_seconds', y='best_feasible_cost')
display(history.groupby('selection_reason').agg(
    evaluations=('status', 'size'), feasible=('accepted', 'sum'),
    improvements=('improved_incumbent', 'sum'),
))
```

The historical 216-evaluation baseline was EUR 17.204m/year. It is a comparison
reference, not an optimization input or a claimed global optimum.

For a later full-hour comparison, use 3600 seconds, `max_iter: 500`, and set
`hpp_sizing_max_unique_evaluations: null` in BOTH arms. Time remains a ceiling;
500 iterations or other existing termination conditions can still stop earlier.

## H. Limitations and readiness

Coordinate refinement can miss coupled changes that require two variables to
move together. Native normalized steps are coarse on large bounds. It does not
repair classifier calibration, guarantee a feasible region will be found, prove
global optimality, or guarantee a lower cost than the baseline. Small scheduling
overhead does not imply small physical-evaluation cost. Sparse or cached global
pools can change the nominal global/refinement evaluation fraction.

The appropriate next action is the controlled experiment above, not further
classifier or objective changes in the same experiment.
