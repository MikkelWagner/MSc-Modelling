# Fixed data-center load and HPP sizing: implementation record

Implemented on 2026-10-26 in the local HyDesign repository. The later instruction
overrides the attached prompt's 99.9% default: the new scenario uses **16 MW facility
load, 99% hourly reliability and a 600-second outer budget**. Initially only turbine
count is optimized. PV stays at 180 MW AC; battery stays at 25 MW / 200 MWh.

**A. Created files**

All paths below are relative to the repository root. No existing file was replaced.

| File | Purpose |
|---|---|
| `hydesign/assembly/hpp_assembly_simple_datacenter_with_hpp_sizing.py` | Subclass of the existing HPP-cost assembly; integer normalization, solve diagnostics, feasibility classification and optimizer score |
| `hydesign/sizing_simple_datacenter_with_hpp_sizing.py` | Native EGO runner, bounded evaluation scheduling, cost rejection bound, progress recording and independent final validation |
| `hydesign/examples/thesis_simple_datacenter/simple_datacenter_with_hpp_sizing.yml` | Separate scenario and optimizer configuration |
| `docs/notebooks/simple_datacenter_with_hpp_sizing.ipynb` | User entry point; reads YAML, optional overrides, runs sizing and displays results |
| `hydesign/tests/test_simple_datacenter_with_hpp_sizing.py` | Eleven small automated tests |
| `scripts/validate_with_hpp_sizing.py` | Guarded Windows-compatible controlled annual validation script |
| `docs/benchmarks/validation_with_hpp_sizing.json` | Measured results of controlled validation cases |
| `docs/benchmarks/implementation_with_hpp_sizing.md` | This report |

Validation also created the run artifacts listed under L. Each run directory and
each artifact filename includes `_with_hpp_sizing`. The complete generated-file
inventory is appended below after the preservation audit.

**B. Preservation**

A SHA-256 comparison against the pre-implementation snapshot verified **463
pre-existing repository files unchanged**, including tracked and nonignored
untracked thesis files. No existing source, EMS, optimizer, cost file, YAML or
notebook was edited. Temporary empty test directories were removed.

**C. Native HyDesign reused unchanged**

| File | Reused functionality |
|---|---|
| `hydesign/Parallel_EGO.py` | `EfficientGlobalOptimizationDriver.run`, mixed-integer design space/sampling, design-vector expansion, surrogate construction, acquisition/candidate search and convergence logic |
| `hydesign/assembly/hpp_assembly.py` | `hpp_base`: parameter and weather loading, common model infrastructure and reporting |
| `hydesign/weather/weather.py` | `ABL_comp` |
| `hydesign/wind/wind.py` | `get_rotor_d`, `genericWT_surrogate_comp`, `genericWake_surrogate_comp`, `wpp_comp` and their models |
| `hydesign/pv/pv.py` | `pvp_comp` and its PV model |
| `hydesign/costs/costs.py` | `wpp_cost`, `pvp_cost`, `battery_cost` and their OpenMDAO component wrappers |
| `hydesign/openmdao_wrapper.py` | Existing component wrapping used by the inherited assembly |
| `hydesign/examples/examples_sites.csv` and the selected weather CSV | Site lookup and the existing Denmark_good_solar weather profile |

Native EGO has no suitable total-runtime/feasibility recorder interface. A small
subclass of its `ParallelEvaluator` is bound temporarily while `driver.run()`
executes. The native search code itself is called unchanged. The binding is
restored on exit. A process-local lock prevents two concurrent searches from
changing this binding in the same Python process.

**D. Existing thesis files reused unchanged**

- `hydesign/assembly/hpp_assembly_simple_datacenter_with_hpp_cost.py`:
  `hpp_model_simple_datacenter_with_hpp_cost`; the entire OpenMDAO graph and
  evaluation method are inherited, not copied.
- `hydesign/ems/ems_simple_datacenter.py`: the component, annual constraints,
  CPLEX solve procedure and result extraction.
- `hydesign/metrics_simple_datacenter.py`: `firm_load_metrics` and
  `validate_dispatch`.
- `hydesign/costs/cost_simple_datacenter.py`: DC annualization and cost component.
- `hydesign/costs/cost_simple_hpp_with_hpp_cost.py`: HPP annualization.
- The existing `simple_datacenter_with_hpp_cost.yml` provided the physical and
  economic assumptions for the separate new YAML.

The previous integrated DC-sizing MILP and its strengthened variants are not called.
No EMS or cost module was duplicated for HPP sizing.

**E. Execution path**

```text
new notebook -> run_hpp_sizing(new YAML + optional overrides)
  -> validate design dictionary; derive native cost bound; start run recorder
  -> constrained reference / upper-bound feasibility scouts
  -> native EfficientGlobalOptimizationDriver.run()
       -> native DOE / surrogate / candidate proposals
       -> BudgetedEvaluator -> bounded process-worker batch
            -> new assembly adapter.evaluate(candidate)
                 -> inherited HPP-cost assembly.evaluate(candidate)
                      -> native wind/PV generation for that HPP design
                      -> existing fixed-load CPLEX EMS
                           1. maximize supplied energy with hard hourly reliability
                           2. preserve revenue and minimize battery throughput
                      -> native wind/PV/battery costs + existing annualization
                      -> existing DC economics and physical/hourly validation
                 -> classify result; expose negative HPP-cost score
            -> parent independently validates and records accepted candidates
       -> native optimizer proposes next HPP design
  -> independently validate saved best dispatch; save result metadata
```

Generation is recalculated for every candidate, including wake effects. No frozen
reference wind/PV arrays are substituted. Facility load and PUE remain fixed
within each run; CPLEX optimizes dispatch, not DC size.

**F. Configuration and variables**

The YAML uses native `var_type`, `value`, `limits`, and `types` fields. `value`
means the fixed value or initial feasibility-scout value. The four capacity
variables can independently switch between `fixed` and `design`.

| Exact native/assembly key | Default role | Value | Bounds | Type |
|---|---|---:|---|---|
| `Nwt` | design | 20 | 0 to 400 | int |
| `solar_MW [MW AC]` | fixed | 180 | 0 to 400 | float |
| `b_P [MW]` | fixed | 25 | 0 to 100 | float |
| `b_E_h [h]` | fixed | 8 | 1 to 24 | float |

Battery nominal energy remains `b_P * b_E_h`. Native integer sampling is retained;
only floating-point roundoff within 1e-9 of an integer is normalized in the new
adapter. Fractional turbine counts are rejected. Geometry, specific power,
turbine rating, density, PV angles and DC/AC ratio stay fixed at 35/300/5/7/42/180/1.25.
These fixed values remain configurable, but cannot be additional design variables
in this stage. Input names and order are checked against the full eleven-input
design vector. No search bounds are narrowed automatically.

All-fixed, wind-only, PV-plus-battery, and all-four-variable configurations pass
configuration/vector tests. The annual optimization validation exercises wind only.
The notebook reads the YAML without resetting edited settings; commented lines
provide optional per-run overrides. Legacy DC-sizing keys retained in the copied
YAML are explicitly marked unused by this HPP-sizing runner.

**G. Hard reliability and rejected candidates**

The existing EMS still enforces its failed-hour binary budget. The initial case
allows `floor((1 - 0.99) * 8760 + 1e-9) = 87` failed hours. Independent acceptance
also checks the original physical constraints and recomputes hourly reliability
from `unserved_facility_mw` using the existing `epsilon_mw = 0.001` convention.
EENS does not replace this constraint. The same configuration accepts higher
targets up to 1.0; no automatic relaxation occurs.

Native EGO requires a finite scalar score. The rejection bound is obtained by
calling the unchanged native cost classes with the maximum configured capacities
(or fixed values) and fixed geometry. Wind variable OPEX uses 8,760 hours of
nameplate generation as a conservative bound, allowing for the configured WPP
efficiency multiplier. Nonnegative cost coefficients are required. PV and battery
costs use the maximum configured AC capacity and power-duration product. The
existing annualization then produces `C_bound`.

```text
accepted score = -annualized_cost_hpp / 1e6
rejection score = -(C_bound + max(1 EUR/year, 0.01*C_bound)) / 1e6
```

Every accepted cost is checked against `C_bound`; a breach stops the search as a
configuration/model error. In the default wind-only domain the bound is
199,990,346.17 EUR/year and rejection score is -201.99024963034242. Thus rejected
results cannot outrank a validated design within this domain. This is an outer
surrogate compatibility mechanism, not an economic trade-off with reliability.
The bound assumes the native turbine power curves respect rated power; changed
generation/cost models require rechecking that assumption.

The reference is evaluated with the actual requested constraint before it can
become an incumbent. An upper-domain scout varies only active design variables.
If neither scout succeeds, a small native mixed-integer DOE explores further.
If no feasible result is found, the runner returns no best design and a clear
termination reason, rather than fitting a surrogate to uniformly rejected scores.
This does not establish that the entire domain is infeasible.

**H. Solver diagnostics and conservative timeout handling**

The adapter temporarily observes the existing EMS's `Model.solve` through a
subclass, using structured CPLEX status codes. It does not parse exception text
to decide physical feasibility and does not change constraints or objectives.

| Status | Handling |
|---|---|
| `FEASIBLE` | Both existing EMS stages and physical/hourly validation completed; eligible for incumbent |
| `PROVEN_INFEASIBLE` | Primary CPLEX solve reports infeasible; finite rejection score |
| `TIME_LIMIT_WITH_FEASIBLE_DISPATCH` | Solver reports a solution, but the complete existing solve/validation route did not finish; conservatively rejected, separately diagnosed |
| `TIME_LIMIT_UNRESOLVED` | No solver solution available before timeout; rejected without claiming physical infeasibility |
| `NUMERICAL_ERROR` | Other numerical/solver failure, including infeasible cleanup after a successful primary solve; rejected and diagnosed |
| `CONFIGURATION_OR_PROGRAMMING_ERROR` | Invalid inputs, model/validation errors or unexpected implementation failure; stop the search, preserving an earlier validated incumbent |

A successful primary solve followed by failed cleanup must not be labelled
physical infeasibility of the original problem. Tests cover this distinction.
Rejected timeout results are cached within the run and not automatically retried.
Increasing the inner limit and starting a new run may resolve them.

**I. Objective and economic accounting**

The engineering objective is minimum positive `annualized_cost_hpp`. Native EGO
maximizes the separate `hpp_sizing_score [MEUR/year] = -annualized_cost_hpp/1e6`.
Its internal minimization sign is handled by the unchanged driver interface.
Tests verify cheaper HPPs rank higher and rejection ranks below feasible scores.

DC revenue remains 4,000 EUR per served MWh_IT, with installed IT capacity
`facility_load_mw / pue`. DC annualization retains 7.5%, 20-year facility and
5-year IT recovery. HPP costs retain 6% WACCs and 25-year recovery. Reported values
include component CAPEX/annual costs, DC revenue/cost, DC-only annual value and
`net_annual_value_system = Revenue_dc - annualized_cost_dc - annualized_cost_hpp`.
DC economics do not enter the outer objective. No shared/grid/land/decommissioning
costs, degradation or battery replacements were added.

The existing positive PPA coefficient remains only the EMS served-energy weight.
For fixed load, maximizing that energy minimizes EENS. The original secondary
battery-throughput solve is retained. No outer EENS weight, tie-break rule or
lexicographic stage was added.

**J. Outer time limit**

`hpp_sizing_time_limit_seconds` is a **best-effort wall-clock budget**, starting
before run configuration, feasibility scouts and native optimization. Before
submitting each worker-sized batch, the runner checks the deadline. Workers also
check before construction/evaluation and cap each CPLEX solve by remaining time.
Native surrogate calls are checked before and after execution. No long task queue
is submitted beyond the available worker batch.

Active construction, surrogate work, CPLEX return/cleanup, serialization and final
validation can overrun the deadline. They are not forcibly killed. A notebook
KeyboardInterrupt stops submission, drains the bounded active batch and preserves
completed feasible work; that can also take time. A kernel/process kill cannot
guarantee orderly cleanup, though already written history/incumbents remain.
Native EGO may stop earlier because of its iteration/convergence settings.
No result is labelled a globally proven optimum.

**K. Inner CPLEX limit**

The new development YAML uses `ems_time_limit_seconds: 60`, separately from the
600-second outer budget. This gives the initial wind-only search time for multiple
candidates instead of letting a difficult reference evaluation consume the entire
outer budget. Existing YAML files and EMS defaults were not changed.

In the existing EMS this is a limit **per solve**, not a strict total per candidate:
there are two sequential solves, plus generation/model/validation overhead. Each
is additionally capped by remaining outer time. Thus a candidate can require up
to approximately two inner limits plus overhead if enough outer time remains.
`ems_threads: 4` and `n_procs: 1` are conservative initial defaults; consider their
product when increasing parallelism. Only CPLEX is used.

**L. Progress and persisted results**

The parent process is the sole writer. Runs use UTC timestamp + UUID directories.

- `configuration_with_hpp_sizing.json`: effective configuration.
- `history_with_hpp_sizing.csv`: completed evaluation index/time/status, all four
  capacities plus derived wind MW/battery MWh, score, costs, EENS, reliability,
  failed hours, DC/system economics, best feasible design/cost so far and diagnostics.
- `best_with_hpp_sizing.json`: most recent best validated candidate.
- `dispatch_with_hpp_sizing.npz`: its hourly arrays and 8,761 SOC states.
- `result_with_hpp_sizing.json`: termination, elapsed time, counts, configuration,
  cost bound, bound-hit diagnostics, selected design and independent validation.
- `native_final_with_hpp_sizing.csv`: native EGO's optional final output when it
  completes normally. The native driver re-evaluates its selected point; this
  extra evaluation is deadline-limited but is not part of the controller history.
  The saved controller best/result/dispatch are authoritative.

No hourly file is written when no incumbent exists; economic/technical fields for
failed candidates remain empty rather than inventing feasible results. Exact
candidate results are cached in memory within a run. There is no cross-run resume.

**M. Verification**

Eleven unittest tests pass in `hydesign_clean`. They cover variable combinations,
exact names/vector/energy mapping, integer handling, objective/bound direction,
notebook code compilation, structured solver classifications, cleanup failure,
real small-hour CPLEX feasibility/infeasibility, deadline submission checks,
history/cache/incumbent retention and seeded native DOE alignment/sign.

Controlled annual checks in `validation_with_hpp_sizing.json` also passed:

| Check | Result |
|---|---|
| A: unchanged reference regression | Identical hourly arrays (absolute tolerance 1e-8) and identical economics versus the existing HPP-cost assembly; reliability 98.5616438356% |
| B: no generation/storage at target 99% | CPLEX status 103, `PROVEN_INFEASIBLE` |
| C: injected structured timeout through real assembly/EMS route | `TIME_LIMIT_WITH_FEASIBLE_DISPATCH`; not accepted and not labelled infeasible |
| D: short native EGO search | Five completed candidate evaluations, three accepted, no runtime error |
| E1: 0.01-second budget | Zero candidates launched; safe no-result termination |
| E2: 25-second budget, two Windows spawn workers | Stopped after 26.08 seconds, retained independently validated incumbent |
| E3: deliberately infeasible wind-only 0..1 turbine domain without PV/storage | Both distinct candidates proven infeasible; clean no-best result |

Check A explicitly sets reliability target zero only to reproduce the earlier
unconstrained reference calculation. It is a regression diagnostic and never
seeds the optimization. All sizing validation runs use the requested 99% target.
Unit tests use a different target only where the small synthetic case requires it.
The unit test suite does not launch an annual outer optimization.

Run small tests from the repository root with:

```text
python -B -m unittest hydesign.tests.test_simple_datacenter_with_hpp_sizing -v
```

The controlled annual validation script is separate and guarded for Windows.
No full 600-second optimization was launched during implementation.

**N. Short optimization results**

The 90-second-budget check actually finished in 51.95 seconds using `n_doe=3`,
`max_iter=1`, `npred=300`, one worker and 15 seconds per inner solve. Its best
candidate was 229 turbines, 180 MW AC PV and 25 MW / 200 MWh battery:
19 failed hours, 99.783105% reliability, 188.694 MWh EENS and
108,564,028.80 EUR/year annual HPP cost. The reference with 20 turbines timed out
without an incumbent after about 15 seconds; its constrained feasibility remained
unresolved and it was not accepted.

The separate two-worker 25-second deadline test found a better validated candidate
before timing out: **85 turbines (425 MW wind)**, unchanged PV/battery, **37 failed
hours**, **99.577626% reliability**, **326.122 MWh EENS**, and **44,188,776.71 EUR/year**
annual HPP cost. Six of eight completed evaluations were accepted (the rich
reference and upper scout coincided and were both evaluated in that parallel
batch). This test establishes preservation of a feasible incumbent at the deadline.

These deliberately small searches demonstrate operation, not minimum necessary
wind capacity or economic optimality. Neither large wind design is recommended as
the final thesis optimum. Lower-wind candidates near the feasibility boundary can
be substantially harder for the annual binary EMS.

**O. First real notebook run**

Open `docs/notebooks/simple_datacenter_with_hpp_sizing.ipynb`, select
`hydesign_clean`, and Run All. Review the printed Python/repository paths.

Keep the new YAML defaults initially: 16 MW, PUE 1.15, reliability 0.99,
outer 600 seconds, inner 60 seconds per solve, Nwt design in 0..400, PV/battery
fixed at 180/25/8, `n_doe=8`, `max_iter=3`, `n_clusters=1`, `npred=1000`,
`n_procs=1`, `n_seed=0`, `tol=1e-6`, `min_conv_iter=2`.
The notebook prints the 87-hour budget, all effective settings, final economics,
technical results, bound hits, stopping reason and candidate history.

**P. Remaining limitations**

The outer search is a heuristic surrogate search and provides no global optimality
gap/certificate. A finite rejection plateau is necessary for this native driver
and may be difficult for its surrogate. Conservative timeouts can hide good
designs; no-feasible-result is not proof of global infeasibility. The upper-bound
scout is not guaranteed feasible and is not a recommendation to build that size.
Only wind-only annual searches have been benchmarked; other variable combinations
passed configuration tests but need their own run-time evaluation. The deadline is
best effort, and native surrogate work cannot be interrupted precisely. Temporary
module bindings require one search per Python process and serial candidate work
inside each worker. Run outputs are checkpoints for inspection, not restart files.

**Q. Future HPP + DC co-design**

Generation, dispatch and both cost layers remain separate and unchanged. A later
stage can extend the outer design-vector adapter to pass candidate facility load
into the inherited assembly and change only the optimizer score to net annual
system value. DC installed costs already follow facility load/PUE. That stage
will need an explicit additional variable/bounds, corresponding configuration
validation and a new defensible rejection bound. Variable DC capacity and full
co-design were not implemented here.

**Generated validation artifact inventory**

Paths relative to the repository root. The first configuration-only directory
records an attempted validation whose multiprocessing launch was blocked by the
execution sandbox; the later runs used reviewed Windows process access.

```text
docs/benchmarks/runs_with_hpp_sizing/20261026T113903Z_7bc20a11_with_hpp_sizing/configuration_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T113958Z_5281de34_with_hpp_sizing/best_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T113958Z_5281de34_with_hpp_sizing/configuration_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T113958Z_5281de34_with_hpp_sizing/dispatch_with_hpp_sizing.npz
docs/benchmarks/runs_with_hpp_sizing/20261026T113958Z_5281de34_with_hpp_sizing/history_with_hpp_sizing.csv
docs/benchmarks/runs_with_hpp_sizing/20261026T113958Z_5281de34_with_hpp_sizing/native_final_with_hpp_sizing.csv
docs/benchmarks/runs_with_hpp_sizing/20261026T113958Z_5281de34_with_hpp_sizing/result_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114050Z_92da30b2_with_hpp_sizing/configuration_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114050Z_92da30b2_with_hpp_sizing/result_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114430Z_35701dd6_with_hpp_sizing/best_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114430Z_35701dd6_with_hpp_sizing/configuration_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114430Z_35701dd6_with_hpp_sizing/dispatch_with_hpp_sizing.npz
docs/benchmarks/runs_with_hpp_sizing/20261026T114430Z_35701dd6_with_hpp_sizing/history_with_hpp_sizing.csv
docs/benchmarks/runs_with_hpp_sizing/20261026T114430Z_35701dd6_with_hpp_sizing/native_final_with_hpp_sizing.csv
docs/benchmarks/runs_with_hpp_sizing/20261026T114430Z_35701dd6_with_hpp_sizing/result_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114522Z_6a056c8e_with_hpp_sizing/configuration_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114522Z_6a056c8e_with_hpp_sizing/result_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114523Z_e4ad5cca_with_hpp_sizing/best_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114523Z_e4ad5cca_with_hpp_sizing/configuration_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114523Z_e4ad5cca_with_hpp_sizing/dispatch_with_hpp_sizing.npz
docs/benchmarks/runs_with_hpp_sizing/20261026T114523Z_e4ad5cca_with_hpp_sizing/history_with_hpp_sizing.csv
docs/benchmarks/runs_with_hpp_sizing/20261026T114523Z_e4ad5cca_with_hpp_sizing/result_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114549Z_46ce3f3e_with_hpp_sizing/configuration_with_hpp_sizing.json
docs/benchmarks/runs_with_hpp_sizing/20261026T114549Z_46ce3f3e_with_hpp_sizing/history_with_hpp_sizing.csv
docs/benchmarks/runs_with_hpp_sizing/20261026T114549Z_46ce3f3e_with_hpp_sizing/result_with_hpp_sizing.json
```
