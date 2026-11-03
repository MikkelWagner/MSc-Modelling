# Formulation-strengthening implementation and benchmark

## Scope and files

The annual integrated fixed-HPP sizing MILP is retained. No HPP sizing, Benders,
outer load search, representative periods, changed reliability metric, fixed primary
failure pattern, new battery operating assumption, or HPP economics was introduced.

New files:

- `hydesign/sizing_simple_datacenter_with_dc_cost_strengthened.py`: experimental
  copy of the existing sizing structure, with independent formulation switches.
- `scripts/benchmark_simple_datacenter_strengthening.py`: serial benchmark runner.
- `hydesign/tests/test_simple_datacenter_strengthened.py`: exactness/regression tests.
- This directory: derivations, usage, report, machine-readable measurements,
  CPLEX logs and validated hourly dispatch/SOC arrays.

The original sizing, EMS, assemblies, metrics, cost module, YAML and notebooks are
preserved. Input/source hashes are in the run manifest. The original module itself
supplies the baseline; no old solution is supplied to any primary solve.

## Exact mathematical changes and validity

Write `L` for facility MW, `u_t` for facility shortage, `z_t` for failure permission,
`p` for PUE, `r` for revenue per served MWh_IT, `c` for annual cost per installed
MW_IT, and `K=floor((1-R)T+1e-9)`. Here T=8760, K=87, p=1.15, r=4000 and
c=4,980,720.163340803 EUR/(MW_IT year).

**Upper bound / A.** Two fresh baseline LP relaxations give a maximum-load bound
and an economic objective bound. Every integer-feasible dispatch has at least T-K
fully served hours, so

```
V >= B L,  B = [r(T-K)-c]/p = 25,835,895.510138433.
```

The fresh economic LP gives V_LP=446,987,295.0182019 EUR. Therefore L<=V_LP/B
when B>0. The independently maximized load LP gives L<=17.36912661245 MW.
Taking their minimum with the original bound and adding outward numerical slack
gives **U=17.30101910649 MW**, replacing **23.18404623874 MW**.
If an LP fails to establish optimality its certificate is not used. If B<=0 the
economic quotient is skipped. Nothing is hard-coded from the old 30-minute bound.

These certified load bounds preserve feasible integer loads. Reusing the original
high-generation rule at the new U also fixes more failure binaries. This rule is
optimality-preserving dominance preprocessing: a deliberately short-supplied hour
with generation >= U + maximum charge can be fully supplied by reducing curtailment,
with identical battery operation and no decrease in objective. It does remove
dominated dispatches, so it is not literal preservation of every wasteful schedule.

**Aggregate row / B.** Add `sum(u_t) <= K L`, retaining the original Big-M rows and
failure-count constraint. With at most K positive shortages, each <=L, this is
redundant for every integer-feasible solution. It strengthens only the relaxation;
it does not replace hour-based reliability with an EENS requirement.

**Objective-based pruning / C.** Since full service gives an upper bound on value,

```
V <= A L,  A = [rT-c]/p = 26,138,504.20579061.
```

For A>0 and a valid same-case feasible objective V_inc, no load below V_inc/A can
match that objective. Using the explicitly supplied **413,046,614.91 EUR** and
downward EUR/MW slack gives **ell=15.80222741591 MW**. This is not a physical
minimum and is not a MIP start. It safely removes inferior feasible solutions,
provided the caller's certificate really belongs to this unchanged case.

Add `u_t <= L - ell(1-z_t)`. For z=0 the original Big-M row already forces u=0
and L>=ell; for z=1 the new row is the existing demand bound u<=L. The hourly row
therefore preserves all integer solutions in the retained interval. For A<=0,
the incumbent-derived lower-bound argument is skipped.

**Scaling / D.** Divide the primary objective by 1e6, but retain EUR reporting
and the secondary preservation constraint in EUR with its original EUR 1e-4
allowance. Convert the absolute MIP-gap parameter into the same objective units
and restore it for cleanup. Relative gap, simplex/integrality tolerances, threads,
search, presolve and cut settings are unchanged. No solver tuning was applied.

## Experimental design and interpretation

Each primary solve has a 60-second limit, four threads and a 0.0001 relative-gap
target. The same read-only progress listener and separate diagnostic LP clone are
used in every variant. LP certificates, construction, diagnostics and secondary
cleanup are outside the primary limit and reported separately. All-in wall time
is also retained. The 300-second test is another cold solve, not a continuation.

Variants C/F use an extra scalar incumbent certificate. They are explicitly
objective-pruning experiments, not information-neutral comparisons. No old load,
failure pattern, dispatch or SOC is passed as a primary MIP start. The standard
secondary solve can internally reuse its own primary solution, just as the baseline
does; this does not import a solution from another run.

G adds scaling to the better validated E/F combination. Here F found no incumbent
in 60 seconds, so **G = A+B+scaling**, without incumbent-based pruning or hourly rows.

Only one short run of each variant was performed. Differences in search behavior
and wall-clock cutoffs mean small timing/incumbent differences should not be treated
as statistically established speedups. The large LP-bound differences are directly
measured formulation effects. A time-limit candidate is never called optimal.

## Results

The completed tables and assessment are appended below. Raw evidence is in
[results.json](2026-10-23/results.json), [summary.csv](2026-10-23/summary.csv), and
[manifest.json](2026-10-23/manifest.json). Each variant also has a full CPLEX log.


### Primary outcomes

Money is million EUR/year; gap is CPLEX's relative primary gap. All short runs use 60 seconds.


| Variant | LP bound | Incumbent | Final upper bound | Gap | Change vs baseline (EUR) | Facility MW |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 446.987 | 409.650 | 444.950 | 8.6171% | 0.00 | 15.823532 |
| A | 427.463 | 411.939 | 426.889 | 3.6291% | 2,288,295.12 | 15.917534 |
| B | 423.915 | 410.272 | 423.915 | 3.3254% | 621,606.42 | 15.849722 |
| C | 442.751 | 410.575 | 426.033 | 3.7650% | 924,301.80 | 15.862082 |
| D | 446.987 | 411.216 | 445.468 | 8.3295% | 1,565,439.65 | 15.888978 |
| E | 423.915 | 411.284 | 423.915 | 3.0712% | 1,633,688.65 | 15.890497 |
| F | 423.915 | - | ~423.915* | undefined | - | - |
| G | 423.915 | 412.499 | 423.915 | 2.7676% | 2,848,597.47 | 15.940319 |
| G_long | 423.915 | 412.500 | 423.915 | 2.7674% | 2,849,245.46 | 15.940319 |


*F had no integer incumbent. Its last printed log bound is approximately EUR 423.915m; the initial runner did not retain the final API bound on exceptions. No exact final bound or gap is fabricated. The runner now retains that failure telemetry for future runs. G_long has a 300-second primary budget.


### Model size and preparation

Every model has 52,562 continuous variables and 8,760 declared binaries. Build+bound includes fresh LP-certificate work; it is outside the primary time limit. F has no returned formulation metadata, so its preparation time cannot be split from the recorded evidence.


| Variant | Constraints | Fixed binaries | Presolved binaries | Big-M MW | Lower MW | Build only s | Bound work s | Build+bound s | All-in wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 35043 | 4600 | 4160 | 23.184046 | 1.000000 | 2.323 | 0.000 | 2.323 | 67.498 |
| A | 35043 | 4945 | 3815 | 17.301019 | 1.000000 | 2.616 | 6.914 | 9.530 | 74.042 |
| B | 35044 | 4600 | 4160 | 23.184046 | 1.000000 | 2.770 | 0.000 | 2.770 | 68.045 |
| C | 43803 | 4600 | 4160 | 23.184046 | 15.802227 | 3.473 | 0.000 | 3.473 | 69.416 |
| D | 35043 | 4600 | 4160 | 23.184046 | 1.000000 | 2.594 | 0.000 | 2.594 | 67.668 |
| E | 35044 | 4945 | 3815 | 17.301019 | 1.000000 | 2.513 | 6.969 | 9.483 | 74.653 |
| F | 43804 | 4945 | 3815 | 17.301019 | 15.802227 | - | - | 10.283 | 75.198 |
| G | 35044 | 4945 | 3815 | 17.301019 | 1.000000 | 2.555 | 7.135 | 9.691 | 74.492 |
| G_long | 35044 | 4945 | 3815 | 17.301019 | 1.000000 | 2.535 | 6.740 | 9.275 | 314.272 |


### Root bounds and timing

Root-after-cuts is the last observed initial-root bound before branching. Crossing times are observed callback times; a final gap proves only an upper limit on crossing time if no callback recorded the crossing.


| Variant | Root after cuts (MEUR) | First incumbent s | <5% s | <2% s | <1% s | Proven optimum |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 446.616 | 0.300 | not reached | not reached | not reached | No |
| A | 427.195 | 0.280 | 7.766 | not reached | not reached | No |
| B | 423.915 | 0.260 | 29.469 | not reached | not reached | No |
| C | 440.295 | 47.625 | <=60.047 (final only) | not reached | not reached | No |
| D | 446.616 | 0.280 | not reached | not reached | not reached | No |
| E | 423.915 | 0.270 | 9.719 | not reached | not reached | No |
| F | 423.915 | - | not reached | not reached | not reached | No |
| G | 423.915 | 0.250 | 9.485 | not reached | not reached | No |
| G_long | 423.915 | 0.250 | 9.125 | not reached | not reached | No |


### Dispatch and cleanup validation


| Variant | Facility MW | IT MW | Failed hours | Reliability | Secondary status | Secondary s | Physical checks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 15.823532 | 13.759593 | 85 | 99.02968% | integer optimal solution | 0.297 | passed |
| A | 15.917534 | 13.841334 | 86 | 99.01826% | integer optimal solution | 0.281 | passed |
| B | 15.849722 | 13.782367 | 87 | 99.00685% | integer optimal solution | 0.500 | passed |
| C | 15.862082 | 13.793115 | 87 | 99.00685% | integer optimal solution | 0.281 | passed |
| D | 15.888978 | 13.816503 | 86 | 99.01826% | integer optimal solution | 0.297 | passed |
| E | 15.890497 | 13.817823 | 87 | 99.00685% | integer optimal solution | 0.281 | passed |
| F | - | - | - | - | not run | - | no candidate |
| G | 15.940319 | 13.861147 | 87 | 99.00685% | integer optimal solution | 0.265 | passed |
| G_long | 15.940319 | 13.861147 | 87 | 99.00685% | integer optimal solution | 0.328 | passed |


All eight returned dispatches passed the original renewable/power/demand/IT/PUE balances, SOC transitions and bounds, initial/terminal SOC, power limits, curtailment, and simultaneous-charge/discharge checks. They also met the 87-hour budget and 99% hourly reliability. The maximum reported balance residual was 8.17e-14. F supplied no candidate; it did not supply an invalid dispatch.


### What helped, what did not

- **B (one aggregate row) gave the largest isolated initial LP-bound improvement:**
  EUR 446.987m to EUR 423.915m, a reduction of EUR 23.072m. It adds no variables
  and does not alter the hourly reliability definition.
- **A reduced Big-M and binary count:** 23.184046 to 17.301019 MW; 4,600 to 4,945
  binaries fixed by the original rule; 4,160 to 3,815 binaries after presolve.
  It reached a <5% gap sooner than B, and found a stronger 60-second incumbent
  than B/E, although its upper bound remained worse.
- **E (A+B) helped incumbent search relative to B without further tightening the
  initial LP bound.** Its final bound remained EUR 423.915m, exactly the main
  residual relaxation barrier seen in B, F, G and the longer run.
- **C did not improve the aggregate-combination LP bound.** On its own it lowered
  the root LP only to EUR 442.751m, with further bound improvement during search.
  Its first incumbent arrived around 47.625 seconds. In F, the combination added
  8,760 rows and prevented a feasible incumbent being found within the short limit.
  This is a negative computational interaction, not evidence of mathematical invalidity.
- **D (scaling only) did not change the theoretical LP bound.** It found a better
  incumbent than baseline but produced a worse final upper bound. That small-gap
  improvement is search behavior, not a tighter formulation.
- **G (E+scaling) was the best short run:** EUR 412.499m, 2.7676% gap, 15.940319 MW.
  Scaling helped incumbent search in this one comparison, but did not move the bound.

### Longer cold run and practical conclusion

The fresh 300-second G run returned EUR 412,499,739.39/year at 15.940319 MW,
with a EUR 423,915,385.98 upper bound and **2.7674% gap**. Compared with the
60-second G run, the incumbent improved by only **EUR 647.995/year** and the bound
was unchanged to reporting precision. The primary ran about 300.156 seconds;
all-in runtime including bound preparation and diagnostics was 314.272 seconds.

Therefore:

- **<5%:** reached, including about 9.485 seconds in the 60-second G solve.
- **<2%:** not reached.
- **<1%:** not reached.
- **Proven optimum:** not reached by any primary solve. Secondary optimality only
  certifies cleanup of the selected primary load/failure permissions.

The fresh baseline was EUR 409.650m, 8.6171% gap and 15.823532 MW, consistent
with the earlier approximately EUR 409.65m/8.78% case within finite-runtime search
variation. Every successful strengthened short run improved its incumbent, but
none beat the user's earlier 30-minute **EUR 413.046615m** incumbent. That earlier
candidate was never supplied to the solver. If its feasibility for the same inputs
is retained, combining it analytically with the new bound gives a **2.6314%**
certificate; this is not the gap achieved by the cold benchmark solve itself.

**Keep A+B as the simple structural improvement.** G (A+B+scaling) is the best
observed operational configuration, with scaling still an option rather than a
claim of universal speedup. Do not enable C/F as the default for cold short solves.
A/B/G need no prior incumbent value. Use these explicit options:

```python
from hydesign.sizing_simple_datacenter_with_dc_cost_strengthened import (
    size_datacenter_with_dc_cost as size_datacenter,
)

# Add these keyword arguments to the existing size_datacenter(...) call:
# tighten_load_upper_bound=True,
# aggregate_shortage=True,
# hourly_strengthening=False,
# incumbent_objective_eur=None,
# primary_objective_scale=1e6,
```

Further MILP work should be limited to a small confirmation/tuning study if a
sub-2% certificate is important. This evidence does not justify automatically
investing in a more complicated decomposition or another long run. The main
thesis effort can move toward the planned HyDesign-style full HPP+DC co-design,
keeping this strengthened fixed-HPP model as a reference benchmark with explicit
bounds and gaps. Moving DC load to an outer optimizer alone will not eliminate
integer EMS difficulty if the same exact hourly-reliability requirement stays
inside dispatch; that architectural choice still needs to be made explicitly.

### Verification

**40 tests passed**: the new strengthening suite plus the existing technical and
cost-enabled simple-data-center suites. The source hash comparison confirms all
10 original code/input/notebook files in the manifest remained unchanged. The
experimental formulation source also matches the exact version benchmarked.
All primary models started with zero explicit MIP starts and their primary logs
contained no imported MIP start. Dynamic search remained enabled throughout.
See [verification.json](2026-10-23/verification.json).

