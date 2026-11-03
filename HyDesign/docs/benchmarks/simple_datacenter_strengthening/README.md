# Fixed-HPP data-center sizing: formulation experiment

The production sizing, EMS, assembly, economics, YAML and notebooks are unchanged.
The experimental module is `hydesign/sizing_simple_datacenter_with_dc_cost_strengthened.py`.
It copies the cost-enabled sizing structure and reuses the original EMS constraint
builder, extraction and physical validation. It does not introduce another EMS.

## Run

From the HyDesign repository root, in the full-CPLEX `hydesign_clean` environment:

```powershell
python -m unittest hydesign.tests.test_simple_datacenter_strengthened -v
python scripts/benchmark_simple_datacenter_strengthening.py --output docs/benchmarks/simple_datacenter_strengthening/new_run --incumbent-eur 413046614.91 --long-best 300
```

Use a fresh output directory. The numerical incumbent above is ONLY valid for the
unchanged case described by the saved manifest. It is explicitly supplied by the
caller, not inferred from YAML and not a MIP start. If inputs change, omit C/F/G
until a same-case feasible objective is available:

```powershell
python scripts/benchmark_simple_datacenter_strengthening.py --output docs/benchmarks/simple_datacenter_strengthening/no_certificate --variants baseline A B D E
```

`--seconds` defaults to 60 per primary solve. `--long-best 300` makes a fresh,
cold five-minute solve of the validated short-run variant with the smallest gap.
Runs are serial. The original module supplies the baseline. All variants use the
same read-only progress instrumentation and a separate LP clone for diagnostics.
No previous primal solution is supplied to a primary solve. The existing secondary
cleanup still re-solves its own primary model, as in production; CPLEX can internally
reuse that solve's incumbent. This is not a start from an earlier benchmark run.

## Mathematical changes

Let `L` be facility MW, `u[t]` facility shortfall, `z[t]` failure permission,
`T` the number of one-hour intervals, `K=floor((1-R)*T+1e-9)`, `p` PUE,
`r` the revenue per served MWh_IT and `c` annual cost per installed MW_IT.

### A: fresh upper-bound certificates

Build the baseline formulation with its original physical bound and relax its
failure binaries. Solve two LPs: maximize net value (`V_LP`), then maximize load
(`L_LP`). These LP solutions are never used as MIP starts.

At least `T-K` hours are fully served by every integer-feasible solution, so

```
V >= B*L,  B = (r*(T-K)-c)/p.
```

If `B>0`, the LP objective upper bound implies `L <= V_LP/B`. Independently,
`L <= L_LP`. Take the minimum with the original physical bound. Only LPs solved
to optimality supply certificates; otherwise that certificate is skipped.
Outward slack is at least EUR 1 on the economic bound and 1e-6 MW on load bounds.
This does not use the previous 30-minute solver bound or assume its incumbent is
optimal. It preserves integer-feasible loads under the original formulation.

Rebuild the same EMS using the tightened upper bound as Big-M. Its existing
preprocessing again fixes `z[t]=0` when `g[t] >= U + max_charge`.
This last step is **dominance preprocessing**, not literal preservation of every
intentionally wasteful schedule: a shortfall can be removed by reducing curtailment,
without changing battery operation, and with nondecreasing revenue. Consequently
the optimum is preserved. This distinction also applies to the original rule.

### B: aggregate shortage inequality

```
sum(u[t]) <= K*L
```

At most K hours fail and each shortfall is at most L. This is redundant for integer
solutions and strengthens the relaxation. The original hourly Big-M rows and
failure-count constraint remain. Reliability has NOT been replaced by EENS.

### C: objective-based lower bound and hourly inequality

Full-service value bounds every feasible objective from above:

```
V <= A*L,  A = (r*T-c)/p.
```

If `A>0` and the caller supplies a certified same-case feasible objective `V_inc`,
any optimum must satisfy `L >= V_inc/A`. Use EUR 1 and 1e-6 MW downward slack.
If `A<=0`, this argument gives no load lower bound and the physical minimum stays.
This removes inferior feasible solutions; it is **objective-based pruning**, not a
new physical minimum. Invalid or stale incumbent certificates can invalidate it.

For a valid effective lower bound `ell`, also add

```
u[t] <= L - ell*(1-z[t]).
```

For z=0 the original row already forces u=0 and L>=ell; for z=1 this is u<=L.
This row preserves all integer solutions within the retained load interval.
With no certificate, the same row can use the original physical minimum safely.

### D: objective units

Divide ONLY the primary objective by `primary_objective_scale` (1e6 for MEUR).
Return primary objectives and bounds in EUR. Keep the secondary net-value row and
its EUR 1e-4 allowance in EUR. Convert the absolute MIP-gap setting to the scaled
units, restoring it for cleanup; this preserves its EUR meaning, not solver tuning.
The relative gap, feasibility/integrality tolerances and all search settings stay.

## Variant switches

| Variant | Fresh upper bound | Aggregate row | Incumbent pruning + hourly rows | Primary scaling |
|---|---|---|---|---|
| baseline | No | No | No | No |
| A | Yes | No | No | No |
| B | No | Yes | No | No |
| C | No | No | Yes | No |
| D | No | No | No | Yes |
| E | Yes | Yes | No | No |
| F | Yes | Yes | Yes | No |
| G | Best validated E/F combination | As selected | As selected | Yes |

G selects E or F by the observed 60-second gap and records the exact options.
C/F (and G if it selects F) use extra scalar information from a previous feasible
objective. They are not completely information-neutral comparisons, even though
they supply no MIP start. A/B/D/E do not use that scalar information.

The experimental function has the original name `size_datacenter_with_dc_cost`.
Its added keyword options are `tighten_load_upper_bound`, `aggregate_shortage`,
`hourly_strengthening`, `incumbent_objective_eur` and `primary_objective_scale`.
The first three default to true, the certificate defaults to None and scale to 1.
For an isolated variant, explicitly set all switches as in the benchmark runner.

## Evidence

Each run directory contains a manifest (inputs, source/generation hashes, CPLEX
version), per-variant logs and JSON, `results.json`, `summary.csv`, and compressed
validated hourly/SOC arrays. Solver bounds and the primary gap always belong to
the first solve. The secondary status does not certify global sizing optimality.
The reported root-after-cuts bound is the last observed initial-root bound before
branching; missing telemetry is left blank rather than fabricated.

Construction time includes fresh bound generation; the bound's own elapsed time
is recorded separately in `formulation.bound_details.seconds`. LP diagnostics are
outside the 60-second primary budget and timed separately. All-in wall time includes
construction, certificates, diagnostics, primary solve, cleanup and validation.

Small-model tests enumerate integer failure patterns, compare optima and feasibility,
cover zero/full reliability and zero revenue, check the certificate sign conditions,
verify EUR-scaled reporting and cleanup, and assert no explicit MIP starts.
