# Single-stage data-center EMS: implementation and validation
Date: 2026-10-30. Local hydesign_clean / full CPLEX. No outer HPP optimization was launched.

## Implemented objective and configuration
`battery_power_fluctuation_penalty: 0.0` is the default in all three thesis YAMLs.
Its value is lambda_B in MWh_IT per MW of adjacent AC net-battery-power change (hours).
With hourly steps:
- E_IT = sum(served_firm_it_mw) * 1 h.
- B[t] = discharge[t] - charge[t], on the AC/facility bus.
- F = sum(abs(B[t] - B[t-1])), t = 1,...,T-1 [MW]. No end-to-start ramp.
- Fixed-load/basic DC sizing: maximize E_IT - lambda_B * F [MWh_IT].
- Economic DC sizing: maximize Revenue_dc - annualized_cost_dc - r_IT * lambda_B * F [EUR].
- Economic r_IT is the existing datacenter_firm_revenue_per_mwh_it. If r_IT=0, the economic ramp penalty is zero.
- Strengthened sizing divides the COMPLETE penalized objective by primary_objective_scale.
- HPP outer sizing still minimizes annualized HPP cost.

PPA parameters remain accepted for backward compatibility and reporting only.
Neither PPA revenue nor its price scales the fixed/basic dispatch objective.
No artificial penalty is deducted from reported DC/system net annual value or added to HPP costs.

For lambda > 0 the existing EMS file creates T-1 nonnegative continuous ramp auxiliaries:
a[t] >= B[t]-B[t-1], a[t] >= B[t-1]-B[t].
At lambda=0 none are created. No binary variables or physical restrictions were added.
This penalizes changes in net power, NOT throughput, cycles, or degradation.

Each of the three dispatch solvers now calls CPLEX once. Primary-objective locks,
second-stage size/binary fixing, cleanup checks and the throughput objective were removed.
The strengthened workflow still performs its preparatory LP bound solves when enabled.

## Compatibility and reporting
Hourly arrays and their ordering are unchanged. solver_status.stages contains one item:
- objective / best_bound / relative_gap refer to the penalized objective;
- objective_units is MWh_IT for energy sizing and EUR for economic sizing;
- unpenalized_objective reports served IT energy or net DC value separately;
- battery_power_fluctuation_mw is computed from actual returned net flows;
- battery_penalty_equivalent_it_mwh or battery_penalty_eur reports the artificial penalty.
Economic primary_net_annual_value retains its unpenalized accounting meaning.
No obsolete second-stage lock tolerance or stages[1] access remains in active notebooks/scripts.

All eight simple-data-center notebooks were updated and their stale outputs cleared.
Each warns about the observed zero-penalty validation failure before running.
The benchmark script now reports one dispatch stage and uses a consistently penalized LP diagnostic.

The strengthened feasibility/load upper-bound LPs intentionally remain UNPENALIZED:
their derivation bounds the feasible set. incumbent_objective_eur must be a caller-certified
penalized value for the exact case, coefficient and revenue assumptions. Do not reuse
an old unpenalized certificate at positive lambda without recomputing it.
The current notebooks pass None for this certificate.

The physical constraint-builder function was compared directly with its pre-edit source
and is unchanged. metrics_simple_datacenter.py, native ems.py, native ems_constant_output.py,
cost components, annualization, reliability equations and native outer optimizers are unchanged.
Existing validator rejection is preserved.

## Small cases and regression tests
Before editing, 40 tests in the existing EMS, DC economics and strengthened suites passed.
After updating stage-specific assertions and adding regression coverage, the focused suite
passed all 44 tests:
```powershell
python -m unittest hydesign.tests.test_simple_datacenter hydesign.tests.test_simple_datacenter_with_dc_cost.SizingEconomicsTests hydesign.tests.test_simple_datacenter_strengthened -q
```
Set OPENMDAO_REPORTS=0 and use hydesign_clean.

Coverage includes:
- lambda = 0, 0.0001, 0.001; nonfinite/negative coefficient rejection;
- PPA independence (0, 70, 700 EUR/MWh) and one solve;
- adjacent signed-power linearization, one-hour horizon, zero extra binaries;
- no generation, surplus, zero battery, efficiency, power/SOC limits and boundary transitions;
- unchanged reliability, failed-hour rounding, CPLEX failure/timeout acceptance and gaps;
- DC accounting, EUR conversion, 1 and 1e6 objective scales, valid incumbent pruning;
- simultaneous-operation witness rejected by the unchanged validator at all three weights.

Representative exact/small optima:
| Case | Baseline primary result | Single-stage result at 0 / 0.0001 / 0.001 |
|---|---|---|
| [32,0] MW renewable, 16 MW facility, 25 MW/200 MWh battery, eta=sqrt(.86) | 25.878260870 MWh_IT (first solve); EENS ~2.24 MWh | Same served energy and EENS; F=29.76 MW |
| Economic version of that battery case, r_IT=100 | Unpenalized net value -2281.739130435 EUR | Same net value; penalties 0 / 0.2976 / 2.976 EUR |
| Four-hour economic toy [10,10,10,2], PUE=2, annual OPEX=350/MW_IT, r_IT=100 | 2 MW facility; net value 50 EUR | Same optimum |
| Basic 20-hour 95% case [10]*19+[2], no battery | 10 MW facility; one failed hour | Same zero-penalty optimum |

Old cleanup could reduce the first-solve objective by EUR 0.0001, so compare within that
tolerance as well as the CPLEX gap. This explains tiny positive differences after its removal.

A deliberate low-efficiency stress test (eta=0.01 per direction, not the thesis assumption)
with renewable [17,15.9999] MW shows that even lambda=0.0001 can reduce served IT energy:
loss = 0.0001/1.15 = 0.0000869565 MWh_IT; lambda=0.001 gives the same loss.
Positive penalties are a trade-off, not guaranteed harmless tie-breaks.

Broader final run: 111 test methods executed, with two failing assertions and two class-setup
errors (so the complete suite is NOT green):
1. test_cost_simple_datacenter expects a null discount rate, but YAML already contained 0.075
   before this work. No discount-rate value was changed.
2. test_simple_datacenter_with_hpp_cost expects its reliability target to equal the other YAML;
   the two files already contained 0.9999999 versus 0.999 before this work. Neither was changed.
3. FixedEvaluationEconomicsTests setup is rejected for simultaneous charging/discharging at lambda=0.
4. FixedEvaluationTests setup is rejected for the same reason.
The two reference-fixture failures were not hidden by weakening validation, changing the default,
or rewriting the fixtures to avoid the battery. These remain visible limitations.

All affected Python files and all eight notebooks compile. No notebook cells were executed as
a full sizing run. Existing synthetic search/refinement tests ran, not a full physical outer search.
The benchmark instrumentation was also exercised on the two-hour battery case, both unscaled
and scaled by 1e6 at lambda=0.001: LP bound and penalized incumbent both equalled
-2284.715130435 EUR; validation passed and no second-stage reporting was emitted.
Benchmark writes were mocked for this small instrumentation check.

## Annual fixed reference comparison
Denmark_good_solar, 8760 hours, 16 MW facility, PUE 1.15; design
[35,300,5,20,7,180,42,180,1.25,25,8]. Minimum reliability is DISABLED (R_target=0)
for this evaluation, matching the saved reference. The measured reliability is not a constraint here.

Saved baseline:
docs/benchmarks/simple_datacenter_with_hpp_cost_results.json, evaluation.
Baseline served IT = 120816.388108141 MWh_IT; EENS = 1221.153675638 MWh;
LOLH=126 (98.5616438%); throughput=29274.102261768 MWh;
curtailment=419164.944240744 MWh. Saved primary/cleanup solve times: 0.156 / 0.062 s.

| lambda | Served IT MWh | EENS MWh | LOLH | Reliability | F, MW | Throughput MWh | Simultaneous hours | CPLEX seconds | Validation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 120816.388109382 | 1221.153674 | 126 | 98.561644% | 24152.258 | 33529.114 | 39 | 0.172 | REJECTED |
| 0.0001 | 120816.388109382 | 1221.153674 | 344 | 96.073059% | 5758.974 | 33788.633 | 0 | 1.406 | Accepted |
| 0.001 | 120816.388109382 | 1221.153674 | 344 | 96.073059% | 5758.974 | 33788.633 | 0 | 1.157 | Accepted |

Zero-penalty values above are diagnostic only: the unchanged validator rejects the dispatch.
Maximum overlapping charge/discharge was 23.184046 MW. This is not solver roundoff.
At 0.0001 and 0.001, no simultaneous operation occurred for this particular reference.

All three cases preserve served energy to about 1.24e-6 MWh relative to the saved result
(~EUR 0.005 in DC revenue), but positive ramp smoothing spreads shortages across more hours:
126 -> 344, reliability 98.5616% -> 96.0731%, with unchanged EENS.
This does NOT weaken the reliability constraint: this case explicitly has no minimum constraint.
It demonstrates why energy and hour-based reliability must not be confused.

For the positive cases, F is 5758.974 MW, curtailment is 418825.140815 MWh and throughput
is 33788.633475 MWh. Smoother power does not imply lower throughput.
SOC starts/ends at 100 MWh. Maximum power/SOC residuals were below 6e-14.
Actual DC and HPP installed costs match the saved reference; artificial penalties are not accounting expenses.

## Annual HPP-sizing feasibility evaluation (no outer search)
Re-evaluated the saved design from
hpp_sizing_runs/20261027T093548Z_6fa9ea2c_with_hpp_sizing/best_with_hpp_sizing.json:
Nwt=7, PV=213.89280771226126 MW AC, b_P=46.413932947060964 MW,
b_E_h=12.984484530044755 h, all other design values as above.
Fixed facility 16 MW, PUE 1.15, R_target=0.99, max_failed_hours=87.
Each dispatch capped at 15 s, unchanged configured MIP relative-gap tolerance 1e-4.

| lambda | Validation | LOLH | Served IT MWh | F, MW | Simultaneous hours | CPLEX seconds | Penalized objective gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Rejected | 87 | 120691.747058 | 66220.633 | 182 | 7.234 | 0.001464% |
| 0.0001 | Accepted | 87 | 120692.541904 | 12873.316 | 0 | 8.375 | 0.000872% |
| 0.001 | Accepted | 87 | 120692.393287 | 12785.007 | 0 | 6.500 | 0.000985% |

At zero, the candidate is rejected, not counted as a feasible incumbent. The current adapter
classifies this validation exception as CONFIGURATION_OR_PROGRAMMING_ERROR. Its outer controller
can therefore stop rather than treating it as proven physical infeasibility. No status policy was weakened.
The rejected zero-penalty case overlaps in 182 hours, maximum 43.042511 MW.

Both positive cases passed physical validation and supplied the full load in 8673/8760 hours
(99.0068493%). EENS was 1363.576811 and 1363.747720 MWh respectively.
Initial/final SOC = 301.33049716482384 MWh; all power/SOC residuals below 4e-13.
Annualized HPP cost stays EUR 12344256.08681534 in both cases, exactly matching the saved cost.

Compared with the saved dispatch, served IT energy increases by 0.794847 MWh at 0.0001
and 0.646230 MWh at 0.001. These are tolerance-optimal MILP incumbents with nonzero gaps,
not proof that a penalty improves the unpenalized optimum. Relative to the 0.0001 result,
0.001 serves 0.148617 MWh_IT less (~EUR 594.47 less DC revenue); the gaps mean the
difference cannot be attributed solely to the mathematical penalty trade-off.

## Remaining limitation and recommended next decision
The requested zero-default, single-stage formulation is implemented, but it does NOT make
all existing annual notebooks runnable under the strict validator. Their zero-penalty reference
can stop with simultaneous charging/discharging, as demonstrated above.

A small positive lambda passed the two annual designs tested, but is not a general exclusivity
guarantee and may change failed-hour distribution. The default remains exactly zero.

For a guarantee at zero penalty, the simplest explicit physical solution is charge/discharge
mode exclusivity (one mode binary per hour with the existing charge/discharge bounds).
That requires a separately authorized model change and can increase MILP effort.
Restoring cleanup is another option but contradicts the requested single-stage workflow.
Neither remedy was implemented; no dispatch was repaired, accepted after failed validation,
or silently assigned a nonzero coefficient.

## Modified existing files
- hydesign/sizing_simple_datacenter.py
- hydesign/sizing_simple_datacenter_with_dc_cost.py
- hydesign/sizing_simple_datacenter_with_dc_cost_strengthened.py
- hydesign/sizing_simple_datacenter_with_hpp_cost.py
- hydesign/assembly/hpp_assembly_simple_datacenter.py
- hydesign/assembly/hpp_assembly_simple_datacenter_with_dc_cost.py
- hydesign/assembly/hpp_assembly_simple_datacenter_with_hpp_cost.py
- hydesign/assembly/hpp_assembly_simple_datacenter_with_hpp_sizing.py
- hydesign/ems/ems_simple_datacenter.py
- hydesign/tests/test_simple_datacenter.py
- hydesign/tests/test_simple_datacenter_strengthened.py
- hydesign/tests/test_simple_datacenter_with_dc_cost.py
- hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml
- hydesign/examples/thesis_simple_datacenter/simple_datacenter_with_hpp_cost.yml
- hydesign/examples/thesis_simple_datacenter/simple_datacenter_with_hpp_sizing.yml
- docs/notebooks/simple_datacenter.ipynb
- docs/notebooks/simple_datacenter_sizing.ipynb
- docs/notebooks/simple_datacenter_sizing_with_dc_cost.ipynb
- docs/notebooks/simple_datacenter_sizing_with_dc_cost_strengthened.ipynb
- docs/notebooks/simple_datacenter_sizing_with_hpp_cost.ipynb
- docs/notebooks/simple_datacenter_with_dc_cost.ipynb
- docs/notebooks/simple_datacenter_with_hpp_cost.ipynb
- docs/notebooks/simple_datacenter_with_hpp_sizing.ipynb
- scripts/benchmark_simple_datacenter_strengthening.py

Only this report is new; no duplicate EMS/assembly or new solver framework was created.
Historical benchmark outputs and sizing histories were left untouched.

