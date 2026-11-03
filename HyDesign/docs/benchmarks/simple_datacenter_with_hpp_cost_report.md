# HPP-cost extension implementation report

Implemented and validated on 2026-10-25 with the hydesign_clean environment and full CPLEX.

The native wind/PV/battery cost classes are reused through their existing OpenMDAO
wrappers. No native asset cost equation, EMS constraint, DC economics or strengthened
sizing formulation was changed. HPP costs are subtracted only in economic reporting.

## A?B. Files and preservation

Created files (all names contain with_hpp_cost):

- `hydesign/assembly/hpp_assembly_simple_datacenter_with_hpp_cost.py`
- `hydesign/costs/cost_simple_hpp_with_hpp_cost.py`
- `hydesign/sizing_simple_datacenter_with_hpp_cost.py`
- `hydesign/examples/thesis_simple_datacenter/simple_datacenter_with_hpp_cost.yml`
- `docs/notebooks/simple_datacenter_with_hpp_cost.ipynb`
- `docs/notebooks/simple_datacenter_sizing_with_hpp_cost.ipynb`
- `hydesign/tests/test_simple_datacenter_with_hpp_cost.py`
- `docs/benchmarks/simple_datacenter_with_hpp_cost_results.json`
- `docs/benchmarks/simple_datacenter_with_hpp_cost_report.md`

SHA-256 comparison checked all 454 pre-existing tracked and non-ignored untracked
repository files. No original file changed. New notebooks are saved without execution
outputs so no old results appear to belong to this new workflow.

## C?D. Configuration

All pre-existing simple_datacenter.yml values are retained, including the 99.9%
sizing reliability target, 60-second sizing limit, DC discount rate 7.5%, DC facility
life 20 years, IT life 5 years, and IT revenue EUR 4,000/MWh_IT.

The following values were copied from examples/Europe/hpp_pars.yml:

| Parameter | Value |
|---|---:|
| `wind_turbine_cost` | 640000 |
| `wind_civil_works_cost` | 260000 |
| `wind_fixed_onm_cost` | 12600 |
| `wind_variable_onm_cost` | 1.35 |
| `d_ref` | 145 |
| `hh_ref` | 100 |
| `p_rated_ref` | 5.0 |
| `solar_PV_cost` | 110000 |
| `solar_hardware_installation_cost` | 100000 |
| `solar_inverter_cost` | 20000 |
| `solar_fixed_onm_cost` | 4500 |
| `battery_energy_cost` | 62000 |
| `battery_power_cost` | 16000 |
| `battery_BOP_installation_commissioning_cost` | 80000 |
| `battery_control_system_cost` | 2250 |
| `battery_energy_onm_cost` | 0 |
| `wind_WACC` | 0.06 |
| `solar_WACC` | 0.06 |
| `battery_WACC` | 0.06 |

Original N_life=25 is represented as hpp_economic_lifetime_years=25, applying to
wind, PV and battery. Technical N_life remains 1. The source N_life value is not
assigned to the simulation horizon. This is a simplified annual recovery model,
not a lifetime degradation or replacement simulation.

No other source-YAML key was imported. Specifically, source keys not imported are:

- `G_MW`
- `year`
- `wpp_efficiency`
- `wind_deg_yr`
- `wind_deg`
- `share_WT_deg_types`
- `land_use_per_solar_MW`
- `tracking`
- `pv_deg_yr`
- `pv_deg`
- `latitude`
- `longitude`
- `altitude`
- `battery_depth_of_discharge`
- `battery_charge_efficiency`
- `battery_price_reduction_per_year`
- `min_LoH`
- `n_full_power_hours_expected_per_day_at_peak_price`
- `peak_hr_quantile`
- `price_H2`
- `storage_eff`
- `ptg_deg_yr`
- `ptg_deg`
- `hhv`
- `water_consumption`
- `electrolyzer_capex_cost`
- `electrolyzer_opex_cost`
- `electrolyzer_power_electronics_cost`
- `water_cost`
- `water_treatment_cost`
- `H2_storage_capex_cost`
- `H2_storage_opex_cost`
- `H2_transportation_cost`
- `H2_transportation_distance`
- `penalty_factor_H2`
- `electrolyzer_eff_curve_name`
- `electrolyzer_eff_curve_type`
- `min_power_standby`
- `hpp_BOS_soft_cost`
- `hpp_grid_connection_cost`
- `land_cost`
- `tax_rate`
- `ptg_WACC`
- `phasing_yr`
- `phasing_CAPEX`
- `inflation_yr`
- `inflation`
- `ref_yr_inflation`
- `depreciation_yr`
- `depreciation`
- `bi_directional_status`
- `penalty_BM`
- `era5_zarr`
- `ratio_gwa_era5`
- `era5_ghi_zarr`
- `elevation_fn`

Some names in that exclusion list already exist in the simple YAML as technical
settings; their existing simple-workflow values were retained, not overwritten from
the source YAML. In particular land_use_per_solar_MW remains necessary for the PV
component's technical area output; no land monetary cost is included.

The native battery class receives life_y=1 and healthy SoH=np.ones(8760). This
array horizon counts the initial purchase once and does not set economic life.
battery_price_reduction_per_year=0 is passed explicitly because replacements and
future price effects are excluded; the source 10% price reduction is not imported.
The initial-battery test also verifies that healthy SoH for 25 years still produces
one initial purchase even if a nonzero price decline is supplied.

## E?L. HPP results

All three CRFs are 0.078226718212274 at 6% over 25 years.

| Asset | Initial CAPEX EUR | OPEX EUR/year | Annualized CAPEX EUR/year | Annual total EUR/year |
|---|---:|---:|---:|---:|
| Wind | 92,670,295.51 | 1,706,923.90 | 7,249,293.09 | 8,956,217.00 |
| PV | 50,850,000.00 | 1,012,500.00 | 3,977,828.62 | 4,990,328.62 |
| Battery | 14,856,250.00 | 0.00 | 1,162,155.68 | 1,162,155.68 |
| HPP total | 158,376,545.51 | 2,719,423.90 | 12,389,277.40 | 15,108,701.30 |

Wind costs include native geometry scaling and variable OPEX from the available
wind generation profile before EMS curtailment. PV is 180 MW AC / 225 MW DC.
Battery CAPEX is EUR 14,856,250: initial energy capacity plus native power/BOP/control
costs, with no replacement. Shared HPP BOS is excluded; battery-specific BOP remains
included as explicitly requested.

## M?P. Evaluation, sizing and invariance

| Quantity | Fixed-load evaluation | DC sizing |
|---|---:|---:|
| Facility load MW | 16.000000 | 10.527556 |
| IT capacity MW_IT | 13.913043 | 9.154397 |
| Served IT energy MWh/year | 120,816.388108 | 80,119.280858 |
| DC revenue EUR/year | 483,265,552.432563 | 320,477,123.433488 |
| Annual DC cost EUR/year | 69,296,976.185611 | 45,595,488.761835 |
| DC-only net annual value EUR/year | 413,968,576.246952 | 274,881,634.671653 |
| System net annual value EUR/year | 398,859,874.947803 | 259,772,933.372505 |

Fixed-load achieved reliability: 98.56164384%;
LOLH 126. As in the original fixed-load notebook,
this 16 MW evaluation deliberately imposes no minimum reliability constraint.

Sizing reliability: 99.90867580%;
LOLH 8, meeting the YAML 99.9% target.
CPLEX status: integer optimal, tolerance. DC-only relative gap: 0.00949511%,
within the configured 0.01% tolerance. This is optimal within solver tolerance,
not an assertion of a mathematically zero gap.

The full-year old and new fixed-load assemblies have identical technical metrics,
DC economics, and hourly generation/dispatch/SOC (tested with absolute tolerance
1e-8 for arrays). Native raw cost outputs exactly equal standalone class calls.

During full-year notebook validation, the sizing wrapper's delegated result was
captured before cost reporting. DC size and every DC economic field were identical
after reporting; hourly arrays, metrics, solver status and formulation were the same
returned objects. No second optimization or objective offset was introduced.
A separate fully solved short case independently confirms the same 2 MW optimum
with/without reporting fixed HPP costs. Annual evaluation/sizing HPP costs match exactly.

net_annual_value retains its DC-only meaning. net_annual_value_dc_only is an explicit
alias. net_annual_value_system subtracts annualized_cost_hpp. The CPLEX bound and
gap still refer to DC-only value. Adding HPP reporting does not alter either solve.

## Q. Verification

11 new unittest tests passed, covering source/YAML parity, CRFs (including zero rate),
invalid economic input rejection, initial battery only, native raw cost parity,
excluded components, unchanged full-year technical/DC results, accounting,
correct sizing delegation, and an actual strengthened CPLEX optimum comparison.

Every code cell in both new notebooks executed, including their assertions and plot
construction, with Matplotlib's noninteractive Agg backend. Jupyter's notebook runner
was not used: nbformat is not installed. Notebook JSON and code-cell syntax were
checked; cells were executed sequentially as Python. The notebook files remain clear
of outputs. No package installation was required. pytest is not installed; tests ran
through the standard unittest runner.

Command (from repository root, hydesign_clean active):

```text
python -B -m unittest hydesign.tests.test_simple_datacenter_with_hpp_cost -v
```

## R. Evaluation path

```text
simple_datacenter_with_hpp_cost.ipynb
  -> hpp_model_simple_datacenter_with_hpp_cost(new YAML)
  -> evaluate(fixed HPP design)
  -> ABL / wind turbine / wake / wind / PV generation
  -> existing fixed-load EMS -> CPLEX -> existing reliability/physical checks
  -> existing annualized DC cost component
  -> native wpp_cost_comp / pvp_cost_comp / battery_cost_comp
  -> hpp_cost_annualized_comp
  -> economics: DC-only value and total system value
```

## S. Sizing path

```text
simple_datacenter_sizing_with_hpp_cost.ipynb
  -> new HPP-cost assembly: prepare generation + fixed HPP costs
     (same unconstrained preparation dispatch as the previous sizing notebook)
  -> size_datacenter_with_hpp_cost(hpp, existing sizing options)
  -> existing strengthened size_datacenter_with_dc_cost
  -> existing EMS constraints + bound/aggregate strengthening
  -> CPLEX: maximize DC net value, then minimize throughput with size/value locked
  -> existing physical/reliability validation
  -> wrapper adds fixed HPP cost fields and system value
```

The wrapper obtains generation and battery ratings from the evaluated assembly;
separate wind/PV/battery overrides are rejected. The notebooks retain variant G:
tighter load upper bound, aggregate shortage, objective scale 1e6, no incumbent
objective and no extra hourly strengthening. Sizing plots use the sizing result,
not the preparation dispatch.

## T. Remaining limitations

The chosen 25-year battery capital recovery period with no replacements is the
explicit simplified assumption for this stage. It is not a prediction of physical
battery life. The system value excludes shared/grid/land/decommissioning costs,
tax/inflation and lifetime cash-flow effects, so it is an annual-equivalent value
under the selected scope. A copied assembly must be maintained alongside its source;
this preserves original files and the existing repository structure as requested.
Native battery cost lacks energy-unit metadata; the new assembly resolves its shared
b_E input to MW*h locally without modifying the native component or EMS.

To run: select hydesign_clean, then Run All in either new notebook. They are independent.
