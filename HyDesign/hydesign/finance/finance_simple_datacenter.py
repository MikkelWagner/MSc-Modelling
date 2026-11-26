"""Integrated HPP/DC lifetime reporting; no EMS or sizing objective changes.

All assumptions are supplied by the shared YAML. Annual operation is repeated,
without degradation. Cash flows use years 0..T, initial commissioning at zero,
and end-of-year operating cash flows. Replacement purchases at year y serve
years y+1 onward. No purchase occurs at the terminal project boundary.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import numpy_financial as npf

from hydesign.finance.finance import calculate_NPV_IRR_phased, get_inflation_index
from hydesign.openmdao_wrapper import ComponentWrapper


FINANCE_PARAMETERS = (
    "hpp_economic_lifetime_years", "wind_life_y", "solar_life_y",
    "battery_life_y", "shared_life_y", "datacenter_facility_lifetime_years",
    "datacenter_it_lifetime_years", "project_discount_rate", "tax_rate",
    "battery_replacement_year", "battery_replacement_cost_fraction",
    "datacenter_it_replacement_cost_fraction", "inflation_yr", "inflation",
    "ref_yr_inflation", "depreciation_method", "residual_value_fraction",
    "phasing_yr", "phasing_CAPEX",
)
ASSETS = ("w", "s", "b", "sh", "dc_facility", "dc_it")
CAPEX_INPUTS = tuple("CAPEX_" + asset for asset in ASSETS)
OPEX_INPUTS = ("OPEX_w", "OPEX_s", "OPEX_b", "OPEX_sh", "OPEX_dc")
INPUT_NAMES = CAPEX_INPUTS + OPEX_INPUTS + (
    "served_it_energy_mwh", "datacenter_firm_revenue_per_mwh_it",
)
SCALAR_OUTPUTS = (
    "CAPEX_system", "OPEX_system", "Revenue_dc", "NPV_system",
    "NPV_over_CAPEX_system", "IRR_system", "payback_years", "discounted_payback_years",
    "CAPEX_lifetime_system", "CAPEX_PV_system", "CAPEX_replacement_PV_system",
    "NPV_over_PV_CAPEX_system",
)
VECTOR_OUTPUTS = tuple("CAPEX_" + asset + "_vec" for asset in ASSETS) + (
    "finance_years", "CAPEX_system_vec", "replacement_battery_vec", "replacement_it_vec",
    "OPEX_system_vec", "Revenue_dc_vec", "depreciation_vec", "taxes_vec",
    "residual_value_vec", "cashflow_vec", "discounted_cashflow_vec",
    "cumulative_discounted_cashflow_vec",
)
OUTPUT_NAMES = SCALAR_OUTPUTS + VECTOR_OUTPUTS


def _number(value, name, *, integer=False):
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    value = float(array.item())
    if not np.isfinite(value) or value < 0 or (integer and value != int(value)):
        raise ValueError(f"Invalid {name}: {value}")
    return int(value) if integer else value


def _payback(cashflow):
    """First recovery after a negative cumulative balance; NaN if not reached.

    Linear interpolation within the crossing year is a reporting convention.
    Subsequent replacements can cause another negative cumulative balance.
    """
    cumulative = np.cumsum(cashflow)
    for year in range(1, len(cumulative)):
        if cumulative[year - 1] < 0 <= cumulative[year]:
            return float(year - 1 - cumulative[year - 1] / cashflow[year])
    return np.nan


class finance_simple_datacenter:
    """Native-style compute model for one integrated owner, using annual vectors.

    Native phased tax policy is retained. Residual proceeds are explicitly NET
    of disposal tax and are a configured fraction of initial integrated CAPEX.
    IRR uses numpy_financial directly because native finance returns zero for
    nonpositive NPV. Undefined IRR/payback and zero-CAPEX ratios return NaN.
    For cash flows with multiple sign changes, IRR can be nonunique; inspect
    the cash-flow vector and use NPV as the primary valuation metric.
    """

    def __init__(
        self, hpp_economic_lifetime_years, wind_life_y, solar_life_y,
        battery_life_y, shared_life_y, datacenter_facility_lifetime_years,
        datacenter_it_lifetime_years, project_discount_rate, tax_rate,
        battery_replacement_year, battery_replacement_cost_fraction,
        datacenter_it_replacement_cost_fraction, inflation_yr, inflation,
        ref_yr_inflation, depreciation_method, residual_value_fraction,
        phasing_yr, phasing_CAPEX,
    ):
        self.project_life_y = _number(hpp_economic_lifetime_years, "hpp_economic_lifetime_years", integer=True)
        if self.project_life_y < 1:
            raise ValueError("Project lifetime must be positive")
        self.lifetimes = dict(zip(ASSETS, [
            _number(value, name, integer=True) for name, value in zip(ASSETS, (
                wind_life_y, solar_life_y, battery_life_y, shared_life_y,
                datacenter_facility_lifetime_years, datacenter_it_lifetime_years))]))
        if min(self.lifetimes.values()) < 1:
            raise ValueError("Asset lifetimes must be positive")
        self.discount_rate = _number(project_discount_rate, "project_discount_rate")
        self.tax_rate = _number(tax_rate, "tax_rate")
        self.residual_fraction = _number(residual_value_fraction, "residual_value_fraction")
        if self.tax_rate > 1 or self.residual_fraction > 1:
            raise ValueError("Tax and residual fractions must lie in [0, 1]")
        self.battery_replacement_year = _number(battery_replacement_year, "battery_replacement_year", integer=True)
        if self.battery_replacement_year < 1:
            raise ValueError("Battery replacement year must be positive")
        self.battery_fraction = _number(battery_replacement_cost_fraction, "battery_replacement_cost_fraction")
        self.it_fraction = _number(datacenter_it_replacement_cost_fraction, "datacenter_it_replacement_cost_fraction")
        if depreciation_method != "straight_line":
            raise ValueError("Phase 1 supports depreciation_method=straight_line")
        self.years = np.arange(self.project_life_y + 1)
        inflation_years, rates = np.asarray(inflation_yr, dtype=float), np.asarray(inflation, dtype=float)
        if (inflation_years.ndim != 1 or not inflation_years.size
                or rates.shape != inflation_years.shape or not np.isfinite(inflation_years).all()
                or not np.isfinite(rates).all() or np.any(rates <= -1)
                or np.any(np.diff(inflation_years) <= 0)):
            raise ValueError("Invalid inflation curve")
        if _number(ref_yr_inflation, "ref_yr_inflation", integer=True) != 0:
            raise ValueError("Initial CAPEX and revenue use project year zero as the price reference")
        self.inflation_index = get_inflation_index(self.years, inflation_years, rates, ref_yr_inflation)
        self.payment_years = np.asarray([_number(y, "phasing_yr", integer=True) for y in phasing_yr])
        self.payment_shares = np.asarray(phasing_CAPEX, dtype=float)
        if (not self.payment_years.size or self.payment_shares.shape != self.payment_years.shape
                or not np.isfinite(self.payment_shares).all() or np.any(self.payment_shares < 0)
                or self.payment_shares.sum() <= 0 or np.any(self.payment_years >= self.project_life_y)):
            raise ValueError("Initial investment phasing must have nonnegative shares in years 0..T-1")
        self.payment_shares = self.payment_shares / self.payment_shares.sum()

    def compute(self, CAPEX_w, CAPEX_s, CAPEX_b, CAPEX_sh, CAPEX_dc_facility,
                CAPEX_dc_it, OPEX_w, OPEX_s, OPEX_b, OPEX_sh, OPEX_dc,
                served_it_energy_mwh, datacenter_firm_revenue_per_mwh_it):
        capex = dict(zip(ASSETS, [_number(v, name) for name, v in zip(CAPEX_INPUTS, (
            CAPEX_w, CAPEX_s, CAPEX_b, CAPEX_sh, CAPEX_dc_facility, CAPEX_dc_it))]))
        opex = sum(_number(v, name) for name, v in zip(OPEX_INPUTS, (OPEX_w, OPEX_s, OPEX_b, OPEX_sh, OPEX_dc)))
        revenue = (_number(served_it_energy_mwh, "served_it_energy_mwh")
                   * _number(datacenter_firm_revenue_per_mwh_it, "datacenter_firm_revenue_per_mwh_it"))
        total_capex = sum(capex.values())
        horizon = self.project_life_y
        investment, replacements = {}, {}
        depreciation = np.zeros(horizon + 1)
        for asset, initial_cost in capex.items():
            life = self.lifetimes[asset]
            events = []
            if asset == "dc_it":
                events = [(y, initial_cost * self.it_fraction) for y in range(life, horizon, life)]
            elif asset == "b":
                y = self.battery_replacement_year
                if y < horizon:
                    events = [(y, initial_cost * self.battery_fraction)]
                if initial_cost and (min(y, horizon) > life or (y < horizon and horizon - y > life)):
                    raise ValueError("The single battery replacement does not cover the project lifetime")
            elif initial_cost and life < horizon:
                raise ValueError(f"{asset} life must cover the project; no automatic replacement is modelled")
            payments = np.zeros(horizon + 1)
            np.add.at(payments, self.payment_years,
                      initial_cost * self.payment_shares * self.inflation_index[self.payment_years])
            replacement = np.zeros(horizon + 1)
            for year, cost in events:
                replacement[year] += cost * self.inflation_index[year]
            # Initial equipment is commissioned at zero irrespective of payment phasing.
            # Its nominal depreciable basis is the total phased purchase expenditure.
            for year, basis in [(0, payments.sum())] + [(y, cost * self.inflation_index[y]) for y, cost in events]:
                depreciation[year + 1:min(year + life, horizon) + 1] += basis / life
            investment[asset] = payments + replacement
            replacements[asset] = replacement
        capex_vec = sum(investment.values())
        revenue_base = np.r_[0.0, np.full(horizon, revenue)]
        opex_base = np.r_[0.0, np.full(horizon, opex)]
        revenue_vec, opex_vec = revenue_base * self.inflation_index, opex_base * self.inflation_index
        taxes = self.tax_rate * np.maximum(revenue_vec - opex_vec - depreciation, 0)
        taxes[0] = 0
        residual = np.zeros(horizon + 1)
        residual[-1] = total_capex * self.residual_fraction * self.inflation_index[-1]
        # Native helper treats the final argument as net terminal expenditure.
        # Negative expenditure represents explicitly net-of-tax residual proceeds.
        npv, _ = calculate_NPV_IRR_phased(
            revenue_base, opex_base, capex_vec, depreciation, self.tax_rate,
            self.discount_rate, -residual, self.inflation_index)
        cashflow = revenue_vec - opex_vec - taxes - capex_vec + residual
        discounted = cashflow / (1 + self.discount_rate) ** self.years
        discount_factors = (1 + self.discount_rate) ** self.years
        pv_capex = float(np.sum(capex_vec / discount_factors))
        pv_replacement = float(np.sum(sum(replacements.values()) / discount_factors))
        irr = float(npf.irr(cashflow)) if np.any(cashflow < 0) and np.any(cashflow > 0) else np.nan
        values = dict(
            CAPEX_system=total_capex, OPEX_system=opex, Revenue_dc=revenue,
            CAPEX_lifetime_system=float(capex_vec.sum()), CAPEX_PV_system=pv_capex,
            CAPEX_replacement_PV_system=pv_replacement,
            NPV_over_PV_CAPEX_system=float(npv / pv_capex) if np.isfinite(pv_capex) and pv_capex > 0 else np.nan,
            NPV_system=float(npv), NPV_over_CAPEX_system=float(npv / total_capex) if total_capex else np.nan,
            IRR_system=irr, payback_years=_payback(cashflow), discounted_payback_years=_payback(discounted),
            finance_years=self.years.copy(), CAPEX_system_vec=capex_vec,
            replacement_battery_vec=replacements["b"], replacement_it_vec=replacements["dc_it"],
            OPEX_system_vec=opex_vec, Revenue_dc_vec=revenue_vec, depreciation_vec=depreciation,
            taxes_vec=taxes, residual_value_vec=residual, cashflow_vec=cashflow,
            discounted_cashflow_vec=discounted, cumulative_discounted_cashflow_vec=np.cumsum(discounted),
        )
        values.update({"CAPEX_" + asset + "_vec": investment[asset] for asset in ASSETS})
        return tuple(values[name] for name in OUTPUT_NAMES)


class finance_simple_datacenter_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = finance_simple_datacenter(**insta_inp)
        super().__init__(
            inputs=[(name, {"val": 0.0, **({"units": "MW*h"} if name == "served_it_energy_mwh" else {})})
                    for name in INPUT_NAMES],
            outputs=[(name, {}) for name in SCALAR_OUTPUTS]
                    + [(name, {"shape": [model.project_life_y + 1]}) for name in VECTOR_OUTPUTS],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


def reconstruct_costs(config, design, wind_ts):
    """Re-evaluate native costs and existing DC/annualization models, without EMS."""
    from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
    from hydesign.costs.cost_simple_hpp_with_hpp_cost import (
        hpp_cost_annualized, RAW_HPP_COST_NAMES, ANNUAL_HPP_COST_NAMES)
    from hydesign.costs.cost_simple_datacenter import datacenter_cost_annualized
    from hydesign.wind.wind import get_rotor_d

    def params(*keys):
        return {key: config[key] for key in keys}

    clearance, sp, rated, count, density, solar, tilt, azimuth, ratio, power, duration, it = design
    diameter = get_rotor_d(rated * 1e6 / sp)
    cw, ow = wpp_cost(N_time=len(wind_ts), **params(
        "wind_turbine_cost", "wind_civil_works_cost", "wind_fixed_onm_cost", "wind_variable_onm_cost",
        "d_ref", "hh_ref", "p_rated_ref")).compute(count, clearance + diameter / 2, diameter, rated, wind_ts)
    cs, os = pvp_cost(**params("solar_PV_cost", "solar_hardware_installation_cost",
                              "solar_inverter_cost", "solar_fixed_onm_cost")).compute(solar, ratio)
    cb, ob = battery_cost(life_y=1, battery_price_reduction_per_year=0, **params(
        "battery_energy_cost", "battery_power_cost", "battery_BOP_installation_commissioning_cost",
        "battery_control_system_cost", "battery_energy_onm_cost")).compute(power * duration, power, np.ones(len(wind_ts)))
    csh, osh, *_ = shared_cost(**params("hpp_BOS_soft_cost", "hpp_grid_connection_cost", "land_cost")).compute(
        it * config["pue"] + power, count * rated / density, solar * config["land_use_per_solar_MW"])
    raw = dict(zip(RAW_HPP_COST_NAMES, (cw, ow, cs, os, cb, ob, csh, osh)))
    annual = hpp_cost_annualized(**params("wind_WACC", "solar_WACC", "battery_WACC",
        "hpp_economic_lifetime_years", "battery_replacement_year", "battery_replacement_cost_fraction"))
    raw.update(zip(ANNUAL_HPP_COST_NAMES, annual.compute(**raw)))
    dc = datacenter_cost_annualized(**params("datacenter_facility_capex_per_mw_it", "datacenter_it_capex_per_mw_it",
        "datacenter_fixed_opex_per_mw_it_year", "datacenter_discount_rate", "datacenter_facility_lifetime_years",
        "datacenter_it_lifetime_years"))
    raw.update(zip(("CAPEX_dc_facility", "CAPEX_dc_it", "CAPEX_dc_total", "OPEX_dc",
                   "annualized_CAPEX_dc_facility", "annualized_CAPEX_dc_it", "annualized_CAPEX_dc_total",
                   "annualized_cost_dc"), dc.compute(it)))
    return {key: float(np.asarray(value).item()) for key, value in raw.items()}


def evaluate_saved_finance(result_path, config, overrides=None):
    """Validate saved dispatch, reproduce original economics, then apply YAML inputs.

    No model/solver is instantiated. The returned effective configuration includes
    overrides; the original saved files are never changed. Stored absolute archive
    paths are supported, with a portable run-directory/evaluations fallback.
    """
    from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch
    from hydesign.sizing_native_hpp_sizing_base import _output_path

    result_path = _output_path(result_path)
    saved = json.loads(result_path.read_text(encoding="utf8"))
    best, old_config = saved["best"], saved["configuration"]
    if not best:
        raise ValueError("Saved run has no validated best design")
    archive = _output_path(best["dispatch_path"])
    if not archive.exists():
        archive = _output_path(result_path.parent / "evaluations" / archive.name)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != best["dispatch_sha256"]:
        raise ValueError("Dispatch archive checksum mismatch")
    with np.load(archive, allow_pickle=False) as data:
        hourly = {k: data[k].copy() for k in data.files if k != "soc_mwh"}
        soc = data["soc_mwh"].copy()
    d = best["design"]
    if len(hourly["wind_mw"]) != 8760:
        raise ValueError("Expected a complete hourly representative year")
    np.testing.assert_allclose(hourly["facility_load_mw"], d[-1] * old_config["pue"])
    residuals = validate_dispatch(hourly, soc, best["solver_status"], d[9], d[9] * d[10],
        old_config["battery_depth_of_discharge"], old_config["battery_charge_efficiency"],
        old_config["battery_initial_soc"], old_config["pue"])
    metrics = firm_load_metrics(hourly["unserved_facility_mw"], epsilon_mw=old_config["epsilon_mw"])
    for name in ("LOLH", "EENS", "firm_reliability"):
        np.testing.assert_allclose(metrics[name], best["metrics"][name], rtol=1e-10, atol=1e-6)
    allowed = int(np.floor((1 - old_config["reliability_target"]) * 8760 + 1e-9))
    if metrics["LOLH"] > allowed:
        raise ValueError("Saved dispatch fails its reliability target")
    served = float(hourly["served_firm_it_mw"].sum())
    original = reconstruct_costs(old_config, d, hourly["wind_mw"])
    original["Revenue_dc"] = served * old_config["datacenter_firm_revenue_per_mwh_it"]
    original["net_annual_value_system"] = original["Revenue_dc"] - original["annualized_cost_hpp"] - original["annualized_cost_dc"]
    for name, value in original.items():
        if name in best["economics"]:
            np.testing.assert_allclose(value, best["economics"][name], rtol=1e-10, atol=1e-5, err_msg=name)
    effective = deepcopy(config)
    effective.update(deepcopy(overrides or {}))
    # Saved operation cannot be reinterpreted with changed PUE or battery physics.
    for key in ("pue", "battery_charge_efficiency", "battery_depth_of_discharge", "battery_initial_soc"):
        if effective[key] != old_config[key]:
            raise ValueError(f"Saved operation requires the original {key}")
    costs = reconstruct_costs(effective, d, hourly["wind_mw"])
    inputs = {name: costs[name] for name in CAPEX_INPUTS + OPEX_INPUTS}
    inputs.update(served_it_energy_mwh=served,
                  datacenter_firm_revenue_per_mwh_it=effective["datacenter_firm_revenue_per_mwh_it"])
    model = finance_simple_datacenter(**{name: effective[name] for name in FINANCE_PARAMETERS})
    result = dict(zip(OUTPUT_NAMES, model.compute(**inputs)))
    costs["Revenue_dc"] = result["Revenue_dc"]
    costs["net_annual_value_system"] = result["Revenue_dc"] - costs["annualized_cost_hpp"] - costs["annualized_cost_dc"]
    # Isolate JUST the requested facility-lifetime change from other YAML updates.
    lifetime_only = dict(old_config, datacenter_facility_lifetime_years=effective["datacenter_facility_lifetime_years"])
    revised = reconstruct_costs(lifetime_only, d, hourly["wind_mw"])
    revised["Revenue_dc"] = original["Revenue_dc"]
    revised["net_annual_value_system"] = revised["Revenue_dc"] - revised["annualized_cost_hpp"] - revised["annualized_cost_dc"]
    return dict(configuration=effective, saved_configuration=old_config, design=d, metrics=metrics,
                physical_validation=residuals, original_economics=original, lifetime_only_economics=revised,
                current_economics=costs, inputs=inputs, finance=result,
                served_it_energy_mwh=served, saved_result=str(result_path))


def validate_saved_assembly_finance(case, config_path):
    """Replay a validated archive through the actual joint OpenMDAO assembly.

    Only the EMS solve is replaced by its saved dispatch, so differences in
    financial wiring cannot be hidden by a different optimal dispatch. Native
    generation, costs, finance, physical checks and outer scoring all execute.
    This is a validation helper, not a sizing or production evaluation path.
    """
    from datetime import datetime, timezone
    from uuid import uuid4
    from unittest.mock import patch
    from hydesign.sizing_native_hpp_sizing_base import ROOT, physical_inputs, _output_path
    from hydesign.assembly.hpp_assembly_simple_datacenter_joint_hpp_dc_sizing import (
        hpp_model_simple_datacenter_joint_hpp_dc_sizing, VARIABLE_NAMES, write_json_atomic)
    from hydesign.ems import ems_simple_datacenter_comp_max_reliability as ems_interface

    result_path = _output_path(case['saved_result'])
    best = json.loads(result_path.read_text(encoding='utf8'))['best']
    path = _output_path(best['dispatch_path'])
    if not path.exists():
        path = result_path.parent / 'evaluations' / path.name
    with np.load(path, allow_pickle=False) as archive:
        hourly = {k: archive[k].copy() for k in archive.files if k != 'soc_mwh'}
        soc = archive['soc_mwh'].copy()
    cfg = deepcopy(case['configuration'])
    cfg['reliability_target'] = case['saved_configuration']['reliability_target']
    design = case['design']
    variables = {name: dict(var_type='fixed', types='int' if name == 'Nwt' else 'float', value=value)
                 for name, value in zip(VARIABLE_NAMES, design)}

    def replay(wind, solar, *args, **kwargs):
        np.testing.assert_allclose(wind, hourly['wind_mw'], atol=1e-8, rtol=1e-10)
        np.testing.assert_allclose(solar, hourly['pv_mw'], atol=1e-8, rtol=1e-10)
        return (hourly['served_facility_mw'], hourly['curtailment_mw'],
                hourly['discharge_mw'] - hourly['charge_mw'], soc,
                hourly['charge_mw'], hourly['discharge_mw'], hourly['served_firm_it_mw'],
                hourly['unserved_firm_it_mw'], hourly['unserved_facility_mw'], best['solver_status'])

    directory = _output_path(ROOT/'hpp_sizing_runs/joint_hpp_dc_sizing'/(
        'finance_validation_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_') + uuid4().hex[:8]))
    directory.mkdir(parents=True)
    model = hpp_model_simple_datacenter_joint_hpp_dc_sizing(
        **physical_inputs(config_path, cfg, directory), variables=variables)
    try:
        with patch.object(ems_interface, 'ems_cplex_simple_datacenter_max_reliability', replay):
            outputs = model.evaluate(*design)
        if model.candidate_status != 'FEASIBLE':
            raise AssertionError(model.candidate_error or model.candidate_status)
        for name in hourly:
            np.testing.assert_array_equal(model.hourly[name], hourly[name])
        np.testing.assert_array_equal(model.soc_mwh, soc)
        for name in OUTPUT_NAMES:
            np.testing.assert_allclose(model.prob.get_val(name), case['finance'][name],
                                       rtol=1e-10, atol=1e-5)
        for name in ('annualized_cost_hpp', 'annualized_cost_dc', 'net_annual_value_system'):
            np.testing.assert_allclose(model.economics[name], case['current_economics'][name],
                                       rtol=1e-10, atol=1e-5)
        result = dict(economics=model.record['economics'], metrics=model.metrics,
                      validation=model.final_checks, dispatch_unchanged=True,
                      configuration=cfg, validation_directory=str(directory),
                      outputs={k:float(v) if np.isfinite(v) else None
                               for k,v in zip(model.list_out_vars, outputs)},
                      scoring=model.scoring)
        write_json_atomic(directory/'validation.json', result)
        return result
    finally:
        model.prob.cleanup()


def compare_native_hpp_finance(inputs, config):
    """HPP-only parity fixture against the complete native finance component.

    Use the same HPP costs and annual revenue, excluding DC costs and scheduled
    replacements in BOTH models. Native base finance does not provide the thesis
    replacement schedule. This deliberately restricted fixture uses zero tax,
    inflation and residual value, with initial payment at zero. The artificial
    annual revenue is a numerical comparison input, not an internal PPA.
    This validation-only call includes native break-even-PPA reporting; the
    integrated reporting component itself only calls the cash-flow helper.
    Replacement cash flows are tested separately against manual discounting.
    """
    from hydesign.finance.finance import finance

    if (config["tax_rate"] != 0 or np.any(np.asarray(config["inflation"]) != 0)
            or config["residual_value_fraction"] != 0 or config["phasing_yr"] != [0]):
        raise ValueError("Native base parity fixture requires zero tax/inflation/residual and payment at zero")
    horizon = config["hpp_economic_lifetime_years"]
    assumptions = {name: config[name] for name in FINANCE_PARAMETERS}
    assumptions.update(battery_replacement_year=horizon, battery_life_y=horizon)
    hpp_inputs = dict(inputs, CAPEX_dc_facility=0, CAPEX_dc_it=0, OPEX_dc=0)
    integrated = dict(zip(OUTPUT_NAMES, finance_simple_datacenter(**assumptions).compute(**hpp_inputs)))
    native = finance(
        N_time=8760, life_y=horizon, project_life_y=horizon,
        depreciation_yr=[0, horizon], depreciation=[0, 1],
        inflation_yr=config["inflation_yr"], inflation=config["inflation"],
        ref_yr_inflation=config["ref_yr_inflation"],
        phasing_yr=config["phasing_yr"], phasing_CAPEX=config["phasing_CAPEX"])
    native_outputs = native.compute(
        hpp_t_with_deg=np.full(horizon * 8760, inputs["served_it_energy_mwh"] / 8760),
        price_t_ext=np.full(horizon * 8760, inputs["datacenter_firm_revenue_per_mwh_it"]),
        penalty_t=np.zeros(horizon * 8760),
        **{name: inputs[name] for name in CAPEX_INPUTS[:4] + OPEX_INPUTS[:4]},
        wind_WACC=config["project_discount_rate"], solar_WACC=config["project_discount_rate"],
        battery_WACC=config["project_discount_rate"], tax_rate=config["tax_rate"],
        decommissioning_cost_tot_w=0, decommissioning_cost_tot_s=0)
    native_npv = native_outputs[3]
    np.testing.assert_allclose(integrated["NPV_system"], native_npv, rtol=1e-10, atol=1e-6)
    return dict(integrated_hpp_only_npv=integrated["NPV_system"], native_hpp_only_npv=float(native_npv),
                difference=float(integrated["NPV_system"] - native_npv))
