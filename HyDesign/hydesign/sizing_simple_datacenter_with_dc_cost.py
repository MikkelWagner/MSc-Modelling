"""Joint constant-load and dispatch optimization with DC economics only.

The shared technical EMS constraint builder, physical upper bound and result
validation are reused. Annual costs use the
standalone cost model; revenue is based on served IT energy, never facility MWh.
"""

import warnings

import numpy as np
from docplex.mp.model import Model

from hydesign.costs.cost_simple_datacenter import datacenter_cost_annualized

from hydesign.ems.ems_simple_datacenter import (
    ems_cplex_constraints_simple_datacenter,
    ems_cplex_results_simple_datacenter,
    battery_power_fluctuation_expression,
)
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch


def size_datacenter_with_dc_cost(
    wind_ts,
    solar_ts,
    b_P,
    b_E,
    battery_depth_of_discharge=0.9,
    battery_charge_efficiency=np.sqrt(0.86),
    battery_initial_soc=0.5,
    pue=1.15,
    reliability_target=0.95,
    facility_load_min_mw=1.0,
    facility_load_max_mw=50.0,
    epsilon_mw=1e-3,
    time_limit_seconds=600,
    threads=4,
    log_output=False,
    mip_relative_gap=1e-4,
    allow_time_limit_incumbent=False,
    *,
    datacenter_facility_capex_per_mw_it,
    datacenter_it_capex_per_mw_it,
    datacenter_fixed_opex_per_mw_it_year,
    datacenter_discount_rate,
    datacenter_facility_lifetime_years,
    datacenter_it_lifetime_years,
    datacenter_firm_revenue_per_mwh_it,
    battery_power_fluctuation_penalty=0.0,
):
    """Maximize served IT revenue minus annualized installed DC cost.

    All economic inputs are required and supplied by the caller from YAML.
    One-hour steps are assumed. Production runs use 8760 hours; short arrays
    are supported for controlled tests without rescaling the annual cost.
    No PPA payments or HPP economics enter this objective.
    The optional fluctuation penalty uses the same MWh_IT/MW coefficient as
    fixed dispatch, multiplied by the IT revenue rate to express it in EUR.
    Reported net annual value excludes this artificial operating preference.
    """
    if not np.isfinite(battery_power_fluctuation_penalty) or battery_power_fluctuation_penalty < 0:
        raise ValueError("battery_power_fluctuation_penalty must be finite and nonnegative")
    if (datacenter_firm_revenue_per_mwh_it is None
            or not np.isfinite(datacenter_firm_revenue_per_mwh_it)
            or datacenter_firm_revenue_per_mwh_it < 0):
        raise ValueError("datacenter_firm_revenue_per_mwh_it must be finite and nonnegative")
    cost_model = datacenter_cost_annualized(
        datacenter_facility_capex_per_mw_it,
        datacenter_it_capex_per_mw_it,
        datacenter_fixed_opex_per_mw_it_year,
        datacenter_discount_rate,
        datacenter_facility_lifetime_years,
        datacenter_it_lifetime_years,
    )
    if (not np.isfinite(facility_load_min_mw) or not np.isfinite(facility_load_max_mw)
            or not 0 <= facility_load_min_mw < facility_load_max_mw):
        raise ValueError("Specify finite load bounds with 0 <= minimum < maximum")
    if not np.isfinite(mip_relative_gap) or not 0 <= mip_relative_gap < 1:
        raise ValueError("MIP relative gap must be in [0, 1)")

    # A successful hour cannot exceed renewable power plus the maximum possible
    # battery discharge, even if the battery starts that hour full. The relevant
    # order statistic is therefore a necessary upper bound, not a new assumption.
    renewable = np.asarray(wind_ts, dtype=float) + np.asarray(solar_ts, dtype=float)
    physical_upper_mw = facility_load_max_mw
    if renewable.ndim == 1 and renewable.size and 0 < reliability_target <= 1:
        allowed = int(np.floor((1 - reliability_target) * len(renewable) + 1e-9))
        discharge_limit = min(b_P, b_E * battery_depth_of_discharge) * battery_charge_efficiency
        physical_upper_mw = min(
            facility_load_max_mw, float(np.sort(renewable + discharge_limit)[allowed])
        )
    if physical_upper_mw < facility_load_min_mw:
        raise ValueError("No load within the requested bounds can satisfy hourly reliability")

    with Model(name="simple_datacenter_load_sizing_with_dc_cost") as mdl:
        # The HPP ratings remain fixed inputs; this is the only sizing variable.
        facility_load = mdl.continuous_var(
            lb=facility_load_min_mw, ub=physical_upper_mw,
            name="contracted_facility_load_mw",
        )
        variables = ems_cplex_constraints_simple_datacenter(
            mdl, wind_ts, solar_ts, b_P, b_E,
            battery_depth_of_discharge, battery_charge_efficiency,
            battery_initial_soc, facility_load, physical_upper_mw,
            pue, reliability_target,
        )
        economics = datacenter_economic_expressions(
            mdl, variables, facility_load, pue, cost_model,
            datacenter_firm_revenue_per_mwh_it,
        )
        sol, solver_status = solve_cplex_simple_datacenter_with_dc_cost(
            mdl, variables, economics, facility_load, time_limit_seconds, threads,
            log_output, mip_relative_gap, allow_time_limit_incumbent,
            battery_power_fluctuation_penalty=battery_power_fluctuation_penalty,
            datacenter_firm_revenue_per_mwh_it=datacenter_firm_revenue_per_mwh_it,
        )
        economic_values = {name: float(sol.get_value(value)) for name, value in economics.items()}
        economic_values["datacenter_firm_revenue_per_mwh_it"] = datacenter_firm_revenue_per_mwh_it
        load_mw = float(sol.get_value(facility_load))
        (hpp_t, hpp_curt_t, b_t, soc_mwh, charge_mw, discharge_mw,
         served_it, unserved_it, unmet, _) = ems_cplex_results_simple_datacenter(
            sol, variables, solver_status, pue,
        )
        permitted_failure = np.asarray(sol.get_values(variables["failed_hour"]))

    wind = np.asarray(wind_ts, dtype=float)
    pv = np.asarray(solar_ts, dtype=float)
    hourly = {
        "wind_mw": wind.copy(), "pv_mw": pv.copy(),
        "renewable_available_mw": wind + pv,
        "facility_load_mw": np.full(len(wind), load_mw),
        "firm_demand_it_mw": np.full(len(wind), load_mw / pue),
        "served_facility_mw": hpp_t, "unserved_facility_mw": unmet,
        "served_firm_it_mw": served_it, "unserved_firm_it_mw": unserved_it,
        "charge_mw": charge_mw, "discharge_mw": discharge_mw,
        "curtailment_mw": hpp_curt_t,
    }
    validation = validate_dispatch(
        hourly, soc_mwh, solver_status, b_P, b_E,
        battery_depth_of_discharge, battery_charge_efficiency,
        battery_initial_soc, pue,
    )
    metrics = firm_load_metrics(unmet, epsilon_mw=epsilon_mw)
    if metrics["LOLH"] > solver_status["max_failed_hours"]:
        raise RuntimeError("Post-processed failed-hour count violates the reliability constraint")
    if metrics["firm_reliability"] < reliability_target - 1e-12:
        raise RuntimeError("Post-processed reliability is below its target")

    metrics.update({
        "delivered_energy_mwh": float(hpp_t.sum()),
        "curtailment_mwh": float(hpp_curt_t.sum()),
        "battery_throughput_mwh": float((charge_mw + discharge_mw).sum()),
        "minimum_supplied_mw": float(hpp_t.min()),
        "maximum_shortfall_mw": float(unmet.max()),
    })
    economic_values["facility_load_mw"] = load_mw
    economic_values["served_facility_energy_mwh"] = float(hpp_t.sum())
    return {
        "facility_load_mw": load_mw,
        "it_capacity_mw": load_mw / pue,
        "economics": economic_values,
        "firm_demand_it_mw": load_mw / pue,
        "hourly": hourly,
        "soc_mwh": soc_mwh,
        "b_t": b_t,
        "permitted_failure": permitted_failure,
        "metrics": metrics,
        "solver_status": solver_status,
        "validation": validation,
        "upper_bound_active": load_mw >= facility_load_max_mw - 1e-4,
        "lower_bound_active": load_mw <= facility_load_min_mw + 1e-4,
        "facility_load_bounds_mw": (facility_load_min_mw, facility_load_max_mw),
        "physical_load_upper_mw": physical_upper_mw,
    }


def datacenter_economic_expressions(
    mdl, variables, facility_load, pue, cost_model, datacenter_firm_revenue_per_mwh_it,
):
    """Build linear economics using the standalone model evaluated at 1 MW_IT.

    Costs are linear in installed IT capacity, so multiplying those unit costs
    reuses the exact base equations and both CRFs without duplicating them here.
    For dt=1 h, EUR/MWh_IT * sum(served IT MW) gives annual EUR for 8760 hours.
    Raw CAPEX is reported in EUR; only annualized cost enters the net objective.
    """
    names = (
        "CAPEX_dc_facility", "CAPEX_dc_it", "CAPEX_dc_total", "OPEX_dc",
        "annualized_CAPEX_dc_facility", "annualized_CAPEX_dc_it",
        "annualized_CAPEX_dc_total", "annualized_cost_dc",
    )
    it_capacity_mw = facility_load / pue
    economics = {
        name: float(unit_cost) * it_capacity_mw
        for name, unit_cost in zip(names, cost_model.compute(1.0))
    }
    served_it_energy_mwh = mdl.sum(variables["P_served_IT_t"])
    Revenue_dc = datacenter_firm_revenue_per_mwh_it * served_it_energy_mwh
    economics.update({
        "it_capacity_mw": it_capacity_mw,
        "served_it_energy_mwh": served_it_energy_mwh,
        "Revenue_dc": Revenue_dc,
        "net_annual_value": Revenue_dc - economics["annualized_cost_dc"],
    })
    return economics


def solve_cplex_simple_datacenter_with_dc_cost(
    mdl, variables, economics, load_variable, time_limit_seconds=600, threads=4,
    log_output=False, mip_relative_gap=1e-4, allow_time_limit_incumbent=False,
    battery_power_fluctuation_penalty=0.0,
    datacenter_firm_revenue_per_mwh_it=None,
):
    """One solve: DC net annual value minus an optional fluctuation penalty.

    lambda [MWh_IT/MW] times IT revenue [EUR/MWh_IT] gives EUR per MW of
    adjacent net-battery-power change. Costs/revenue remain unpenalized in
    economic reporting; solver objective, bound and gap include the penalty.
    load_variable is retained for call compatibility, without fixing its value.
    """
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("Time limit must be positive and finite")
    fluctuation = battery_power_fluctuation_expression(
        mdl, variables, battery_power_fluctuation_penalty,
    )
    if datacenter_firm_revenue_per_mwh_it is None:
        if battery_power_fluctuation_penalty > 0:
            raise ValueError("Supply the IT revenue rate to scale the fluctuation penalty to EUR")
        datacenter_firm_revenue_per_mwh_it = 0.0
    if not np.isfinite(datacenter_firm_revenue_per_mwh_it) or datacenter_firm_revenue_per_mwh_it < 0:
        raise ValueError("IT revenue rate must be finite and nonnegative")
    penalty_eur_per_mw = battery_power_fluctuation_penalty * datacenter_firm_revenue_per_mwh_it
    mdl.parameters.threads = threads
    mdl.parameters.timelimit = time_limit_seconds
    mdl.parameters.simplex.tolerances.feasibility = 1e-9
    mdl.parameters.simplex.tolerances.optimality = 1e-9
    mdl.parameters.mip.tolerances.integrality = 1e-9
    mdl.parameters.mip.tolerances.mipgap = mip_relative_gap
    net_annual_value = economics["net_annual_value"]
    scale = 1.0
    mdl.maximize((net_annual_value - penalty_eur_per_mw * fluctuation) / scale)
    stage = "maximum_net_annual_value"
    try:
        sol = mdl.solve(log_output=log_output)
        details = mdl.solve_details
        info = {
            "stage": stage, "status": getattr(details, "status", "no solve details"),
            "status_code": getattr(details, "status_code", None),
            "seconds": getattr(details, "time", None),
        }
        optimal = info["status_code"] in (1, 101, 102)
        accepted_incumbent = (
            allow_time_limit_incumbent and info["status_code"] == 107 and sol is not None
        )
        if sol is None or not (optimal or accepted_incumbent):
            raise RuntimeError(f"Did not reach optimality: {info}")
    except Exception as exc:
        raise RuntimeError(f"CPLEX {stage} solver error: {exc}") from exc
    net = np.asarray(sol.get_values(variables["P_discharge"])) - np.asarray(
        sol.get_values(variables["P_charge"]))
    actual_fluctuation = float(np.abs(np.diff(net)).sum())
    penalty_eur = penalty_eur_per_mw * actual_fluctuation
    info.update({
        "solver_objective": float(sol.objective_value),
        "objective": float(sol.objective_value) * scale, "objective_units": "EUR",
        "unpenalized_objective": float(sol.get_value(net_annual_value)),
        "Revenue_dc": float(sol.get_value(economics["Revenue_dc"])),
        "annualized_cost_dc": float(sol.get_value(economics["annualized_cost_dc"])),
        "net_annual_value": float(sol.get_value(net_annual_value)),
        "battery_power_fluctuation_mw": actual_fluctuation,
        "battery_penalty_eur": penalty_eur,
    })
    if mdl.number_of_binary_variables:
        info["relative_gap"] = float(details.mip_relative_gap)
        info["best_bound"] = float(details.best_bound) * scale
    solver_status = {
        "backend": "cplex", "optimal": optimal, "feasible": True, "stages": [info],
        "primary_net_annual_value": info["net_annual_value"],
        "battery_power_fluctuation_penalty": battery_power_fluctuation_penalty,
        "battery_power_fluctuation_mw": actual_fluctuation,
        "battery_penalty_eur": penalty_eur,
        "mip_relative_gap_tolerance": mip_relative_gap,
        "max_failed_hours": variables["max_failed_hours"],
    }
    if not optimal:
        warnings.warn(
            "CPLEX reached the time limit: returning a feasible candidate, not a "
            f"proven optimum. Penalized net-value gap: {info['relative_gap']:.2%}.",
            RuntimeWarning,
        )
    return sol, solver_status
