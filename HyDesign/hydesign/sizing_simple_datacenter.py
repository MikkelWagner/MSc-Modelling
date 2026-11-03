"""Size contracted facility demand with a fixed HPP and hourly reliability.

Only this module creates the load-sizing variable. The EMS module supplies the
same chronological operating constraints used by its fixed-load component.
No plant sizing, data-center CAPEX, degradation or lifetime finance is included.
"""

import numpy as np
from docplex.mp.model import Model

from hydesign.ems.ems_simple_datacenter import (
    ems_cplex_constraints_simple_datacenter,
    ems_cplex_results_simple_datacenter,
    solve_cplex_simple_datacenter,
)
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch


def size_datacenter(
    wind_ts,
    solar_ts,
    b_P,
    b_E,
    battery_depth_of_discharge=0.9,
    battery_charge_efficiency=np.sqrt(0.86),
    battery_initial_soc=0.5,
    pue=1.15,
    reliability_target=0.95,
    ppa_price=70.0,
    facility_load_min_mw=1.0,
    facility_load_max_mw=50.0,
    epsilon_mw=1e-3,
    time_limit_seconds=600,
    threads=4,
    log_output=False,
    mip_relative_gap=1e-4,
    allow_time_limit_incumbent=False,
    battery_power_fluctuation_penalty=0.0,
):
    """Maximize served IT energy minus optional battery-power fluctuations.

    The load is one continuous MW value for the entire year. Partial supply
    contributes served energy but counts as a failed hour. CPLEX jointly selects load and
    hourly operation; there is no nested heuristic search or EENS objective.
    Epsilon is used only to classify numerical shortfalls after optimization.
    PPA price is used only for optional revenue reporting.
    battery_power_fluctuation_penalty is MWh_IT per MW of adjacent AC net-power
    change, zero by default; it is not a throughput or degradation cost.
    """
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

    with Model(name="simple_datacenter_load_sizing") as mdl:
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
        sol, solver_status = solve_cplex_simple_datacenter(
            mdl, variables, ppa_price, time_limit_seconds, threads, log_output,
            load_variable=facility_load, mip_relative_gap=mip_relative_gap,
            allow_time_limit_incumbent=allow_time_limit_incumbent,
            battery_power_fluctuation_penalty=battery_power_fluctuation_penalty,
        )
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
        "served_it_energy_mwh": float(served_it.sum()),
        "battery_power_fluctuation_mw": float(np.abs(np.diff(b_t)).sum()),
        "battery_throughput_mwh": float((charge_mw + discharge_mw).sum()),
        "curtailment_mwh": float(hpp_curt_t.sum()),
        "ppa_revenue_eur": float(ppa_price * hpp_t.sum()),
        "minimum_supplied_mw": float(hpp_t.min()),
        "maximum_shortfall_mw": float(unmet.max()),
    })
    return {
        "facility_load_mw": load_mw,
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
