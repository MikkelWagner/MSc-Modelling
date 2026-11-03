"""Experimental exact strengthening of the cost-enabled DC sizing model.

Copied from sizing_simple_datacenter_with_dc_cost; the original is untouched.
The shared EMS, economics and dispatch validation keep their original structure.
Independent switches support controlled benchmarks. No MIP start is supplied.
An optional incumbent value MUST belong to the same inputs/objective: it is an
objective-pruning certificate, not a physical minimum or a starting solution.
"""

import time
import warnings

from docplex.mp.relax_linear import LinearRelaxer

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
    tighten_load_upper_bound=True,
    aggregate_shortage=True,
    hourly_strengthening=True,
    incumbent_objective_eur=None,
    primary_objective_scale=1.0,
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
    # incumbent_objective_eur must certify NET VALUE MINUS FLUCTUATION PENALTY
    # for this exact case/weight. An old unpenalized certificate is not valid.
    construction_start = time.perf_counter()
    if not np.isfinite(primary_objective_scale) or primary_objective_scale <= 0:
        raise ValueError("primary_objective_scale must be positive and finite")
    if incumbent_objective_eur is not None and not np.isfinite(incumbent_objective_eur):
        raise ValueError("incumbent_objective_eur must be finite, or None")
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

    original_physical_upper_mw = physical_upper_mw
    effective_lower_mw = facility_load_min_mw
    bound_details = {}
    if tighten_load_upper_bound:
        physical_upper_mw, bound_details = presolve_load_upper_bound(
            wind_ts, solar_ts, b_P, b_E, battery_depth_of_discharge,
            battery_charge_efficiency, battery_initial_soc, pue, reliability_target,
            facility_load_min_mw, physical_upper_mw, cost_model,
            datacenter_firm_revenue_per_mwh_it, time_limit_seconds, threads,
        )
    if incumbent_objective_eur is not None:
        effective_lower_mw = improving_load_lower_bound(
            incumbent_objective_eur, len(renewable), datacenter_firm_revenue_per_mwh_it,
            float(cost_model.compute(1.0)[-1]), pue, facility_load_min_mw,
        )
    if effective_lower_mw > physical_upper_mw:
        raise ValueError("Certified bounds conflict; check the incumbent belongs to this exact case")

    with Model(name="simple_datacenter_load_sizing_with_dc_cost_strengthened") as mdl:
        # The HPP ratings remain fixed inputs; this is the only sizing variable.
        facility_load = mdl.continuous_var(
            lb=effective_lower_mw, ub=physical_upper_mw,
            name="contracted_facility_load_mw",
        )
        variables = ems_cplex_constraints_simple_datacenter(
            mdl, wind_ts, solar_ts, b_P, b_E,
            battery_depth_of_discharge, battery_charge_efficiency,
            battery_initial_soc, facility_load, physical_upper_mw,
            pue, reliability_target,
        )
        add_reliability_strengthening(
            mdl, variables, facility_load, pue, effective_lower_mw,
            aggregate_shortage, hourly_strengthening,
        )
        economics = datacenter_economic_expressions(
            mdl, variables, facility_load, pue, cost_model,
            datacenter_firm_revenue_per_mwh_it,
        )
        formulation = {
            "original_big_m_mw": original_physical_upper_mw,
            "big_m_mw": physical_upper_mw,
            "physical_minimum_mw": facility_load_min_mw,
            "effective_lower_mw": effective_lower_mw,
            "incumbent_objective_eur": incumbent_objective_eur,
            "primary_objective_scale": primary_objective_scale,
            "tighten_load_upper_bound": tighten_load_upper_bound,
            "aggregate_shortage": aggregate_shortage,
            "hourly_strengthening": hourly_strengthening,
            "continuous_variables": mdl.number_of_continuous_variables,
            "binary_variables": mdl.number_of_binary_variables,
            "constraints": mdl.number_of_constraints,
            "fixed_failure_binaries": sum(v.ub == 0 for v in variables["failed_hour"]),
            "construction_and_bound_seconds": time.perf_counter() - construction_start,
            "bound_details": bound_details,
        }
        sol, solver_status = solve_cplex_simple_datacenter_with_dc_cost(
            mdl, variables, economics, facility_load, time_limit_seconds, threads,
            log_output, mip_relative_gap, allow_time_limit_incumbent,
            battery_power_fluctuation_penalty=battery_power_fluctuation_penalty,
            datacenter_firm_revenue_per_mwh_it=datacenter_firm_revenue_per_mwh_it,
            primary_objective_scale=primary_objective_scale,
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
        "formulation": formulation,
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


def improving_load_lower_bound(incumbent_eur, hours, rate, annual_cost, pue, physical_minimum):
    """Prune only loads unable to match a caller-certified feasible objective.

    For nonnegative penalties, penalized V <= unpenalized V <= A*L, A=(rate*T-annual_cost)/PUE, because served energy <= T*L/PUE.
    For A>0 and V>=incumbent, L>=incumbent/A. One EUR and 1e-6 MW slack
    conservatively cover rounding in a manually supplied certificate. For A<=0
    this lower-bound argument does not apply, so retain the physical minimum.
    This is NOT a MIP start and does not impose an objective cutoff.
    """
    coefficient = (rate * hours - annual_cost) / pue
    if coefficient <= 0:
        return physical_minimum
    return max(physical_minimum, (incumbent_eur - 1.0) / coefficient - 1e-6)


def add_reliability_strengthening(mdl, variables, load, pue, lower,
                                  aggregate_shortage=True, hourly_strengthening=True):
    """Additional valid rows; retain all original hourly failure constraints.

    At most K hours fail and each shortfall <= L, hence sum(u) <= K*L.
    This is a consequence of the hourly reliability model, not its replacement.
    For binary z, u <= L-lower*(1-z) is redundant: z=0 already forces u=0
    and L>=lower; z=1 gives the existing demand bound u<=L.
    """
    failures = variables["failed_hour"]
    if not failures:  # R=0 uses the original unconstrained continuous dispatch.
        return
    if aggregate_shortage:
        mdl.add_constraint(
            pue * mdl.sum(variables["P_unserved_IT_t"])
            <= variables["max_failed_hours"] * load,
            ctname="aggregate_shortage_valid_inequality",
        )
    if hourly_strengthening:
        for t, (unserved, failure) in enumerate(zip(variables["P_unserved_IT_t"], failures)):
            mdl.add_constraint(pue * unserved <= load - lower * (1 - failure),
                               ctname=f"hourly_shortage_strengthening_{t}")


def presolve_load_upper_bound(wind, solar, b_p, b_e, dod, eta, initial_soc, pue,
                              reliability, lower, upper, cost_model, rate,
                              time_limit_seconds, threads):
    """Fresh baseline LP certificates; no prior incumbent/bound or MIP start.

    First maximize UNPENALIZED net value in the baseline LP, obtaining V_LP.
    Keep this bound objective unpenalized even when the dispatch penalty is on:
    the certificate below bounds the feasible set, not the penalized optimum.
    Every integer-feasible point has V >= B*L, B=(rate*(T-K)-cost)/PUE.
    If B>0, L <= V_LP/B. Also maximize L in the same relaxation; its optimum
    bounds every integer-feasible load regardless of economic coefficient signs.
    Use only optimal LP solves, with conservative outward EUR/MW slack. If an
    LP is not solved to optimality, omit that certificate (keep the old bound).
    Rebuilding the EMS with the reduced upper bound reruns its existing
    high-generation dominance preprocessing; no new hourly heuristic is used.
    """
    start = time.perf_counter()
    details = {}
    candidates = [upper]
    with Model(name="datacenter_bound_source") as source:
        load = source.continuous_var(lb=lower, ub=upper, name="contracted_facility_load_mw")
        variables = ems_cplex_constraints_simple_datacenter(
            source, wind, solar, b_p, b_e, dod, eta, initial_soc,
            load, upper, pue, reliability,
        )
        economics = datacenter_economic_expressions(source, variables, load, pue, cost_model, rate)
        source.maximize(economics["net_annual_value"])
        with LinearRelaxer.make_relaxed_model(source) as relaxed:
            relaxed.parameters.threads = threads
            relaxed.parameters.timelimit = time_limit_seconds
            relaxed.parameters.simplex.tolerances.feasibility = 1e-9
            relaxed.parameters.simplex.tolerances.optimality = 1e-9
            sol = relaxed.solve()
            details["economic_lp_status"] = relaxed.solve_details.status
            k = variables["max_failed_hours"]
            coefficient = (rate * (len(wind) - k) - float(cost_model.compute(1.0)[-1])) / pue
            details["minimum_value_per_mw"] = coefficient
            if sol is not None and relaxed.solve_details.status_code == 1:
                objective = float(sol.objective_value)
                details["economic_lp_bound_eur"] = objective
                if coefficient > 0:
                    bound = (objective + max(1.0, abs(objective) * 1e-9)) / coefficient + 1e-6
                    details["economic_load_upper_mw"] = bound
                    candidates.append(bound)
            relaxed.maximize(relaxed.get_var_by_name(load.name))
            sol = relaxed.solve()
            details["maximum_load_lp_status"] = relaxed.solve_details.status
            if sol is not None and relaxed.solve_details.status_code == 1:
                bound = float(sol.objective_value) + max(1e-6, abs(sol.objective_value) * 1e-8)
                details["maximum_load_lp_upper_mw"] = bound
                candidates.append(bound)
    details["seconds"] = time.perf_counter() - start
    return min(candidates), details


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
    primary_objective_scale=1.0,
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
    scale = primary_objective_scale
    # Scale the complete objective and preserve the absolute gap in EUR.
    mdl.parameters.mip.tolerances.absmipgap = mdl.parameters.mip.tolerances.absmipgap.get() / scale
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
