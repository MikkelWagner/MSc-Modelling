"""Single-solve target-slack dispatch, reusing the existing battery physics.

Hourly facility-bus powers; battery rating is storage-side MW. A continuous
slack has an integer minimum because H_max and the failure count are integers.
The complete secondary term lies in [0, beta_E * (1 + kappa_B)] < 1.
This gives strict priority to the minimum excess count at a true optimum;
a time-limited incumbent is not assumed to attain that minimum.
"""
import numpy as np
from docplex.mp.model import Model

from hydesign.ems.ems_simple_datacenter_max_reliability import (
    ems_cplex_constraints_simple_datacenter_max_reliability,
)
from hydesign.ems.ems_simple_datacenter import ems_cplex_results_simple_datacenter
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch


def calibrate_penalties(n_hours, facility_load_mw, battery_power_mw, efficiency,
                        beta_energy=.25, kappa_battery=.001):
    """Bound energy deficit and optional throughput together, in slack hours.

    kappa_B is the throughput budget relative to beta_E; zero disables it.
    Uniform scaling makes the served-energy coefficient -1. It does not
    change the objective trade-off. Regularization is not an exclusivity rule.
    """
    values = [n_hours, facility_load_mw, battery_power_mw, efficiency,
              beta_energy, kappa_battery]
    if (not np.isfinite(values).all() or n_hours <= 0 or int(n_hours) != n_hours
            or facility_load_mw <= 0 or battery_power_mw < 0
            or not 0 < efficiency <= 1 or not 0 < beta_energy < 1
            or kappa_battery < 0 or beta_energy * (1 + kappa_battery) >= 1):
        raise ValueError('Require positive load, valid battery and secondary bound < 1')
    energy_max = float(facility_load_mw * n_hours)
    throughput_bound = float(n_hours * battery_power_mw * (1/efficiency + efficiency))
    battery_budget = beta_energy * kappa_battery if throughput_bound else 0.
    return dict(beta_energy=float(beta_energy), kappa_battery=float(kappa_battery),
                energy_max_mwh=energy_max, throughput_bound_mwh=throughput_bound,
                lambda_b=battery_budget/throughput_bound if throughput_bound else 0.,
                secondary_bound=float(beta_energy + battery_budget),
                objective_scale=energy_max/beta_energy)


def slack_certificate(best_bound, secondary_bound, observed_excess, tolerance=1e-5):
    """Bound minimum EXCESS exact-service failures, not minimum annual LOLH.

    LB <= s* + secondary <= s* + secondary_bound. The optimum s* is integer.
    Zero observed excess certifies the minimum zero without a solver bound.
    A positive lower bound does not exclude an epsilon-based feasible dispatch.
    """
    lower = max(0, int(np.ceil(best_bound-secondary_bound-tolerance))) if np.isfinite(best_bound) else 0
    if lower > observed_excess:
        raise ValueError('Inconsistent minimum-slack bound and validated incumbent')
    return dict(minimum_slack_lower_bound=lower,
                minimum_slack_upper_bound=int(observed_excess),
                minimum_reliability_violation_certified=bool(lower == observed_excess),
                exact_service_target_excluded=bool(lower > 0),
                certificate_scope='excess exact-service binary failures; not epsilon-based LOLH')


def ems_cplex_simple_datacenter_target_slack_energy(
    wind_ts, solar_ts, P_batt_MW, E_batt_MWh,
    battery_depth_of_discharge, charge_efficiency, battery_initial_soc,
    facility_load_mw, pue, reliability_target,
    time_limit_seconds=30, threads=1, log_output=False,
    beta_energy=.25, kappa_battery=.001, epsilon_mw=.001,
    mip_relative_gap=1e-4, mip_absolute_gap=1e-6, random_seed=0,
):
    """One CPLEX MILP; return nine original arrays plus validated diagnostics."""
    if not np.isfinite(reliability_target) or not 0 <= reliability_target <= 1:
        raise ValueError('Reliability target must be in [0, 1]')
    if (not np.isfinite([time_limit_seconds, mip_relative_gap, mip_absolute_gap]).all()
            or time_limit_seconds <= 0 or min(mip_relative_gap, mip_absolute_gap) < 0):
        raise ValueError('Invalid CPLEX time limit or gap tolerances')
    calibration = calibrate_penalties(len(wind_ts), facility_load_mw, P_batt_MW,
                                     charge_efficiency, beta_energy, kappa_battery)
    n = len(wind_ts)
    allowed = int(np.floor((1-reliability_target)*n + 1e-9))
    with Model(name='EMS_simple_datacenter_target_slack_energy') as mdl:
        variables = ems_cplex_constraints_simple_datacenter_max_reliability(
            mdl, wind_ts, solar_ts, P_batt_MW, E_batt_MWh,
            battery_depth_of_discharge, charge_efficiency, battery_initial_soc,
            facility_load_mw, pue)
        slack = mdl.continuous_var(lb=0, ub=n-allowed, name='reliability_slack')
        mdl.add_constraint(mdl.sum(variables['failed_hour']) <= allowed + slack)
        scale = calibration['objective_scale']
        secondary = (beta_energy * (1-variables['delivered_energy_mwh']/calibration['energy_max_mwh'])
                     + calibration['lambda_b'] * variables['battery_throughput_mwh'])
        mdl.minimize(scale * (slack + secondary))
        mdl.parameters.timelimit = time_limit_seconds
        mdl.parameters.threads = threads
        mdl.parameters.randomseed = random_seed
        mdl.parameters.simplex.tolerances.feasibility = 1e-9
        mdl.parameters.simplex.tolerances.optimality = 1e-9
        mdl.parameters.mip.tolerances.integrality = 1e-9
        mdl.parameters.mip.tolerances.mipgap = mip_relative_gap
        mdl.parameters.mip.tolerances.absmipgap = mip_absolute_gap * scale
        sol = mdl.solve(log_output=log_output)
        details = mdl.solve_details
        code = details.status_code
        if sol is None or code not in (101, 102, 107):
            raise RuntimeError(f'CPLEX target-slack has no accepted incumbent: {details.status} '
                               f'(code {code}); not proof of target infeasibility')
        binary_raw = np.asarray(sol.get_values(variables['failed_hour']))
        if np.max(np.abs(binary_raw-np.rint(binary_raw))) > 1e-5:
            raise ValueError('Nonintegral failed-hour indicators')
        binary = np.rint(binary_raw).astype(int)
        excess = max(0, int(binary.sum())-allowed)
        raw_slack = float(sol.get_value(slack))
        if raw_slack < excess-1e-5:
            raise ValueError('Reliability slack constraint violated')
        bound = float(details.best_bound)/scale
        certificate = slack_certificate(bound, calibration['secondary_bound'], excess)
        stage = dict(stage='target_slack_energy', status=details.status, status_code=code,
                     seconds=float(details.time), objective=float(sol.objective_value)/scale,
                     objective_scaled=float(sol.objective_value), best_bound=bound,
                     relative_gap=float(details.mip_relative_gap),
                     objective_units='weighted excess failed hours (unscaled)')
        status = dict(backend='cplex', optimal=code in (101,102), feasible=True,
            termination='OPTIMAL_WITHIN_TOLERANCES' if code in (101,102) else 'TIME_LIMIT_INCUMBENT',
            stages=[stage], calibration=calibration, certificate=certificate,
            failed_hour_binary=binary.tolist(), binary_failed_hours=int(binary.sum()),
            reliability_target=float(reliability_target), max_failed_hours=allowed,
            reliability_slack=raw_slack, observed_excess_binary_failed_hours=excess,
            slack_matches_observed_excess=bool(abs(raw_slack-excess) <= 1e-5),
            mip_relative_gap_tolerance=mip_relative_gap, mip_absolute_gap_tolerance=mip_absolute_gap,
            random_seed=random_seed, threads=threads, time_limit_seconds=time_limit_seconds)
        result = ems_cplex_results_simple_datacenter(sol, variables, status, pue)
    hpp, curt, net, soc, charge, discharge, served_it, unmet_it, unmet, status = result
    wind, pv = np.asarray(wind_ts), np.asarray(solar_ts)
    hourly = dict(wind_mw=wind, pv_mw=pv, renewable_available_mw=wind+pv,
                  charge_mw=charge, discharge_mw=discharge,
                  firm_demand_it_mw=np.full(n, facility_load_mw/pue),
                  served_firm_it_mw=served_it, unserved_firm_it_mw=unmet_it,
                  facility_load_mw=np.full(n, facility_load_mw),
                  served_facility_mw=hpp, unserved_facility_mw=unmet, curtailment_mw=curt)
    status['physical_validation'] = validate_dispatch(
        hourly, soc, status, P_batt_MW, E_batt_MWh, battery_depth_of_discharge,
        charge_efficiency, battery_initial_soc, pue)
    if np.any(unmet > facility_load_mw*binary + 1e-5):
        raise ValueError('Failed-hour indicator allows an unreported shortfall')
    status['physically_validated'] = True
    status['metrics'] = firm_load_metrics(unmet, epsilon_mw=epsilon_mw)
    status['reported_excess_failed_hours'] = max(0, status['metrics']['LOLH']-allowed)
    # Binary indicators can remain 1 in fully served hours when the cap has room.
    # Record both counts rather than asserting that these are always identical.
    status['observed_shortfall_hours_1e_7'] = int(np.count_nonzero(unmet > 1e-7))
    status['binary_lolh_mismatch_hours'] = int(np.count_nonzero(binary != (unmet > epsilon_mw)))
    status['tolerance_band_shortfall_hours'] = int(np.count_nonzero((unmet > 1e-7) & (unmet <= epsilon_mw)))
    status['delivered_energy_mwh'] = float(hpp.sum())
    status['curtailment_mwh'] = float(curt.sum())
    status['battery_throughput_mwh'] = float((charge+discharge).sum())
    status['maximum_simultaneous_power_mw'] = float(np.minimum(charge,discharge).max())
    status['secondary_objective'] = (beta_energy * (1-hpp.sum()/calibration['energy_max_mwh'])
                                    + calibration['lambda_b']*status['battery_throughput_mwh'])
    if not -1e-7 <= status['secondary_objective'] <= calibration['secondary_bound']+1e-7:
        raise ValueError('Secondary objective exceeds calibrated bounds')
    return result
