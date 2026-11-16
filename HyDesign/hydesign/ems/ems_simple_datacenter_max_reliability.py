"""Alternative fixed-load, hourly maximum-availability EMS (one CPLEX solve).

Reuses the original physical constraints with its reliability cap disabled.
There is deliberately no reliability-target argument, assembly or EGO hook.
All powers are facility-bus MW except the battery rating (storage-side MW).
Time steps are one hour, exactly as in the original EMS.
"""

import numpy as np
from docplex.mp.model import Model
from docplex.mp.progress import ProgressListener

from hydesign.ems.ems_simple_datacenter import (
    ems_cplex_constraints_simple_datacenter,
    ems_cplex_results_simple_datacenter,
)
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch


def calibrate_penalties(renewable, P_batt_MW, charge_efficiency,
                        beta=0.25, kappa=0.10):
    """Bound the entire secondary objective by beta < one failed hour.

    C_bar = sum(renewable) and Q_bar = T * (P/eta + P*eta), in MWh.
    lambda_C = beta/(C_bar + kappa*Q_bar); lambda_B = kappa*lambda_C.
    kappa must exceed (1-eta**2)/(1+eta**2): otherwise converting curtailment
    into battery losses can improve the secondary objective at fixed service.
    This check does not replace strict validation of simultaneous operation.
    """
    renewable = np.asarray(renewable, dtype=float)
    if (renewable.ndim != 1 or not renewable.size
            or not np.all(np.isfinite(renewable)) or np.any(renewable < 0)):
        raise ValueError("Renewable generation must be finite and nonnegative")
    if (not np.all(np.isfinite([P_batt_MW, charge_efficiency, beta, kappa]))
            or P_batt_MW < 0 or not 0 < charge_efficiency <= 1
            or not 0 < beta < 1):
        raise ValueError("Invalid battery rating, efficiency or beta (require 0 < beta < 1)")
    loss_fraction = (1 - charge_efficiency**2) / (1 + charge_efficiency**2)
    if kappa <= loss_fraction:
        raise ValueError("kappa must exceed the battery loss/throughput fraction")
    curtailment_bound = float(renewable.sum())
    throughput_bound = float(len(renewable) * P_batt_MW
                             * (1 / charge_efficiency + charge_efficiency))
    denominator = curtailment_bound + kappa * throughput_bound
    lambda_c = beta / denominator if denominator else 0.0
    # Scale the WHOLE objective, not its relative weights. This keeps the
    # curtailment coefficient at one instead of using very small coefficients.
    scale = 1 / lambda_c if lambda_c else 1.0
    return dict(beta=float(beta), kappa=float(kappa), lambda_c=lambda_c,
                lambda_b=kappa * lambda_c, objective_scale=scale,
                curtailment_bound_mwh=curtailment_bound,
                throughput_bound_mwh=throughput_bound,
                secondary_bound=beta if denominator else 0.0,
                loss_throughput_fraction=loss_fraction)


def failed_hour_certificate(best_bound, binary_failed_hours, secondary_bound,
                            n_hours, tolerance=1e-5):
    """Conservative integer bound for EXACT-service failures, in unscaled units.

    LB <= F* + secondary <= F* + beta, hence F* >= ceil(LB-beta).
    Does not claim an optimum for the epsilon-based reported LOLH definition.
    """
    lower = (max(0, int(np.ceil(best_bound - secondary_bound - tolerance)))
             if np.isfinite(best_bound) else 0)
    if lower > binary_failed_hours:
        raise ValueError("Inconsistent failed-hour bound and incumbent")
    return dict(failed_hours_lower_bound=lower,
                failed_hours_upper_bound=int(binary_failed_hours),
                maximum_availability_certified=lower == binary_failed_hours,
                exact_reliability_lower_bound=1 - binary_failed_hours / n_hours,
                exact_reliability_upper_bound=1 - lower / n_hours,
                certificate_scope="exact-service binary count; not epsilon-based LOLH")


class _AvailabilityProgress(ProgressListener):
    """Observe a count certificate without stopping or adding another solve."""

    def __init__(self, scale, secondary_bound, n_hours):
        super().__init__()
        self.scale = scale
        self.secondary_bound = secondary_bound
        self.n_hours = n_hours
        self.first_certificate = None

    def notify_progress(self, data):
        if self.first_certificate is not None or not data.has_incumbent:
            return
        # J = integer F + a secondary term in [0,beta), beta < 1.
        objective = data.current_objective / self.scale
        count = int(np.floor(objective + 1e-7))
        if count < 0 or count > self.n_hours:
            return
        certificate = failed_hour_certificate(
            data.best_bound / self.scale, count, self.secondary_bound, self.n_hours)
        if certificate["maximum_availability_certified"]:
            self.first_certificate = dict(seconds=float(data.time),
                                          relative_gap=float(data.mip_gap),
                                          binary_failed_hours=count)


def ems_cplex_constraints_simple_datacenter_max_reliability(
    mdl, wind_ts, solar_ts, P_batt_MW, E_batt_MWh,
    battery_depth_of_discharge, charge_efficiency, battery_initial_soc,
    facility_load_mw, pue,
):
    """Original balances/bounds plus failure indicators, without a failure cap."""
    variables = ems_cplex_constraints_simple_datacenter(
        mdl, wind_ts, solar_ts, P_batt_MW, E_batt_MWh,
        battery_depth_of_discharge, charge_efficiency, battery_initial_soc,
        facility_load_mw, facility_load_mw, pue, reliability_target=0.0,
    )
    renewable = np.asarray(wind_ts) + np.asarray(solar_ts)
    failed = mdl.binary_var_list(len(renewable), name="failed_hour")
    for t, indicator in enumerate(failed):
        mdl.add_constraint(pue * variables["P_unserved_IT_t"][t]
                           <= facility_load_mw * indicator)
        # Same valid surplus fixing as the original EMS, no new physics.
        if renewable[t] >= facility_load_mw + P_batt_MW / charge_efficiency:
            indicator.ub = 0
    variables["failed_hour"] = failed
    variables.pop("max_failed_hours")
    return variables


def solve_cplex_simple_datacenter_max_reliability(
    mdl, variables, calibration, time_limit_seconds=600, threads=4,
    log_output=False, mip_relative_gap=1e-4, mip_absolute_gap=1e-6,
    random_seed=0,
):
    """Minimize F + lambda_C*C + lambda_B*Q in one, uniformly scaled solve.

    Gap defaults are ordinary CPLEX MIP tolerances; absolute gap is specified
    in the UNscaled objective. A valid time-limit incumbent is returned with
    explicit status. The public entry point physically validates it first.
    """
    if (not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0
            or not np.all(np.isfinite([mip_relative_gap, mip_absolute_gap]))
            or min(mip_relative_gap, mip_absolute_gap) < 0):
        raise ValueError("Invalid time limit or MIP tolerances")
    scale = calibration["objective_scale"]
    failed = mdl.sum(variables["failed_hour"])
    curtailment = mdl.sum(variables["P_curtailment_t"])
    throughput = variables["battery_throughput_mwh"]
    mdl.minimize(scale * failed
                 + scale * calibration["lambda_c"] * curtailment
                 + scale * calibration["lambda_b"] * throughput)
    mdl.parameters.threads = threads
    mdl.parameters.timelimit = time_limit_seconds
    mdl.parameters.randomseed = random_seed
    # Preserve the original physical feasibility/integrality tolerances.
    mdl.parameters.simplex.tolerances.feasibility = 1e-9
    mdl.parameters.simplex.tolerances.optimality = 1e-9
    mdl.parameters.mip.tolerances.integrality = 1e-9
    mdl.parameters.mip.tolerances.mipgap = mip_relative_gap
    mdl.parameters.mip.tolerances.absmipgap = mip_absolute_gap * scale
    progress = _AvailabilityProgress(scale, calibration["secondary_bound"],
                                     len(variables["failed_hour"]))
    mdl.add_progress_listener(progress)
    sol = mdl.solve(log_output=log_output)
    details = mdl.solve_details
    code = getattr(details, "status_code", None)
    optimal = code in (101, 102)
    if sol is None or not (optimal or code == 107):
        raise RuntimeError(f"CPLEX maximum availability has no accepted incumbent: "
                           f"{getattr(details, 'status', 'no solve details')} (code {code}); "
                           "not proof of target infeasibility")
    indicators = np.asarray(sol.get_values(variables["failed_hour"]))
    if np.max(np.abs(indicators - np.rint(indicators))) > 1e-5:
        raise ValueError("Nonintegral failed-hour indicators")
    binary_count = int(np.rint(indicators).sum())
    bound = float(details.best_bound) / scale
    certificate = failed_hour_certificate(bound, binary_count,
        calibration["secondary_bound"], len(indicators))
    stage = dict(stage="maximum_availability", status=details.status,
                 status_code=code, seconds=float(details.time),
                 objective=float(sol.objective_value) / scale,
                 objective_units="weighted failed hours (unscaled)",
                 objective_scaled=float(sol.objective_value), best_bound=bound,
                 relative_gap=float(details.mip_relative_gap))
    first = progress.first_certificate
    if first is None and certificate["maximum_availability_certified"]:
        first = dict(seconds=stage["seconds"], relative_gap=stage["relative_gap"],
                     binary_failed_hours=binary_count)
    status = dict(backend="cplex", optimal=optimal, feasible=True, stages=[stage],
                  termination="OPTIMAL_WITHIN_TOLERANCES" if optimal else "TIME_LIMIT_INCUMBENT",
                  calibration=calibration, failed_hour_binary=np.rint(indicators).astype(int).tolist(),
                  binary_failed_hours=binary_count, certificate=certificate,
                  first_availability_certificate=first,
                  mip_relative_gap_tolerance=mip_relative_gap,
                  mip_absolute_gap_tolerance=mip_absolute_gap,
                  random_seed=random_seed, threads=threads,
                  time_limit_seconds=time_limit_seconds)
    return sol, status


def ems_cplex_simple_datacenter_max_reliability(
    wind_ts, solar_ts, P_batt_MW, E_batt_MWh,
    battery_depth_of_discharge, charge_efficiency, battery_initial_soc,
    facility_load_mw, pue, time_limit_seconds=600, threads=4,
    log_output=False, beta=0.25, kappa=0.10, epsilon_mw=1e-3,
    mip_relative_gap=1e-4, mip_absolute_gap=1e-6, random_seed=0,
):
    """Return the original nine dispatch arrays plus status/metrics/certificate.

    No outer reliability target is accepted or imposed. Validation is strict,
    including rejecting simultaneous charge/discharge above 1e-5 MW, even for
    a CPLEX-feasible time-limit incumbent. No exclusivity binaries are added.
    """
    if not np.isfinite(facility_load_mw) or facility_load_mw < 0:
        raise ValueError("Facility load must be finite and nonnegative")
    with Model(name="EMS_simple_datacenter_max_reliability") as mdl:
        variables = ems_cplex_constraints_simple_datacenter_max_reliability(
            mdl, wind_ts, solar_ts, P_batt_MW, E_batt_MWh,
            battery_depth_of_discharge, charge_efficiency, battery_initial_soc,
            facility_load_mw, pue)
        calibration = calibrate_penalties(np.asarray(wind_ts) + np.asarray(solar_ts),
                                          P_batt_MW, charge_efficiency, beta, kappa)
        sol, status = solve_cplex_simple_datacenter_max_reliability(
            mdl, variables, calibration, time_limit_seconds, threads, log_output,
            mip_relative_gap, mip_absolute_gap, random_seed)
        result = ems_cplex_results_simple_datacenter(sol, variables, status, pue)
    hpp, curt, net, soc, charge, discharge, served_it, unmet_it, unmet, status = result
    wind, pv = np.asarray(wind_ts), np.asarray(solar_ts)
    hourly = dict(wind_mw=wind, pv_mw=pv, renewable_available_mw=wind + pv,
                  charge_mw=charge, discharge_mw=discharge,
                  firm_demand_it_mw=np.full(len(wind), facility_load_mw / pue),
                  served_firm_it_mw=served_it, unserved_firm_it_mw=unmet_it,
                  facility_load_mw=np.full(len(wind), facility_load_mw),
                  served_facility_mw=hpp, unserved_facility_mw=unmet,
                  curtailment_mw=curt)
    status["physical_validation"] = validate_dispatch(
        hourly, soc, status, P_batt_MW, E_batt_MWh, battery_depth_of_discharge,
        charge_efficiency, battery_initial_soc, pue)
    status["physically_validated"] = True
    status["metrics"] = firm_load_metrics(unmet, epsilon_mw=epsilon_mw)
    binary = np.asarray(status["failed_hour_binary"], dtype=bool)
    if np.any(unmet > facility_load_mw * binary + 1e-5):
        raise ValueError("Failed-hour indicator permits an unreported shortfall")
    status["binary_lolh_mismatch_hours"] = int(np.count_nonzero(binary != (unmet > epsilon_mw)))
    status["tolerance_band_shortfall_hours"] = int(np.count_nonzero((unmet > 1e-7) & (unmet <= epsilon_mw)))
    status["delivered_energy_mwh"] = float(hpp.sum())
    status["curtailment_mwh"] = float(curt.sum())
    status["battery_throughput_mwh"] = float((charge + discharge).sum())
    status["maximum_simultaneous_power_mw"] = float(np.minimum(charge, discharge).max())
    status["secondary_objective"] = (calibration["lambda_c"] * status["curtailment_mwh"]
                                      + calibration["lambda_b"] * status["battery_throughput_mwh"])
    if status["secondary_objective"] > calibration["secondary_bound"] + 1e-7:
        raise ValueError("Secondary objective exceeds its calibrated bound")
    return result
