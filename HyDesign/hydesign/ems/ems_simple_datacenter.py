# basic libraries
import numpy as np
import pandas as pd
from docplex.mp.model import Model
import warnings

from hydesign.openmdao_wrapper import ComponentWrapper


class ems_simple_datacenter:
    """Off-grid firm-load energy management model.

    Follows ems.py: compute prepares the time series, calls the CPLEX function
    and returns arrays in the order declared by the ComponentWrapper.
    Uses one full hourly chronology, without batching or lifetime expansion.

    Parameters
    ----------
    N_time : Number of hourly timesteps.
    ems_type : Solver type; only CPLEX is supported.

    Notes
    -----
    Reference: Thijs van Waveren's ems_offgrid_datacenter.py, commit
    a51e60652a1f55a2b055d3b07352a4ed541898c9. The final SOC transition is
    connected here. Demand is facility MW, converted to IT MW using PUE.
    """

    def __init__(self, N_time, ems_type="cplex", time_limit_seconds=600, threads=4):
        self.N_time = int(N_time)
        self.ems_type = ems_type
        self.time_limit_seconds = time_limit_seconds
        self.threads = threads
        self.solver_status = None

    def compute(
        self,
        wind_t,
        solar_t,
        b_P,
        b_E,
        battery_depth_of_discharge,
        battery_charge_efficiency,
        battery_initial_soc,
        facility_load_mw,
        pue,
        reliability_target=0.0,
        ppa_price=70.0,
        battery_power_fluctuation_penalty=0.0,
        **kwargs,
    ):
        if self.ems_type != "cplex":
            raise ValueError("This class should only be used for CPLEX")
        if np.shape(wind_t) != (self.N_time,) or np.shape(solar_t) != (self.N_time,):
            raise ValueError("Generation arrays must have N_time entries")
        self.solver_status = None

        # Synthetic hourly index, following the original EMS compute method.
        WSPr_df = pd.DataFrame(
            index=pd.date_range(start="01-01-1991 00:00", periods=self.N_time, freq="1h")
        )
        WSPr_df["wind_t"] = wind_t
        WSPr_df["solar_t"] = solar_t

        (
            hpp_t,
            hpp_curt_t,
            b_t,
            b_E_SOC_t,
            charge_mw,
            discharge_mw,
            served_firm_it_mw,
            unserved_firm_it_mw,
            unserved_facility_mw,
            self.solver_status,
        ) = ems_cplex_simple_datacenter(
            wind_ts=WSPr_df.wind_t,
            solar_ts=WSPr_df.solar_t,
            P_batt_MW=np.asarray(b_P).item(),
            E_batt_MWh=np.asarray(b_E).item(),
            battery_depth_of_discharge=np.asarray(battery_depth_of_discharge).item(),
            charge_efficiency=np.asarray(battery_charge_efficiency).item(),
            battery_initial_soc=np.asarray(battery_initial_soc).item(),
            facility_load_mw=np.asarray(facility_load_mw).item(),
            pue=np.asarray(pue).item(),
            reliability_target=np.asarray(reliability_target).item(),
            ppa_price=np.asarray(ppa_price).item(),
            time_limit_seconds=self.time_limit_seconds,
            threads=self.threads,
            battery_power_fluctuation_penalty=np.asarray(battery_power_fluctuation_penalty).item(),
        )
        return (
            hpp_t,
            hpp_curt_t,
            b_t,
            b_E_SOC_t,
            charge_mw,
            discharge_mw,
            served_firm_it_mw,
            unserved_firm_it_mw,
            unserved_facility_mw,
        )


class ems_simple_datacenter_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = ems_simple_datacenter(**insta_inp)
        self.ems_model = model
        super().__init__(
            inputs=[
                (
                    "wind_t",
                    {
                        "units": "MW",
                        "desc": "WPP power time series",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "solar_t",
                    {
                        "units": "MW",
                        "desc": "PVP power time series",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "b_P",
                    {
                        "units": "MW",
                        "desc": "Battery power capacity",
                    },
                ),
                (
                    "b_E",
                    {
                        "units": "MW*h",
                        "desc": "Nominal battery energy capacity",
                    },
                ),
                (
                    "battery_depth_of_discharge",
                    {
                        "desc": "Battery depth of discharge",
                    },
                ),
                (
                    "battery_charge_efficiency",
                    {
                        "desc": "Efficiency per direction",
                    },
                ),
                (
                    "battery_initial_soc",
                    {
                        "desc": "Initial and terminal SOC fraction",
                    },
                ),
                (
                    "facility_load_mw",
                    {
                        "units": "MW",
                        "desc": "Constant facility demand",
                    },
                ),
                (
                    "pue",
                    {
                        "desc": "Facility power / IT power",
                    },
                ),
                (
                    "reliability_target",
                    {"val": 0.0, "desc": "Minimum hourly full-load factor"},
                ),
                (
                    "ppa_price",
                    {"val": 70.0, "desc": "Reporting only: payment per facility MWh [EUR/MWh]"},
                ),
                (
                    "battery_power_fluctuation_penalty",
                    {"val": 0.0, "desc": "MWh_IT penalty per MW of adjacent AC net-battery-power change"},
                ),
            ],
            outputs=[
                (
                    "hpp_t",
                    {
                        "units": "MW",
                        "desc": "Served facility power",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "hpp_curt_t",
                    {
                        "units": "MW",
                        "desc": "Curtailed renewable power",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "b_t",
                    {
                        "units": "MW",
                        "desc": "Battery power: positive discharge",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "b_E_SOC_t",
                    {
                        "units": "MW*h",
                        "desc": "Stored energy, including terminal state",
                        "shape": [model.N_time + 1],
                    },
                ),
                (
                    "charge_mw",
                    {
                        "units": "MW",
                        "desc": "Battery charging at facility bus",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "discharge_mw",
                    {
                        "units": "MW",
                        "desc": "Battery discharging at facility bus",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "served_firm_it_mw",
                    {
                        "units": "MW",
                        "desc": "Served IT demand",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "unserved_firm_it_mw",
                    {
                        "units": "MW",
                        "desc": "Unserved IT demand",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "unserved_facility_mw",
                    {
                        "units": "MW",
                        "desc": "Unserved facility demand",
                        "shape": [model.N_time],
                    },
                ),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


def ems_cplex_simple_datacenter(
    wind_ts,
    solar_ts,
    P_batt_MW,
    E_batt_MWh,
    battery_depth_of_discharge,
    charge_efficiency,
    battery_initial_soc,
    facility_load_mw,
    pue,
    reliability_target=0.0,
    ppa_price=70.0,
    time_limit_seconds=600,
    threads=4,
    log_output=False,
    battery_power_fluctuation_penalty=0.0,
):
    """Dispatch a FIXED load under an hourly full-load-factor constraint.

    Maximize served IT energy minus an optional net-battery-power fluctuation
    penalty. PPA price is retained for call compatibility, not optimization.
    The reliability constraint separately limits failed hours.
    The sizing variable is defined only in sizing_simple_datacenter.py.
    """
    if not np.isfinite(facility_load_mw) or facility_load_mw < 0:
        raise ValueError("Facility load must be finite and nonnegative")
    with Model(name="EMS_simple_datacenter") as mdl:
        variables = ems_cplex_constraints_simple_datacenter(
            mdl, wind_ts, solar_ts, P_batt_MW, E_batt_MWh,
            battery_depth_of_discharge, charge_efficiency, battery_initial_soc,
            facility_load_mw, facility_load_mw, pue, reliability_target,
        )
        sol, solver_status = solve_cplex_simple_datacenter(
            mdl, variables, ppa_price, time_limit_seconds, threads, log_output,
            battery_power_fluctuation_penalty=battery_power_fluctuation_penalty,
        )
        return ems_cplex_results_simple_datacenter(sol, variables, solver_status, pue)


def ems_cplex_constraints_simple_datacenter(
    mdl, wind_ts, solar_ts, P_batt_MW, E_batt_MWh,
    battery_depth_of_discharge, charge_efficiency, battery_initial_soc,
    facility_load_mw, facility_load_upper_mw, pue, reliability_target,
):
    """Shared hourly constraints for fixed-load dispatch and external sizing.

    facility_load_mw may be a fixed number or a CPLEX variable supplied by the
    sizing module. This function never creates or chooses plant/load capacities.
    A failed-hour binary permits a shortfall; at most floor((1-R)*T) are allowed.
    Success requires full supply exactly; the 1e-3 MW reporting tolerance absorbs
    solver roundoff, rather than being an intentional service allowance.
    """
    wind = np.asarray(wind_ts, dtype=float)
    pv = np.asarray(solar_ts, dtype=float)
    if (wind.ndim != 1 or wind.size == 0 or pv.shape != wind.shape
            or not np.all(np.isfinite(wind)) or not np.all(np.isfinite(pv))
            or np.any(wind < 0) or np.any(pv < 0)):
        raise ValueError("Generation must be matching finite nonnegative 1-D arrays")
    parameters = [P_batt_MW, E_batt_MWh, battery_depth_of_discharge,
                  charge_efficiency, battery_initial_soc, facility_load_upper_mw,
                  pue, reliability_target]
    if not np.all(np.isfinite(parameters)):
        raise ValueError("EMS parameters must be finite")
    if P_batt_MW < 0 or E_batt_MWh < 0 or facility_load_upper_mw < 0 or pue < 1:
        raise ValueError("Invalid battery rating, load bound or PUE")
    if not 0 < charge_efficiency <= 1 or not 0 <= battery_depth_of_discharge <= 1:
        raise ValueError("Invalid battery efficiency or depth of discharge")
    if not 1 - battery_depth_of_discharge <= battery_initial_soc <= 1:
        raise ValueError("Initial SOC must lie within the usable SOC range")
    if not 0 <= reliability_target <= 1:
        raise ValueError("Reliability target must be a fraction in [0, 1]")

    N_t = len(wind)
    renewable = wind + pv
    max_failed_hours = int(np.floor((1 - reliability_target) * N_t + 1e-9))

    # Variables definition, using the original EMS naming convention.
    P_served_IT_t = mdl.continuous_var_list(
        N_t, lb=0, ub=facility_load_upper_mw / pue, name="served_it"
    )
    P_unserved_IT_t = mdl.continuous_var_list(
        N_t, lb=0, ub=facility_load_upper_mw / pue, name="unserved_it"
    )
    P_charge = mdl.continuous_var_list(
        N_t, lb=0, ub=P_batt_MW / charge_efficiency, name="charge"
    )
    P_discharge = mdl.continuous_var_list(
        N_t, lb=0, ub=P_batt_MW * charge_efficiency, name="discharge"
    )
    P_curtailment_t = mdl.continuous_var_list(
        N_t, lb=0, ub=renewable.tolist(), name="curtailment"
    )
    E_SOC_t = mdl.continuous_var_list(
        N_t + 1, lb=(1 - battery_depth_of_discharge) * E_batt_MWh,
        ub=E_batt_MWh, name="soc"
    )
    # R=0 is an unconstrained diagnostic case; binaries are unnecessary there.
    failed_hour = mdl.binary_var_list(N_t, name="failed_hour") if reliability_target > 0 else []

    # Initial and terminal SOC.
    initial = battery_initial_soc * E_batt_MWh
    mdl.add_constraints([E_SOC_t[0] == initial, E_SOC_t[-1] == initial])

    # Hourly power, demand and SOC constraints; dt = 1 hour.
    for t in range(N_t):
        mdl.add_constraint(
            P_served_IT_t[t] + P_unserved_IT_t[t] == facility_load_mw / pue
        )
        mdl.add_constraint(
            renewable[t] + P_discharge[t]
            == pue * P_served_IT_t[t] + P_charge[t] + P_curtailment_t[t]
        )
        mdl.add_constraint(
            E_SOC_t[t + 1] == E_SOC_t[t]
            + charge_efficiency * P_charge[t] - P_discharge[t] / charge_efficiency
        )
        if failed_hour:
            mdl.add_constraint(
                pue * P_unserved_IT_t[t] <= facility_load_upper_mw * failed_hour[t]
            )
            # If generation covers the largest load AND maximum charging, a
            # shortfall cannot help the battery or reliability and loses revenue.
            if renewable[t] >= facility_load_upper_mw + P_batt_MW / charge_efficiency:
                failed_hour[t].ub = 0
    if failed_hour:
        mdl.add_constraint(mdl.sum(failed_hour) <= max_failed_hours)

    return {
        "P_served_IT_t": P_served_IT_t,
        "P_unserved_IT_t": P_unserved_IT_t,
        "P_charge": P_charge,
        "P_discharge": P_discharge,
        "P_curtailment_t": P_curtailment_t,
        "E_SOC_t": E_SOC_t,
        "failed_hour": failed_hour,
        "max_failed_hours": max_failed_hours,
        "delivered_energy_mwh": pue * mdl.sum(P_served_IT_t),
        "battery_throughput_mwh": mdl.sum(P_charge + P_discharge),
    }


def battery_power_fluctuation_expression(mdl, variables, penalty_weight):
    """Sum absolute adjacent changes in AC net battery power [MW].

    penalty_weight is nonnegative, in MWh_IT per MW of fluctuation (hours).
    Economic callers multiply it by their IT-energy revenue rate. With zero
    weight no auxiliary variables/rows are needed. There is no end-to-start
    ramp, no throughput penalty, and no restriction on simultaneous operation.
    """
    if not np.isfinite(penalty_weight) or penalty_weight < 0:
        raise ValueError("battery_power_fluctuation_penalty must be finite and nonnegative")
    if penalty_weight == 0:
        return mdl.sum([])
    net = [discharge - charge for charge, discharge in
           zip(variables["P_charge"], variables["P_discharge"])]
    change = mdl.continuous_var_list(max(len(net) - 1, 0), lb=0, name="battery_power_change")
    for t, absolute in enumerate(change):
        mdl.add_constraint(absolute >= net[t + 1] - net[t])
        mdl.add_constraint(absolute >= net[t] - net[t + 1])
    return mdl.sum(change)


def solve_cplex_simple_datacenter(
    mdl, variables, ppa_price=70.0, time_limit_seconds=600, threads=4,
    log_output=False, load_variable=None, mip_relative_gap=1e-4,
    allow_time_limit_incumbent=False, battery_power_fluctuation_penalty=0.0,
):
    """One solve: served IT energy minus optional battery-power fluctuations.

    One-hour intervals: lambda * sum(abs(diff(discharge-charge))) is in MWh_IT.
    ppa_price and load_variable are accepted for compatibility only; neither
    PPA revenue nor a fixed-load cleanup constraint enters the optimization.
    Accepted time-limit incumbents retain their status and objective gap.
    """
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("Time limit must be positive and finite")
    fluctuation = battery_power_fluctuation_expression(
        mdl, variables, battery_power_fluctuation_penalty,
    )
    served_it_energy = mdl.sum(variables["P_served_IT_t"])
    mdl.parameters.threads = threads
    mdl.parameters.timelimit = time_limit_seconds
    mdl.parameters.simplex.tolerances.feasibility = 1e-9
    mdl.parameters.simplex.tolerances.optimality = 1e-9
    mdl.parameters.mip.tolerances.integrality = 1e-9
    mdl.parameters.mip.tolerances.mipgap = mip_relative_gap
    mdl.maximize(served_it_energy - battery_power_fluctuation_penalty * fluctuation)
    stage = "maximum_served_it_energy"
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
    # Evaluate actual flows, not auxiliary ramp bounds (especially at lambda=0).
    actual_fluctuation = float(np.abs(np.diff(net)).sum())
    penalty = battery_power_fluctuation_penalty * actual_fluctuation
    info.update({
        "objective": float(sol.objective_value), "objective_units": "MWh_IT",
        "unpenalized_objective": float(sol.get_value(served_it_energy)),
        "battery_penalty_equivalent_it_mwh": penalty,
        "battery_power_fluctuation_mw": actual_fluctuation,
    })
    if mdl.number_of_binary_variables:
        info["relative_gap"] = float(details.mip_relative_gap)
        info["best_bound"] = float(details.best_bound)
    solver_status = {
        "backend": "cplex", "optimal": optimal, "feasible": True, "stages": [info],
        "primary_served_it_energy_mwh": info["unpenalized_objective"],
        "battery_power_fluctuation_penalty": battery_power_fluctuation_penalty,
        "battery_power_fluctuation_mw": actual_fluctuation,
        "battery_penalty_equivalent_it_mwh": penalty,
        "mip_relative_gap_tolerance": mip_relative_gap,
        "max_failed_hours": variables["max_failed_hours"],
    }
    if not optimal:
        warnings.warn(
            "CPLEX reached the time limit: returning a feasible candidate, not a "
            f"proven optimum. Penalized served-energy gap: {info['relative_gap']:.2%}.",
            RuntimeWarning,
        )
    return sol, solver_status


def ems_cplex_results_simple_datacenter(sol, variables, solver_status, pue):
    """Extract arrays in the same order as the EMS compute/ComponentWrapper."""
    P_served_IT_ts = np.asarray(sol.get_values(variables["P_served_IT_t"]))
    P_unserved_IT_ts = np.asarray(sol.get_values(variables["P_unserved_IT_t"]))
    P_charge_ts = np.asarray(sol.get_values(variables["P_charge"]))
    P_discharge_ts = np.asarray(sol.get_values(variables["P_discharge"]))
    return (
        pue * P_served_IT_ts,
        np.asarray(sol.get_values(variables["P_curtailment_t"])),
        P_discharge_ts - P_charge_ts,
        np.asarray(sol.get_values(variables["E_SOC_t"])),
        P_charge_ts,
        P_discharge_ts,
        P_served_IT_ts,
        P_unserved_IT_ts,
        pue * P_unserved_IT_ts,
        solver_status,
    )
