"""Firm-load service metrics, independent of dispatch and equipment availability."""

import numpy as np


def firm_load_metrics(unserved_facility_mw, *, epsilon_mw=1e-3, delta_t_hours=1.0):
    """Count failed intervals; LOLH and EENS use the supplied interval duration.

    Epsilon is in facility MW, not IT MW. EENS includes small shortfalls even
    when they fall below the failure-count threshold. No annual extrapolation.
    """
    unmet = np.asarray(unserved_facility_mw, dtype=float)
    if unmet.ndim != 1 or not unmet.size or not np.all(np.isfinite(unmet)):
        raise ValueError("Unserved power must be a finite, nonempty 1-D array")
    if not np.isfinite(epsilon_mw) or epsilon_mw < 0:
        raise ValueError("epsilon_mw must be finite and nonnegative")
    if not np.isfinite(delta_t_hours) or delta_t_hours <= 0:
        raise ValueError("delta_t_hours must be positive and finite")
    if np.any(unmet < -1e-7):
        raise ValueError("Unserved power cannot be negative")
    unmet = np.maximum(unmet, 0)
    failed = int(np.count_nonzero(unmet > epsilon_mw))
    # Longest chronological shortage within the simulated horizon (no wrapping).
    longest = current = 0
    for failure in unmet > epsilon_mw:
        current = current + 1 if failure else 0
        longest = max(longest, current)
    return {
        "LOLH": failed * delta_t_hours,
        "firm_reliability": 1.0 - failed / unmet.size,
        "EENS": float(unmet.sum() * delta_t_hours),
        "successful_hours": (unmet.size - failed) * delta_t_hours,
        "total_hours": unmet.size * delta_t_hours,
        "epsilon_mw": epsilon_mw,
        "full_load_factor": 1.0 - failed / unmet.size,
        "longest_shortage_hours": longest * delta_t_hours,
    }


def validate_dispatch(
    hourly, soc_mwh, solver_status, b_P, b_E, battery_depth_of_discharge,
    battery_charge_efficiency, battery_initial_soc, pue, atol=1e-5,
):
    """Raise on invalid physics/status; return residual maxima for inspection."""
    h = hourly
    n = len(h["wind_mw"])
    required = (
        "wind_mw", "pv_mw", "renewable_available_mw", "charge_mw",
        "discharge_mw", "firm_demand_it_mw", "served_firm_it_mw",
        "unserved_firm_it_mw", "facility_load_mw", "served_facility_mw",
        "unserved_facility_mw", "curtailment_mw",
    )
    for name in required:
        a = np.asarray(h[name])
        if a.shape != (n,) or n == 0 or not np.all(np.isfinite(a)):
            raise ValueError(f"Invalid hourly array: {name}")
        if np.min(a) < -atol:
            raise ValueError(f"Negative power: {name}")
    soc = np.asarray(soc_mwh)
    if soc.shape != (n + 1,) or not np.all(np.isfinite(soc)):
        raise ValueError("SOC must contain T+1 finite states")
    if not (solver_status.get("optimal", False) or solver_status.get("feasible", False)):
        raise ValueError("Dispatch has no solver-certified feasible solution")
    dt = 1.0
    residuals = {
        "renewable_balance_mw": h["renewable_available_mw"] - h["wind_mw"] - h["pv_mw"],
        "power_balance_mw": h["renewable_available_mw"] + h["discharge_mw"]
        - h["served_facility_mw"] - h["charge_mw"] - h["curtailment_mw"],
        "demand_balance_mw": h["served_facility_mw"] + h["unserved_facility_mw"] - h["facility_load_mw"],
        "it_balance_mw": h["served_firm_it_mw"] + h["unserved_firm_it_mw"] - h["firm_demand_it_mw"],
        "served_pue_mw": pue * h["served_firm_it_mw"] - h["served_facility_mw"],
        "unserved_pue_mw": pue * h["unserved_firm_it_mw"] - h["unserved_facility_mw"],
        "demand_pue_mw": pue * h["firm_demand_it_mw"] - h["facility_load_mw"],
        "soc_transition_mwh": np.diff(soc) - dt * (
            battery_charge_efficiency * h["charge_mw"] - h["discharge_mw"] / battery_charge_efficiency),
    }
    maxima = {k: float(np.max(np.abs(v))) for k, v in residuals.items()}
    if any(v > atol for v in maxima.values()):
        raise ValueError(f"Physical balance failure: {maxima}")
    initial = battery_initial_soc * b_E
    if abs(soc[0] - initial) > atol or abs(soc[-1] - initial) > atol:
        raise ValueError("Initial/terminal SOC mismatch")
    if (soc.min() < (1 - battery_depth_of_discharge) * b_E - atol
            or soc.max() > b_E + atol):
        raise ValueError("SOC outside bounds")
    if (h["charge_mw"].max() > b_P / battery_charge_efficiency + atol
            or h["discharge_mw"].max() > b_P * battery_charge_efficiency + atol):
        raise ValueError("Battery power outside bounds")
    if np.any(h["curtailment_mw"] > h["renewable_available_mw"] + atol):
        raise ValueError("Curtailment exceeds available renewables")
    for name in ("served_facility_mw", "unserved_facility_mw"):
        if np.any(h[name] > h["facility_load_mw"] + atol):
            raise ValueError(f"Demand bound violated: {name}")
    if np.any(np.minimum(h["charge_mw"], h["discharge_mw"]) > atol):
        raise ValueError("Simultaneous battery charging/discharging")
    return maxima


