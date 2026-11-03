import numpy as np
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.ems.ems_simple_datacenter import ems_simple_datacenter_comp as ems
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch
from hydesign.pv.pv import pvp_comp as pvp
from hydesign.weather.weather import ABL_comp as ABL
from hydesign.wind.wind import genericWake_surrogate_comp as genericWake_surrogate
from hydesign.wind.wind import genericWT_surrogate_comp as genericWT_surrogate
from hydesign.wind.wind import get_rotor_d
from hydesign.wind.wind import wpp_comp as wpp


class hpp_model_simple_datacenter(hpp_base):
    """Fixed off-grid HPP evaluator, following hpp_assembly_constantoutput.py.

    Uses the original hpp_base input handling and component assembly pattern.
    Only ABL, turbine, wake, wind, PV and EMS components are assembled.
    No finance, degradation, lifetime operation or outer sizing is run.
    """

    def __init__(self, sim_pars_fn, **kwargs):
        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        N_time = self.N_time
        N_ws = self.N_ws
        sim_pars = self.sim_pars
        input_ts_fn = sim_pars["input_ts_fn"]

        comps = [
            ("abl", ABL(weather_fn=input_ts_fn, N_time=N_time)),
            ("genericWT", genericWT_surrogate(genWT_fn=sim_pars["genWT_fn"], N_ws=N_ws)),
            ("genericWake", genericWake_surrogate(genWake_fn=sim_pars["genWake_fn"], N_ws=N_ws)),
            ("wpp", wpp(N_time=N_time, N_ws=N_ws, wpp_efficiency=self.wpp_efficiency)),
            (
                "pvp",
                pvp(
                    weather_fn=input_ts_fn,
                    N_time=N_time,
                    latitude=sim_pars["latitude"],
                    longitude=sim_pars["longitude"],
                    altitude=sim_pars["altitude"],
                    tracking=sim_pars["tracking"],
                ),
            ),
            (
                "ems",
                ems(
                    N_time=N_time,
                    ems_type=sim_pars["ems_type"],
                    time_limit_seconds=sim_pars["ems_time_limit_seconds"],
                    threads=sim_pars["ems_threads"],
                ),
            ),
        ]
        prob = self.get_prob(comps)
        prob.setup()

        # Additional parameters, as in the original assembly.
        prob.set_val("land_use_per_solar_MW", sim_pars["land_use_per_solar_MW"])
        prob.set_val("battery_depth_of_discharge", sim_pars["battery_depth_of_discharge"])
        prob.set_val("battery_charge_efficiency", sim_pars["battery_charge_efficiency"])
        prob.set_val("battery_initial_soc", sim_pars["battery_initial_soc"])
        prob.set_val("facility_load_mw", sim_pars["facility_load_mw"])
        prob.set_val("pue", sim_pars["pue"])
        prob.set_val("reliability_target", sim_pars["reliability_target"])
        prob.set_val("ppa_price", sim_pars["ppa_price"])  # Reporting only.
        prob.set_val("battery_power_fluctuation_penalty",
                     sim_pars.get("battery_power_fluctuation_penalty", 0.0))
        self.prob = prob

        self.list_vars = [
            "clearance [m]", "sp [W/m2]", "p_rated [MW]", "Nwt",
            "wind_MW_per_km2 [MW/km2]", "solar_MW [MW AC]",
            "surface_tilt [deg]", "surface_azimuth [deg]", "DC_AC_ratio",
            "b_P [MW]", "b_E_h [h]",
        ]
        self.list_out_vars = ["LOLH [h]", "firm_reliability [-]", "EENS [MWh]"]

    def check_inputs(self, sim_pars):
        # Check before hpp_base can truncate/sample a nonannual weather file.
        super().check_inputs(sim_pars)
        if sim_pars.get("input_ts_fn") is None:
            raise ValueError("Stage 1 requires an existing hourly weather file")
        if sim_pars.get("weeks_per_season_per_year") is not None:
            raise ValueError("Stage 1 requires the full year, without sampling")
        if sim_pars.get("life_y", sim_pars.get("N_life")) != 1:
            raise ValueError("Stage 1 uses one weather year, not lifetime operation")
        weather = pd.read_csv(sim_pars["input_ts_fn"], index_col=0, parse_dates=True)
        index = pd.DatetimeIndex(weather.index)
        if (len(index) != 8760 or index.hasnans or not index.is_unique
                or not np.all(index[1:] - index[:-1] == pd.Timedelta(hours=1))):
            raise ValueError("Stage 1 requires 8760 consecutive hourly weather rows")
        if not np.all(np.isfinite(weather.select_dtypes(include=[np.number]).values)):
            raise ValueError("Weather contains nonfinite values")

    def evaluate(
        self,
        clearance,
        sp,
        p_rated,
        Nwt,
        wind_MW_per_km2,
        solar_MW,
        surface_tilt,
        surface_azimuth,
        DC_AC_ratio,
        b_P,
        b_E_h,
    ):
        """Evaluate one fixed design with the original positional design inputs.

        Parameters
        ----------
        clearance : Blade tip clearance [m].
        sp : Wind turbine specific power [W/m2].
        p_rated : Turbine rated power [MW].
        Nwt : Number of turbines.
        wind_MW_per_km2 : Wind installation density [MW/km2].
        solar_MW : PV AC capacity [MW].
        surface_tilt, surface_azimuth : PV angles [degrees].
        DC_AC_ratio : PV DC/AC ratio.
        b_P : Battery reference power rating [MW].
        b_E_h : Nominal battery duration [h]; b_E = b_P * b_E_h.

        Returns
        -------
        outputs : Array containing LOLH, firm reliability and EENS.
            Hourly arrays are also in self.hourly and self.prob; SOC is
            self.soc_mwh, with N_time + 1 states. Solver details are in
            self.solver_status. Full-load reliability is constrained by the EMS
            and independently checked from hourly shortfalls afterwards.
        """
        self.inputs = [
            clearance,
            sp,
            p_rated,
            Nwt,
            wind_MW_per_km2,
            solar_MW,
            surface_tilt,
            surface_azimuth,
            DC_AC_ratio,
            b_P,
            b_E_h,
        ]
        if not np.all(np.isfinite(self.inputs)):
            raise ValueError("Design inputs must be finite")
        if min(sp, p_rated, wind_MW_per_km2, DC_AC_ratio) <= 0:
            raise ValueError("Specific power, turbine rating, density and DC/AC ratio must be positive")
        if min(clearance, Nwt, solar_MW, b_P, b_E_h) < 0 or int(Nwt) != Nwt:
            raise ValueError("Invalid capacity, clearance or turbine count")
        if not 0 <= surface_tilt <= 90:
            raise ValueError("PV tilt must be between 0 and 90 degrees")

        prob = self.prob
        d = get_rotor_d(p_rated * 1e6 / sp)
        hh = d / 2 + clearance
        wind_MW = Nwt * p_rated
        Awpp = wind_MW / wind_MW_per_km2
        b_E = b_E_h * b_P

        # Pass design variables, matching the original assembly.
        prob.set_val("hh", hh)
        prob.set_val("d", d)
        prob.set_val("p_rated", p_rated)
        prob.set_val("Nwt", Nwt)
        prob.set_val("Awpp", Awpp)
        prob.set_val("surface_tilt", surface_tilt)
        prob.set_val("surface_azimuth", surface_azimuth)
        prob.set_val("DC_AC_ratio", DC_AC_ratio)
        prob.set_val("solar_MW", solar_MW)
        prob.set_val("b_P", b_P)
        prob.set_val("b_E", b_E)

        # Clear previous results so a failed evaluation cannot expose stale ones.
        self.hourly = None
        self.soc_mwh = None
        self.solver_status = None
        self.metrics = None
        self.outputs = None
        prob.run_model()

        self.solver_status = prob.model.ems.ems_model.solver_status
        self.soc_mwh = prob.get_val("b_E_SOC_t").copy()
        facility_load_mw = prob.get_val("facility_load_mw").item()
        pue = prob.get_val("pue").item()
        self.hourly = {
            "wind_mw": prob.get_val("wind_t").copy(),
            "pv_mw": prob.get_val("solar_t").copy(),
            "renewable_available_mw": prob.get_val("wind_t") + prob.get_val("solar_t"),
            "facility_load_mw": np.full(self.N_time, facility_load_mw),
            "firm_demand_it_mw": np.full(self.N_time, facility_load_mw / pue),
            "served_facility_mw": prob.get_val("hpp_t").copy(),
            "unserved_facility_mw": prob.get_val("unserved_facility_mw").copy(),
            "served_firm_it_mw": prob.get_val("served_firm_it_mw").copy(),
            "unserved_firm_it_mw": prob.get_val("unserved_firm_it_mw").copy(),
            "charge_mw": prob.get_val("charge_mw").copy(),
            "discharge_mw": prob.get_val("discharge_mw").copy(),
            "curtailment_mw": prob.get_val("hpp_curt_t").copy(),
        }
        self.validation = validate_dispatch(
            self.hourly, self.soc_mwh, self.solver_status, b_P, b_E,
            prob.get_val("battery_depth_of_discharge").item(),
            prob.get_val("battery_charge_efficiency").item(),
            prob.get_val("battery_initial_soc").item(), pue,
        )
        self.metrics = firm_load_metrics(
            self.hourly["unserved_facility_mw"], epsilon_mw=self.sim_pars["epsilon_mw"]
        )
        if self.metrics["LOLH"] > self.solver_status["max_failed_hours"]:
            raise RuntimeError("Hourly results violate the reliability constraint")
        self.metrics["delivered_energy_mwh"] = float(self.hourly["served_facility_mw"].sum())
        self.metrics["ppa_revenue_eur"] = (
            prob.get_val("ppa_price").item() * self.metrics["delivered_energy_mwh"]
        )
        outputs = np.hstack([
            self.metrics["LOLH"], self.metrics["firm_reliability"], self.metrics["EENS"],
        ])
        self.outputs = outputs
        return outputs
