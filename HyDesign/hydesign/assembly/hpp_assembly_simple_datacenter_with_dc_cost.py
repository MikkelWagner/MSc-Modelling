import numpy as np
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.costs.cost_simple_datacenter import datacenter_cost_annualized_comp
from hydesign.ems.ems_simple_datacenter import ems_simple_datacenter_comp as ems
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch
from hydesign.pv.pv import pvp_comp as pvp
from hydesign.weather.weather import ABL_comp as ABL
from hydesign.wind.wind import genericWake_surrogate_comp as genericWake_surrogate
from hydesign.wind.wind import genericWT_surrogate_comp as genericWT_surrogate
from hydesign.wind.wind import get_rotor_d
from hydesign.wind.wind import wpp_comp as wpp


class hpp_model_simple_datacenter_with_dc_cost(hpp_base):
    """Fixed-load evaluation with DC economic reporting.

    Copy of the simple assembly: generation, EMS and technical checks are
    unchanged. The cost component uses installed IT capacity; revenue uses
    actually served IT energy. Economics do not feed back into this dispatch.
    """

    def __init__(self, sim_pars_fn, **kwargs):
        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        N_time = self.N_time
        N_ws = self.N_ws
        sim_pars = self.sim_pars
        input_ts_fn = sim_pars["input_ts_fn"]
        self.datacenter_firm_revenue_per_mwh_it = sim_pars["datacenter_firm_revenue_per_mwh_it"]
        rate = self.datacenter_firm_revenue_per_mwh_it
        if rate is None or not np.isfinite(rate) or rate < 0:
            raise ValueError("datacenter_firm_revenue_per_mwh_it must be finite and nonnegative")

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
            (
                "datacenter_cost",
                datacenter_cost_annualized_comp(
                    datacenter_facility_capex_per_mw_it=sim_pars["datacenter_facility_capex_per_mw_it"],
                    datacenter_it_capex_per_mw_it=sim_pars["datacenter_it_capex_per_mw_it"],
                    datacenter_fixed_opex_per_mw_it_year=sim_pars["datacenter_fixed_opex_per_mw_it_year"],
                    datacenter_discount_rate=sim_pars["datacenter_discount_rate"],
                    datacenter_facility_lifetime_years=sim_pars["datacenter_facility_lifetime_years"],
                    datacenter_it_lifetime_years=sim_pars["datacenter_it_lifetime_years"],
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
        self.cost_output_names = (
            "CAPEX_dc_facility", "CAPEX_dc_it", "CAPEX_dc_total", "OPEX_dc",
            "annualized_CAPEX_dc_facility", "annualized_CAPEX_dc_it",
            "annualized_CAPEX_dc_total", "annualized_cost_dc",
        )
        self.list_out_vars = ["LOLH [h]", "firm_reliability [-]", "EENS [MWh]"] + [
            "it_capacity_mw [MW_IT]", "served_it_energy_mwh [MWh_IT/year]",
            "Revenue_dc [EUR/year]",
        ] + [
            name + (" [EUR]" if name.startswith("CAPEX_") else " [EUR/year]")
            for name in self.cost_output_names
        ] + ["net_annual_value [EUR/year]"]

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
        outputs : Technical metrics followed by DC size, energy, revenue and costs.
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
        # Installed capacity is requested facility load / PUE, not average supply.
        prob.set_val(
            "it_capacity_mw",
            prob.get_val("facility_load_mw").item() / prob.get_val("pue").item(),
        )

        # Clear previous results so a failed evaluation cannot expose stale ones.
        self.hourly = None
        self.soc_mwh = None
        self.solver_status = None
        self.metrics = None
        self.outputs = None
        self.economics = None
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
        self.metrics["curtailment_mwh"] = float(self.hourly["curtailment_mw"].sum())
        self.metrics["battery_throughput_mwh"] = float(
            (self.hourly["charge_mw"] + self.hourly["discharge_mw"]).sum()
        )
        # dt = 1 hour for the validated 8760-hour profile. EUR/MWh_IT times
        # served MWh_IT/year gives EUR/year, comparable with annualized cost.
        served_it_energy_mwh = float(self.hourly["served_firm_it_mw"].sum())
        Revenue_dc = self.datacenter_firm_revenue_per_mwh_it * served_it_energy_mwh
        self.economics = {
            name: prob.get_val(name).item() for name in self.cost_output_names
        }
        self.economics.update({
            "facility_load_mw": facility_load_mw,
            "it_capacity_mw": prob.get_val("it_capacity_mw").item(),
            "served_it_energy_mwh": served_it_energy_mwh,
            "served_facility_energy_mwh": self.metrics["delivered_energy_mwh"],
            "datacenter_firm_revenue_per_mwh_it": self.datacenter_firm_revenue_per_mwh_it,
            "Revenue_dc": Revenue_dc,
            "net_annual_value": Revenue_dc - self.economics["annualized_cost_dc"],
        })
        # PPA is reporting-only. The artificial EMS fluctuation penalty is
        # likewise excluded from actual DC revenue and annualized cost accounting.
        outputs = np.hstack([
            self.metrics["LOLH"], self.metrics["firm_reliability"], self.metrics["EENS"],
            self.economics["it_capacity_mw"], served_it_energy_mwh, Revenue_dc,
            *[self.economics[name] for name in self.cost_output_names],
            self.economics["net_annual_value"],
        ])
        self.outputs = outputs
        return outputs
