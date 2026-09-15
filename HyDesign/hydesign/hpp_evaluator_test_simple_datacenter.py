"""Evaluate wind/PV/battery designs for the simple off-grid load test."""

from pathlib import Path
from collections import Counter
from time import perf_counter

import numpy as np
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_model as base_hpp_model
from hydesign.ems_test_simple_datacenter import OffgridReliabilityEMS
from hydesign.examples import examples_filepath


class SimpleDatacenterHPP:
    """Adapter combining the unchanged HPP resource/cost model with the new EMS."""

    list_vars = [
        "clearance [m]",
        "sp [W/m2]",
        "p_rated [MW]",
        "Nwt",
        "wind_MW_per_km2 [MW/km2]",
        "solar_MW [MW]",
        "surface_tilt [deg]",
        "surface_azimuth [deg]",
        "DC_AC_ratio",
        "b_P [MW]",
        "b_E_h [h]",
        "cost_of_battery_P_fluct_in_peak_price_ratio",
    ]
    list_out_vars = [
        "CAPEX [MEuro]",
        "EENS [MWh/year]",
        "Firm reliability [-]",
        "Energy served [MWh/year]",
        "Reliability target [-]",
        "Wind capacity [MW]",
        "PV capacity [MW]",
        "Battery power [MW]",
        "Battery energy [MWh]",
        "Loss-of-load hours [h/year]",
        "Maximum shortfall [MW]",
        "Battery throughput [MWh/year]",
    ]

    def __init__(
        self,
        sim_pars_fn,
        input_ts_fn,
        reliability_target=0.90,
        load_mw=10.0,
        work_dir="./",
        epsilon_mw=1e-3,
        ems_time_limit_s=None,
        **kwargs,
    ):
        self.sim_pars_fn = str(sim_pars_fn)
        self.input_ts_fn = str(input_ts_fn)
        self.reliability_target = float(reliability_target)
        self.load_mw = float(load_mw)
        self.work_dir = str(work_dir)
        self.kwargs = kwargs
        self.epsilon_mw = epsilon_mw
        self.ems_time_limit_s = ems_time_limit_s
        self.solve_counts = Counter()
        self.solver_status_counts = Counter()
        self.primary_only_count = 0
        self.simultaneous_candidates = 0
        self.max_simultaneous_mw = 0.0
        self.cache = {}
        weather = pd.read_csv(self.input_ts_fn, index_col=0, parse_dates=True)
        if len(weather) != 8760 or not np.all(
            weather.index[1:] - weather.index[:-1] == pd.Timedelta(hours=1)
        ):
            raise ValueError("This annual test requires 8760 consecutive hourly observations")

    def _site_inputs(self):
        sites = pd.read_csv(
            Path(examples_filepath) / "examples_sites.csv", index_col=0, sep=";"
        )
        site = sites.loc[sites["input_ts_fn"] == self._relative_input_ts()]
        if site.empty:
            site = sites.loc[sites["sim_pars_fn"] == self._relative_sim_pars()]
        if site.empty:
            raise ValueError("Could not match input files to an HyDesign example site")
        return site.iloc[0]

    def _relative_input_ts(self):
        return str(Path(self.input_ts_fn).resolve()).replace(
            str(Path(examples_filepath).resolve()) + "\\", ""
        ).replace("\\", "/")

    def _relative_sim_pars(self):
        return str(Path(self.sim_pars_fn).resolve()).replace(
            str(Path(examples_filepath).resolve()) + "\\", ""
        ).replace("\\", "/")

    def evaluate(self, *x):
        key = tuple(map(float, x))
        if key in self.cache:
            record = self.cache[key]
            self.outputs, self.operation, self.objective, self.inputs = record
            return self.outputs.copy()
        started = perf_counter()
        values = dict(zip(self.list_vars, map(float, x)))
        site = self._site_inputs()
        base = base_hpp_model(
            latitude=float(site["latitude"]),
            longitude=float(site["longitude"]),
            altitude=float(site["altitude"]),
            max_num_batteries_allowed=10,
            work_dir=self.work_dir,
            sim_pars_fn=self.sim_pars_fn,
            input_ts_fn=self.input_ts_fn,
            G_MW=0.0,
            weeks_per_season_per_year=None,
            save_finance_ts=False,
            verbose=False,
        )
        base.evaluate(*[values[key] for key in self.list_vars])

        wind_mw = base.prob.get_val("wind_t").reshape(-1)
        solar_mw = base.prob.get_val("solar_t").reshape(-1)
        if len(wind_mw) != 8760 or len(solar_mw) != 8760:
            raise ValueError("The annual EMS requires full-year generation profiles")
        # Resource components can produce negligible negative numerical roundoff.
        if np.min(wind_mw) < -1e-7 or np.min(solar_mw) < -1e-7:
            raise ValueError("HyDesign returned negative generation")
        wind_mw = np.maximum(wind_mw, 0.0)
        solar_mw = np.maximum(solar_mw, 0.0)
        battery_power_mw = values["b_P [MW]"]
        battery_energy_mwh = battery_power_mw * values["b_E_h [h]"]
        ems = OffgridReliabilityEMS(
            load_mw=self.load_mw,
            reliability_target=self.reliability_target,
            epsilon_mw=self.epsilon_mw,
            time_limit_s=self.ems_time_limit_s,
        )
        operation = ems.solve(
            wind_mw=wind_mw,
            solar_mw=solar_mw,
            battery_power_mw=battery_power_mw,
            battery_energy_mwh=battery_energy_mwh,
            battery_depth_of_discharge=float(
                base.prob.get_val("battery_depth_of_discharge")[0]
            ),
            charge_efficiency=float(
                base.prob.get_val("battery_charge_efficiency")[0]
            ),
        )

        capex_eur = sum(
            float(base.prob.get_val(name)[0])
            for name in ("CAPEX_w", "CAPEX_s", "CAPEX_b", "CAPEX_sh")
        )
        capex_meuro = capex_eur / 1e6
        feasible = operation["status"] == "feasible" and operation["lolh"] <= ems.outage_budget(8760)
        self.objective = capex_meuro if feasible else np.inf
        self.solve_counts[operation["status"]] += 1
        self.solver_status_counts[operation["solver_status"]] += 1
        self.primary_only_count += int(operation["eens_optimal"] and not operation["throughput_optimal"])
        if feasible:
            self.simultaneous_candidates += int(operation["simultaneous_hours"] > 0)
            self.max_simultaneous_mw = max(self.max_simultaneous_mw,
                                           operation["max_simultaneous_mw"])
        operation["evaluation_runtime_s"] = perf_counter() - started

        self.prob = base.prob
        self.operation = operation
        self.inputs = values
        self.outputs = np.array(
            [
                capex_meuro,
                operation["eens_mwh"],
                operation["reliability"],
                operation["energy_served_mwh"],
                self.reliability_target,
                values["p_rated [MW]"] * values["Nwt"],
                values["solar_MW [MW]"],
                battery_power_mw,
                battery_energy_mwh,
                operation["lolh"],
                operation["maximum_shortfall_mw"],
                operation["throughput_mwh"],
            ]
        )
        self.cache[key] = (self.outputs.copy(), operation, self.objective, values)
        return self.outputs
