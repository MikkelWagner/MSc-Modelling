"""Phase 1 tests; optional saved-design benchmark, with no EGO/assembly calls.

python -m unittest hydesign.tests.test_simple_datacenter_max_reliability -v
python -m hydesign.tests.test_simple_datacenter_max_reliability --benchmark
"""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from docplex.mp.model import Model

from hydesign.ems.ems_simple_datacenter_max_reliability import (
    calibrate_penalties, failed_hour_certificate,
    ems_cplex_constraints_simple_datacenter_max_reliability,
    ems_cplex_simple_datacenter_max_reliability as dispatch,
)
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch


class MaximumReliabilityTests(unittest.TestCase):
    def run_case(self, generation, **kwargs):
        args = dict(wind_ts=generation, solar_ts=np.zeros(len(generation)),
                    P_batt_MW=0, E_batt_MWh=0, battery_depth_of_discharge=.9,
                    charge_efficiency=np.sqrt(.86), battery_initial_soc=.5,
                    facility_load_mw=1, pue=1.15, threads=1, time_limit_seconds=10)
        args.update(kwargs)
        result = dispatch(**args)
        self.assertTrue(result[-1]["physically_validated"])
        self.assertLessEqual(result[-1]["secondary_objective"], .5 + 1e-7)
        return result

    def test_zero_generation(self):
        r = self.run_case([0, 0, 0])
        self.assertEqual(r[-1]["metrics"]["LOLH"], 3)
        self.assertTrue(r[-1]["certificate"]["maximum_availability_certified"])
        self.assertEqual(r[-1]["calibration"]["lambda_c"], 0)

    def test_zero_generation_cannot_borrow_initial_soc(self):
        r = self.run_case([0, 0, 0], P_batt_MW=2, E_batt_MWh=5)
        np.testing.assert_allclose(r[3], 2.5)
        self.assertAlmostEqual(r[-1]["battery_throughput_mwh"], 0)

    def test_no_battery_partial_supply(self):
        r = self.run_case([1, .5, 0, 2])
        self.assertEqual(r[-1]["binary_failed_hours"], 2)
        self.assertEqual(r[-1]["binary_lolh_mismatch_hours"], 0)
        self.assertAlmostEqual(r[-1]["metrics"]["EENS"], 1.5)
        np.testing.assert_allclose(r[0], [1, .5, 0, 1])

    def test_surplus_does_not_cycle_to_destroy_energy(self):
        r = self.run_case([10] * 8, P_batt_MW=2, E_batt_MWh=4)
        self.assertEqual(r[-1]["metrics"]["LOLH"], 0)
        self.assertAlmostEqual(r[-1]["battery_throughput_mwh"], 0, places=6)
        self.assertAlmostEqual(r[-1]["curtailment_mwh"], 72, places=6)

    def test_storage_efficiency_power_convention_and_boundaries(self):
        r = self.run_case([3, 0], P_batt_MW=2, E_batt_MWh=4)
        self.assertEqual(r[-1]["binary_failed_hours"], 0)
        self.assertAlmostEqual(r[4][0], 1 / .86, places=6)
        self.assertAlmostEqual(r[5][1], 1, places=6)
        self.assertAlmostEqual(r[3][0], r[3][-1], places=7)

    def test_zero_load(self):
        r = self.run_case([0, 2], facility_load_mw=0)
        self.assertEqual(r[-1]["binary_failed_hours"], 0)

    def test_scarce_storage_serves_whole_hours(self):
        r = self.run_case([2.5, 0, 0], P_batt_MW=2, E_batt_MWh=2.5,
                          battery_depth_of_discharge=1, charge_efficiency=1)
        self.assertEqual(r[-1]["metrics"]["LOLH"], 1)
        self.assertEqual(r[-1]["binary_lolh_mismatch_hours"], 0)
        self.assertTrue(r[-1]["certificate"]["maximum_availability_certified"])
        self.assertEqual(np.count_nonzero(r[0] >= 1 - 1e-7), 2)

    def test_exact_binary_vs_reporting_tolerance_is_explicit(self):
        r = self.run_case([.9995])
        self.assertEqual(r[-1]["binary_failed_hours"], 1)
        self.assertEqual(r[-1]["metrics"]["LOLH"], 0)
        self.assertEqual(r[-1]["binary_lolh_mismatch_hours"], 1)
        self.assertIn("not epsilon", r[-1]["certificate"]["certificate_scope"])

    def test_outer_target_is_not_an_ems_argument(self):
        with self.assertRaises(TypeError):
            self.run_case([1, 0], reliability_target=.99)
        a, b = self.run_case([1, .5]), self.run_case([1, .5])
        np.testing.assert_allclose(a[0], b[0])

    def test_only_failure_binaries_and_no_reliability_cap(self):
        with Model() as mdl:
            v = ems_cplex_constraints_simple_datacenter_max_reliability(
                mdl, [0, 0], [0, 0], 1, 2, .9, np.sqrt(.86), .5, 1, 1.15)
            self.assertEqual(mdl.number_of_binary_variables, 2)
            self.assertNotIn("max_failed_hours", v)
            self.assertEqual(mdl.number_of_constraints, 2 + 4 * 2)

    def test_penalty_dominance_and_calibration(self):
        c = calibrate_penalties([3, 4], 2, np.sqrt(.86))
        self.assertAlmostEqual(c["lambda_b"] / c["lambda_c"], .1)
        self.assertAlmostEqual(c["lambda_c"] * c["curtailment_bound_mwh"]
            + c["lambda_b"] * c["throughput_bound_mwh"], .25)
        for invalid in [0, 1, -1, float("nan")]:
            with self.assertRaises(ValueError):
                calibrate_penalties([1], 2, .9, beta=invalid)
        with self.assertRaises(ValueError):
            calibrate_penalties([1], 2, np.sqrt(.86), kappa=.07)

    def test_scaled_solve_has_the_requested_objective(self):
        r = self.run_case([3, 0, .2], P_batt_MW=1, E_batt_MWh=1)
        status = r[-1]
        self.assertAlmostEqual(status["stages"][0]["objective"],
                               status["binary_failed_hours"] + status["secondary_objective"])

    def test_count_certificate_does_not_require_secondary_convergence(self):
        c = failed_hour_certificate(4.30, 5, .25, 20)
        self.assertTrue(c["maximum_availability_certified"])
        self.assertEqual(c["failed_hours_lower_bound"], 5)
        self.assertFalse(failed_hour_certificate(4.20, 5, .25, 20)["maximum_availability_certified"])
        self.assertEqual(failed_hour_certificate(4.250000001, 5, .25, 20)["failed_hours_lower_bound"], 4)

    def test_synthetic_chronology_certifies_nine_failed_hours(self):
        generation = np.random.default_rng(0).uniform(0, 1.6, 48)
        r = self.run_case(generation, P_batt_MW=1, E_batt_MWh=3)
        self.assertEqual(r[-1]["binary_failed_hours"], 9)
        self.assertTrue(r[-1]["certificate"]["maximum_availability_certified"])
        self.assertIsNotNone(r[-1]["first_availability_certificate"])

    def test_time_limit_incumbent_is_not_called_optimal(self):
        original = Model.solve
        def time_limit(model, **kwargs):
            sol = original(model, **kwargs)
            model._solve_details._solve_status_code = 107
            model._solve_details._solve_status = "time limit exceeded"
            return sol
        with patch.object(Model, "solve", time_limit):
            r = self.run_case([1, 0])
        self.assertFalse(r[-1]["optimal"])
        self.assertEqual(r[-1]["termination"], "TIME_LIMIT_INCUMBENT")
        self.assertTrue(r[-1]["certificate"]["maximum_availability_certified"])

    def test_no_incumbent_is_not_accepted(self):
        with patch.object(Model, "solve", return_value=None):
            with self.assertRaises(RuntimeError):
                self.run_case([0, 1])

    def test_invalid_physics_is_not_silently_returned(self):
        with patch("hydesign.ems.ems_simple_datacenter_max_reliability.validate_dispatch",
                   side_effect=ValueError("Simultaneous battery charging/discharging")):
            with self.assertRaisesRegex(ValueError, "Simultaneous"):
                self.run_case([1, 0])

    def test_penalty_sensitivity_on_storage(self):
        for beta, kappa in [(.1, .1), (.25, .1), (.5, .1), (.25, .08), (.25, .2)]:
            r = self.run_case([3, 0, 3, 0], P_batt_MW=2, E_batt_MWh=4,
                              beta=beta, kappa=kappa)
            self.assertEqual(r[-1]["metrics"]["LOLH"], 0)
            self.assertAlmostEqual(r[-1]["battery_throughput_mwh"], 2 + 2 / .86)


def benchmark_saved_designs(time_limit=30, case_names=None, include_sensitivity=True, threads=1):
    """Reproducible, opt-in Phase 1 comparison; saves new timestamped artifacts.

    Reads saved wind/PV arrays directly: no generation, cost, assembly or outer
    optimization is rerun. Original solver helpers are imported unchanged to
    expose its binary counts and permit labelled time-limit incumbents.
    """
    from datetime import datetime, timezone
    import hashlib
    import time
    import uuid
    import pandas as pd
    from hydesign.ems.ems_simple_datacenter import (
        ems_cplex_constraints_simple_datacenter, solve_cplex_simple_datacenter,
        ems_cplex_results_simple_datacenter,
    )

    root = Path(__file__).resolve().parents[2]
    saved = root / "hpp_sizing_runs/lcoe_reliability_frontier"
    out = root / "hpp_sizing_runs" / ("phase1_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                                       + "_" + uuid.uuid4().hex[:6] + "_max_reliability")
    out.mkdir()
    rows = []
    sources = [
        ("shared80", "experiment_20261105_131955Z_5fe42393", "080"),
        ("shared90", "experiment_20261105_131955Z_5fe42393", "090"),
        ("low70", "experiment_20261105_231423Z_85c2d00e", "070"),
        ("mid90", "experiment_20261105_231423Z_85c2d00e", "090"),
        ("full100", "experiment_20261105_231423Z_85c2d00e", "100"),
    ]
    for label, experiment, target in sources:
        if case_names is not None and label not in case_names:
            continue
        folder = saved / experiment / ("reliability_" + target)
        source = json.loads((folder / "result.json").read_text())
        config = source["configuration"]
        hfile = next(folder.glob("*/hourly_hpp_sizing_base.csv"))
        h = pd.read_csv(hfile)
        design = source["best"]["design"]
        params = dict(wind_ts=h.wind_mw.to_numpy(), solar_ts=h.pv_mw.to_numpy(),
                      P_batt_MW=design[9], E_batt_MWh=design[9] * design[10],
                      battery_depth_of_discharge=config["battery_depth_of_discharge"],
                      charge_efficiency=config["battery_charge_efficiency"],
                      battery_initial_soc=config["battery_initial_soc"],
                      facility_load_mw=config["facility_load_mw"], pue=config["pue"])
        variants = [("original", None, None, 1e-4), ("availability", .25, .10, 1e-4)]
        if label == "shared80" and include_sensitivity:
            variants += [("beta010", .10, .10, 1e-4), ("beta050", .50, .10, 1e-4),
                         ("kappa008", .25, .08, 1e-4), ("kappa020", .25, .20, 1e-4),
                         ("gap_zero", .25, .10, 0), ("gap_loose", .25, .10, 1e-3)]
        for variant, beta, kappa, gap in variants:
            name = label + "_" + variant
            started = time.perf_counter()
            row = dict(case=label, variant=variant, source_target=config["reliability_target"],
                       beta=beta, kappa=kappa, mip_relative_gap_tolerance=gap,
                       time_limit_seconds=time_limit, threads=threads, random_seed=0,
                       Nwt=design[3], pv_mw=design[5], battery_power_mw=design[9],
                       battery_energy_mwh=design[9]*design[10],
                       source=str(folder), source_hourly_sha256=hashlib.sha256(hfile.read_bytes()).hexdigest(),
                       generation_sha256=hashlib.sha256(h[["wind_mw", "pv_mw"]].to_numpy().tobytes()).hexdigest())
            print("START", name, flush=True)
            try:
                if variant == "original":
                    with Model(name="original_ems_phase1_comparison") as mdl:
                        v = ems_cplex_constraints_simple_datacenter(
                            mdl, **params, facility_load_upper_mw=params["facility_load_mw"],
                            reliability_target=config["reliability_target"])
                        mdl.parameters.randomseed = 0
                        sol, status = solve_cplex_simple_datacenter(
                            mdl, v, time_limit_seconds=time_limit, threads=threads,
                            allow_time_limit_incumbent=True, battery_power_fluctuation_penalty=.0001)
                        indicators = np.rint(sol.get_values(v["failed_hour"])).astype(int)
                        status["failed_hour_binary"] = indicators.tolist()
                        status["binary_failed_hours"] = int(indicators.sum())
                        r = ems_cplex_results_simple_datacenter(sol, v, status, params["pue"])
                else:
                    r = dispatch(**params, time_limit_seconds=time_limit, threads=threads,
                                 beta=beta, kappa=kappa, mip_relative_gap=gap)
                served, curt, net, soc, ch, dis, it, uit, unmet, status = r
                hourly = h.copy()
                for column, array in dict(served_facility_mw=served, curtailment_mw=curt,
                    charge_mw=ch, discharge_mw=dis, served_firm_it_mw=it,
                    unserved_firm_it_mw=uit, unserved_facility_mw=unmet).items():
                    hourly[column] = array
                residual = validate_dispatch(hourly, soc, status, params["P_batt_MW"],
                    params["E_batt_MWh"], params["battery_depth_of_discharge"],
                    params["charge_efficiency"], params["battery_initial_soc"], params["pue"])
                metric = firm_load_metrics(unmet)
                stage = status["stages"][0]
                row.update(metric, physical_validation=True,
                    binary_failed_hours=status["binary_failed_hours"],
                    binary_lolh_difference=status["binary_failed_hours"]-metric["LOLH"],
                    delivered_energy_mwh=float(served.sum()), curtailment_mwh=float(curt.sum()),
                    battery_throughput_mwh=float((ch+dis).sum()),
                    maximum_simultaneous_power_mw=float(np.minimum(ch,dis).max()),
                    maximum_balance_residual=max(residual.values()), cplex_seconds=stage["seconds"],
                    relative_gap=stage.get("relative_gap"), solver_status=stage["status"],
                    weighted_objective_optimal_within_tolerances=status["optimal"])
                row.update(status.get("certificate", {}))
                row["first_availability_certificate"] = status.get("first_availability_certificate")
                (out / (name + "_status_max_reliability.json")).write_text(json.dumps(status, indent=2))
                hourly.to_csv(out / (name + "_hourly_max_reliability.csv"), index=False)
                np.save(out / (name + "_soc_max_reliability.npy"), soc)
            except Exception as exc:
                row.update(physical_validation=False, error=repr(exc))
            row["wall_seconds"] = time.perf_counter() - started
            rows.append(row)
            (out / "results_max_reliability.json").write_text(json.dumps(rows, indent=2))
            pd.DataFrame(rows).to_csv(out / "summary_max_reliability.csv", index=False)
            print(json.dumps(row), flush=True)
    print("OUTPUT", out, flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        benchmark_saved_designs()
    else:
        unittest.main()
