"""Run with python -m unittest hydesign.tests.test_simple_datacenter -v."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch, PropertyMock

import numpy as np
import openmdao.api as om
from docplex.mp.model import Model

from hydesign.ems.ems_simple_datacenter import (
    ems_simple_datacenter, ems_simple_datacenter_comp,
    ems_cplex_simple_datacenter, ems_cplex_constraints_simple_datacenter,
    solve_cplex_simple_datacenter, battery_power_fluctuation_expression,
)
from hydesign.metrics_simple_datacenter import firm_load_metrics, validate_dispatch
from hydesign.sizing_simple_datacenter import size_datacenter


class SimpleDatacenterTests(unittest.TestCase):
    def test_time_limit_candidate_is_explicitly_not_optimal(self):
        original_solve = Model.solve
        calls = []

        def report_time_limit(model, **kwargs):
            solution = original_solve(model, **kwargs)
            calls.append(True)
            if len(calls) == 1:
                details = model._solve_details
                details._solve_status_code = 107
                details._solve_status = "time limit exceeded"
                details._miprelgap = 0.05
                details._best_bound = 1.05 * solution.objective_value
            return solution

        with patch.object(Model, "solve", report_time_limit), self.assertWarns(RuntimeWarning):
            result = size_datacenter(
                [10, 10, 10, 2], [0, 0, 0, 0], 0, 0,
                reliability_target=0.75, facility_load_min_mw=0,
                facility_load_max_mw=20, allow_time_limit_incumbent=True,
            )
        self.assertFalse(result["solver_status"]["optimal"])
        self.assertTrue(result["solver_status"]["feasible"])
        self.assertEqual(result["solver_status"]["stages"][0]["relative_gap"], 0.05)
        self.assertEqual(result["metrics"]["full_load_factor"], 0.75)

    def test_time_limit_option_does_not_accept_missing_solution(self):
        with patch.object(Model, "solve", return_value=None):
            with self.assertRaises(RuntimeError):
                size_datacenter(
                    [10, 10], [0, 0], 0, 0,
                    allow_time_limit_incumbent=True,
                )

    def test_sizing_reliability_and_partial_hour_revenue(self):
        result = size_datacenter(
            [10, 10, 10, 2], [0, 0, 0, 0], 0, 0,
            reliability_target=0.75, facility_load_min_mw=0,
            facility_load_max_mw=20,
        )
        self.assertAlmostEqual(result["facility_load_mw"], 10, places=5)
        self.assertEqual(result["metrics"]["LOLH"], 1)
        self.assertEqual(result["metrics"]["full_load_factor"], 0.75)
        # The 2 MWh in the failed hour is paid too, not discarded.
        self.assertAlmostEqual(result["metrics"]["ppa_revenue_eur"], 32 * 70, places=3)
        # A smaller 2 MW contract has zero EENS, but is NOT the revenue optimum.
        self.assertAlmostEqual(result["metrics"]["EENS"], 8, places=5)
        self.assertFalse(result["upper_bound_active"])

    def test_stricter_reliability_changes_sized_load(self):
        result = size_datacenter(
            [10, 10, 10, 2], [0, 0, 0, 0], 0, 0,
            reliability_target=1.0, facility_load_min_mw=0,
            facility_load_max_mw=20,
        )
        self.assertAlmostEqual(result["facility_load_mw"], 2, places=5)
        self.assertEqual(result["metrics"]["LOLH"], 0)

    def test_95_percent_counts_hours_and_bounds_do_not_drive_result(self):
        generation = [10] * 19 + [2]
        for upper in (20, 100):
            result = size_datacenter(
                generation, np.zeros(20), 0, 0,
                reliability_target=0.95, facility_load_min_mw=0,
                facility_load_max_mw=upper,
            )
            self.assertAlmostEqual(result["facility_load_mw"], 10, places=5)
            self.assertEqual(result["solver_status"]["max_failed_hours"], 1)
            self.assertEqual(result["metrics"]["LOLH"], 1)

    def test_noninteger_failure_budget_rounds_down(self):
        result = size_datacenter(
            [10, 10, 2], [0, 0, 0], 0, 0,
            reliability_target=0.95, facility_load_min_mw=0,
            facility_load_max_mw=20,
        )
        self.assertEqual(result["solver_status"]["max_failed_hours"], 0)
        self.assertAlmostEqual(result["facility_load_mw"], 2, places=5)

    def test_infeasible_fixed_load_reliability_raises(self):
        model = ems_simple_datacenter(N_time=4)
        with self.assertRaisesRegex(RuntimeError, "optimality"):
            model.compute(
                np.array([10, 10, 10, 2]), np.zeros(4), 0, 0,
                0.9, np.sqrt(0.86), 0.5, 10, 1.15,
                reliability_target=1.0, ppa_price=70,
            )

    def test_longest_shortage_and_pue(self):
        metrics = firm_load_metrics([1, 1, 0, 1, 1, 1, 0])
        self.assertEqual(metrics["longest_shortage_hours"], 3)
        result = size_datacenter(
            [10, 10], [0, 0], 0, 0, pue=1.15,
            reliability_target=1.0, facility_load_min_mw=0,
            facility_load_max_mw=20,
        )
        self.assertAlmostEqual(result["firm_demand_it_mw"], 10 / 1.15, places=5)

    def solve(self, wind, b_P=25, b_E=200):
        wind = np.asarray(wind, dtype=float)
        model = ems_simple_datacenter(N_time=len(wind))
        values = model.compute(wind, np.zeros(len(wind)), b_P, b_E,
                               0.9, np.sqrt(0.86), 0.5, 16.0, 1.15)
        hpp, curt, net, soc, charge, discharge, served_it, unmet_it, unmet = values
        hourly = {
            "wind_mw": wind, "pv_mw": np.zeros(len(wind)),
            "renewable_available_mw": wind.copy(),
            "facility_load_mw": np.full(len(wind), 16.0),
            "firm_demand_it_mw": np.full(len(wind), 16.0 / 1.15),
            "served_facility_mw": hpp, "unserved_facility_mw": unmet,
            "served_firm_it_mw": served_it, "unserved_firm_it_mw": unmet_it,
            "charge_mw": charge, "discharge_mw": discharge, "curtailment_mw": curt,
        }
        self.validate(hourly, soc, model.solver_status, b_P, b_E)
        return hourly, soc, model.solver_status

    def validate(self, hourly, soc, status, b_P=25, b_E=200):
        return validate_dispatch(hourly, soc, status, b_P, b_E,
                                 0.9, np.sqrt(0.86), 0.5, 1.15)

    def test_hourly_metrics_are_not_energy_reliability(self):
        r = firm_load_metrics([0, 0.1, 11, 0.0005])
        self.assertEqual(r["LOLH"], 2)
        self.assertEqual(r["firm_reliability"], 0.5)
        self.assertAlmostEqual(r["EENS"], 11.1005)
        self.assertEqual(firm_load_metrics([0.001])["LOLH"], 0)

    def test_no_generation_cannot_borrow_initial_energy(self):
        h, soc, _ = self.solve([0, 0, 0])
        np.testing.assert_allclose(h["unserved_facility_mw"], 16, atol=1e-6)
        np.testing.assert_allclose(soc, 100, atol=1e-6)

    def test_surplus_supplies_facility_not_16mw_it(self):
        h, _, _ = self.solve([20, 20])
        np.testing.assert_allclose(h["served_facility_mw"], 16, atol=1e-6)
        np.testing.assert_allclose(h["served_firm_it_mw"], 16 / 1.15, atol=1e-6)
        np.testing.assert_allclose(h["curtailment_mw"], 4, atol=1e-6)
        np.testing.assert_allclose(h["charge_mw"], 0, atol=1e-6)

    def test_round_trip_and_final_transition(self):
        h, soc, _ = self.solve([32, 0])
        self.assertAlmostEqual(sum(h["unserved_facility_mw"]), 2.24, places=5)
        self.assertAlmostEqual(h["discharge_mw"][-1], 13.76, places=5)
        self.assertEqual(len(soc), 3)
        self.assertAlmostEqual(soc[-1], soc[0], places=6)

    def test_zero_battery(self):
        h, _, _ = self.solve([20, 5], b_P=0, b_E=0)
        np.testing.assert_allclose(h["unserved_facility_mw"], [0, 11], atol=1e-6)

    def test_power_and_energy_limits(self):
        h, soc, _ = self.solve([100, 0, 0, 100], b_P=1, b_E=2)
        self.assertLessEqual(max(h["discharge_mw"]), np.sqrt(0.86) + 1e-6)
        self.assertGreaterEqual(min(soc), 0.2 - 1e-6)
        self.assertLessEqual(max(soc), 2 + 1e-6)

    def test_validator_catches_broken_outputs(self):
        h, soc, status = self.solve([32, 0])
        with self.assertRaises(ValueError):
            self.validate(h, soc[:-1], status)
        for key in ("served_facility_mw", "unserved_facility_mw", "curtailment_mw"):
            bad = copy.deepcopy(h)
            bad[key][0] = -1
            with self.assertRaises(ValueError):
                self.validate(bad, soc, status)
        bad_soc = soc.copy()
        bad_soc[-1] += 1
        with self.assertRaises(ValueError):
            self.validate(h, bad_soc, status)
        with self.assertRaises(ValueError):
            self.validate(h, soc, {"optimal": False})

    def test_solver_failure_raises(self):
        with patch("hydesign.ems.ems_simple_datacenter.Model.solve", return_value=None):
            with self.assertRaises(RuntimeError):
                self.solve([16, 16])

    def test_cplex_time_limit_is_not_accepted(self):
        stopped = SimpleNamespace(status="time limit", status_code=11, time=1.0)
        with patch("hydesign.ems.ems_simple_datacenter.Model.solve", return_value=object()), patch(
            "hydesign.ems.ems_simple_datacenter.Model.solve_details",
            new_callable=PropertyMock, return_value=stopped,
        ):
            with self.assertRaisesRegex(RuntimeError, "time limit"):
                self.solve([16, 16])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.solve([float("nan")])
        with self.assertRaises(ValueError):
            self.solve([16], b_E=-1)
        with self.assertRaises(ValueError):
            firm_load_metrics([-1])

    def test_component_wrapper_outputs_match_model(self):
        expected, soc, _ = self.solve([32, 0])
        prob = om.Problem(reports=False)
        prob.model.add_subsystem("ems", ems_simple_datacenter_comp(N_time=2), promotes=["*"])
        prob.setup()
        for name, value in {
            "wind_t": [32, 0], "solar_t": [0, 0], "b_P": 25, "b_E": 200,
            "battery_depth_of_discharge": 0.9, "battery_charge_efficiency": np.sqrt(0.86),
            "battery_initial_soc": 0.5, "facility_load_mw": 16, "pue": 1.15,
        }.items():
            prob.set_val(name, value)
        prob.run_model()
        np.testing.assert_allclose(prob.get_val("ems.hpp_t"), expected["served_facility_mw"], atol=1e-6)
        np.testing.assert_allclose(prob.get_val("ems.b_E_SOC_t"), soc, atol=1e-6)
        self.assertTrue(prob.model.ems.ems_model.solver_status["optimal"])
        prob.cleanup()


class SingleStageObjectiveTests(unittest.TestCase):
    def test_one_solve_energy_objective_is_independent_of_ppa(self):
        original = Model.solve
        for price in (0, 70, 700):
            for weight in (0., 0.0001, 0.001):
                with self.subTest(price=price, weight=weight):
                    calls = []
                    def observe(mdl, **kwargs):
                        calls.append(True)
                        return original(mdl, **kwargs)
                    with patch.object(Model, "solve", observe):
                        values = ems_cplex_simple_datacenter(
                            [32, 0], [0, 0], 25, 200, .9, np.sqrt(.86), .5,
                            16, 1.15, ppa_price=price,
                            battery_power_fluctuation_penalty=weight,
                        )
                    status = values[-1]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(len(status["stages"]), 1)
                    self.assertAlmostEqual(sum(values[6]), 29.76 / 1.15, places=6)
                    self.assertAlmostEqual(status["battery_power_fluctuation_mw"], 29.76, places=6)
                    self.assertAlmostEqual(status["stages"][0]["objective"],
                                           sum(values[6]) - weight * 29.76, places=6)
                    self.assertLess(np.minimum(values[4], values[5]).max(), 1e-5)
                    np.testing.assert_allclose([values[3][0], values[3][-1]], [100, 100])
                    self.assertNotIn("revenue_lock_tolerance_eur", status)

    def test_linearization_uses_only_adjacent_net_power_and_no_new_binaries(self):
        for count in (1, 3):
            with Model() as mdl:
                ch = mdl.continuous_var_list(count, lb=0)
                dis = mdl.continuous_var_list(count, lb=0)
                variables = {"P_charge": ch, "P_discharge": dis}
                before = (mdl.number_of_variables, mdl.number_of_constraints)
                zero = battery_power_fluctuation_expression(mdl, variables, 0)
                self.assertEqual(before, (mdl.number_of_variables, mdl.number_of_constraints))
                ramp = battery_power_fluctuation_expression(mdl, variables, .001)
                self.assertEqual(mdl.number_of_binary_variables, 0)
                for v, value in zip(ch, [2, 0, 3][:count]):
                    v.lb = v.ub = value
                for v, value in zip(dis, [0, 1, 0][:count]):
                    v.lb = v.ub = value
                mdl.minimize(ramp)
                sol = mdl.solve()
                self.assertAlmostEqual(sol.get_value(ramp), 0 if count == 1 else 7)
                self.assertAlmostEqual(sol.get_value(zero), 0)

    def test_bad_penalties_are_rejected(self):
        for weight in (-.001, np.nan, np.inf):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "fluctuation"):
                ems_cplex_simple_datacenter([16], [0], 0, 0, .9, .9, .5, 16, 1.15,
                                            battery_power_fluctuation_penalty=weight)

    def test_positive_penalty_can_reduce_served_energy(self):
        # Deliberately low round-trip efficiency makes the trade-off observable
        # even at lambda=0.0001. This is a stress test, not the thesis assumption.
        energy = []
        for weight in (0, .0001, .001):
            result = ems_cplex_simple_datacenter(
                [17, 15.9999], [0, 0], 1, 200, .9, .01, .5, 16, 1.15,
                battery_power_fluctuation_penalty=weight,
            )
            energy.append(float(sum(result[6])))
        self.assertAlmostEqual(energy[0] - energy[1], .0001 / 1.15, places=8)
        self.assertAlmostEqual(energy[1], energy[2], places=8)

    def test_validator_still_rejects_feasible_simultaneous_operation(self):
        # Witness allowed by the existing physical formulation, at every weight.
        # Force it only in this test; production constraints remain unchanged.
        for weight in (0, .0001, .001):
            with Model() as mdl:
                variables = ems_cplex_constraints_simple_datacenter(
                    mdl, [20, 20], [0, 0], 25, 200, .9, np.sqrt(.86), .5,
                    16, 16, 1.15, 0,
                )
                for v in variables["P_charge"]:
                    v.lb = v.ub = 1
                for v in variables["P_discharge"]:
                    v.lb = v.ub = .86
                sol, status = solve_cplex_simple_datacenter(
                    mdl, variables, battery_power_fluctuation_penalty=weight)
                self.assertAlmostEqual(status["battery_power_fluctuation_mw"], 0)
                served = np.array(sol.get_values(variables["P_served_IT_t"]))
                hourly = {
                    "wind_mw": np.full(2, 20.), "pv_mw": np.zeros(2),
                    "renewable_available_mw": np.full(2, 20.),
                    "charge_mw": np.ones(2), "discharge_mw": np.full(2, .86),
                    "facility_load_mw": np.full(2, 16.),
                    "firm_demand_it_mw": np.full(2, 16/1.15),
                    "served_firm_it_mw": served, "unserved_firm_it_mw": 16/1.15-served,
                    "served_facility_mw": served*1.15, "unserved_facility_mw": 16-served*1.15,
                    "curtailment_mw": np.array(sol.get_values(variables["P_curtailment_t"])),
                }
                with self.assertRaisesRegex(ValueError, "Simultaneous"):
                    validate_dispatch(hourly, sol.get_values(variables["E_SOC_t"]),
                                      status, 25, 200, .9, np.sqrt(.86), .5, 1.15)


if __name__ == "__main__":
    unittest.main()
