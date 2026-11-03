"""Economic extension tests, using synthetic dispatch and bundled weather only."""

import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from docplex.mp.model import Model

from hydesign.assembly.hpp_assembly_simple_datacenter import hpp_model_simple_datacenter
from hydesign.assembly.hpp_assembly_simple_datacenter_with_dc_cost import (
    hpp_model_simple_datacenter_with_dc_cost,
)
from hydesign.costs.cost_simple_datacenter import datacenter_cost_annualized
from hydesign.ems.ems_simple_datacenter import ems_cplex_constraints_simple_datacenter
from hydesign.sizing_simple_datacenter import size_datacenter
from hydesign.sizing_simple_datacenter_with_dc_cost import (
    datacenter_economic_expressions,
    size_datacenter_with_dc_cost,
)
import hydesign.sizing_simple_datacenter_with_dc_cost as economic_sizing


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
YAML_PATH = EXAMPLES / "thesis_simple_datacenter/simple_datacenter.yml"
COST_KEYS = (
    "datacenter_facility_capex_per_mw_it", "datacenter_it_capex_per_mw_it",
    "datacenter_fixed_opex_per_mw_it_year", "datacenter_discount_rate",
    "datacenter_facility_lifetime_years", "datacenter_it_lifetime_years",
)
COST_OUTPUTS = (
    "CAPEX_dc_facility", "CAPEX_dc_it", "CAPEX_dc_total", "OPEX_dc",
    "annualized_CAPEX_dc_facility", "annualized_CAPEX_dc_it",
    "annualized_CAPEX_dc_total", "annualized_cost_dc",
)


def yaml_parameters():
    with YAML_PATH.open() as stream:
        return yaml.safe_load(stream)


class FixedEvaluationEconomicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = yaml_parameters()
        kwargs = dict(
            sim_pars_fn=str(YAML_PATH),
            input_ts_fn=str(EXAMPLES / "Europe/GWA2/input_ts_Denmark_good_solar.csv"),
            latitude=55.39776, longitude=11.813965, altitude=42,
            facility_load_mw=16, pue=1.15, reliability_target=0,
            ppa_price=cls.parameters["ppa_price"],
        )
        cls.original = hpp_model_simple_datacenter(**kwargs)
        cls.economic = hpp_model_simple_datacenter_with_dc_cost(**kwargs)
        cls.design = [35, 300, 5, 20, 7, 180, 42, 180, 1.25, 25, 8]
        cls.original_outputs = cls.original.evaluate(*cls.design)
        cls.economic_outputs = cls.economic.evaluate(*cls.design)

    @classmethod
    def tearDownClass(cls):
        cls.original.prob.cleanup()
        cls.economic.prob.cleanup()

    def test_technical_dispatch_and_reliability_unchanged(self):
        np.testing.assert_allclose(self.original_outputs, self.economic_outputs[:3], rtol=0, atol=1e-8)
        for name, values in self.original.hourly.items():
            np.testing.assert_allclose(values, self.economic.hourly[name], rtol=0, atol=1e-8, err_msg=name)
        np.testing.assert_allclose(self.original.soc_mwh, self.economic.soc_mwh, rtol=0, atol=1e-8)

    def test_installed_capacity_and_wrapper_costs(self):
        economics = self.economic.economics
        self.assertAlmostEqual(economics["it_capacity_mw"], 16 / 1.15)
        self.assertLess(self.economic.hourly["served_firm_it_mw"].mean(), economics["it_capacity_mw"])
        model = datacenter_cost_annualized(**{key: self.parameters[key] for key in COST_KEYS})
        expected = model.compute(16 / 1.15)
        np.testing.assert_allclose([economics[name] for name in COST_OUTPUTS], expected)
        np.testing.assert_allclose([self.economic.prob.get_val(name).item() for name in COST_OUTPUTS], expected)

    def test_served_it_revenue_and_net_value(self):
        economics = self.economic.economics
        energy = self.economic.hourly["served_firm_it_mw"].sum()
        rate = self.parameters["datacenter_firm_revenue_per_mwh_it"]
        self.assertEqual(economics["served_it_energy_mwh"], energy)
        self.assertEqual(economics["Revenue_dc"], rate * energy)
        self.assertAlmostEqual(economics["served_facility_energy_mwh"], 1.15 * energy)
        self.assertEqual(economics["net_annual_value"], economics["Revenue_dc"] - economics["annualized_cost_dc"])
        self.assertLess(economics["Revenue_dc"], rate * economics["it_capacity_mw"] * 8760)


class SizingEconomicsTests(unittest.TestCase):
    def setUp(self):
        self.parameters = yaml_parameters()
        self.cost = {key: self.parameters[key] for key in COST_KEYS}
        # Synthetic short-horizon economics make the optimum hand-checkable.
        self.toy = {
            **self.cost,
            "datacenter_facility_capex_per_mw_it": 0,
            "datacenter_it_capex_per_mw_it": 0,
            "datacenter_fixed_opex_per_mw_it_year": 350,
            "datacenter_firm_revenue_per_mwh_it": 100,
        }

    def solve_toy(self, **overrides):
        parameters = dict(
            wind_ts=[10, 10, 10, 2], solar_ts=[0, 0, 0, 0], b_P=0, b_E=0,
            pue=2, reliability_target=0.75, facility_load_min_mw=1,
            facility_load_max_mw=20, **self.toy,
        )
        parameters.update(overrides)
        return size_datacenter_with_dc_cost(**parameters)

    def test_yaml_economic_assumptions(self):
        self.assertEqual(self.parameters["datacenter_discount_rate"], 0.075)
        self.assertEqual(self.parameters["datacenter_firm_revenue_per_mwh_it"], 4000)
        self.assertEqual(self.parameters["datacenter_facility_lifetime_years"], 20)
        self.assertEqual(self.parameters["datacenter_it_lifetime_years"], 5)

    def test_linear_expressions_match_standalone_and_served_it_basis(self):
        model = datacenter_cost_annualized(**self.cost)
        rate = self.parameters["datacenter_firm_revenue_per_mwh_it"]
        with Model() as mdl:
            load = mdl.continuous_var(lb=16, ub=16)
            variables = ems_cplex_constraints_simple_datacenter(
                mdl, [16, 8], [0, 0], 0, 0, 0.9, np.sqrt(0.86), 0.5,
                load, 16, 2, 0,
            )
            before = (mdl.number_of_variables, mdl.number_of_constraints)
            economics = datacenter_economic_expressions(mdl, variables, load, 2, model, rate)
            self.assertEqual(before, (mdl.number_of_variables, mdl.number_of_constraints))
            mdl.maximize(economics["net_annual_value"])
            sol = mdl.solve()
            self.assertIsNotNone(sol)
            actual = {name: sol.get_value(value) for name, value in economics.items()}
            np.testing.assert_allclose([actual[name] for name in COST_OUTPUTS], model.compute(8))
            self.assertAlmostEqual(actual["served_it_energy_mwh"], 12)
            self.assertAlmostEqual(actual["Revenue_dc"], rate * 12)
            self.assertLess(actual["Revenue_dc"], rate * 8 * 2)
            self.assertAlmostEqual(actual["net_annual_value"], rate * 12 - model.compute(8)[-1])
            self.assertNotAlmostEqual(actual["net_annual_value"], rate * 12 - actual["CAPEX_dc_total"])
            r = self.cost["datacenter_discount_rate"]
            for n, raw, annual in ((20, "CAPEX_dc_facility", "annualized_CAPEX_dc_facility"),
                                   (5, "CAPEX_dc_it", "annualized_CAPEX_dc_it")):
                crf = r * (1 + r)**n / ((1 + r)**n - 1)
                np.testing.assert_allclose(actual[annual], crf * actual[raw])

    def test_net_objective_changes_size_from_energy_only(self):
        result = self.solve_toy()
        original = size_datacenter(
            [10, 10, 10, 2], [0, 0, 0, 0], 0, 0, pue=2,
            reliability_target=0.75, facility_load_min_mw=1, facility_load_max_mw=20,
        )
        self.assertAlmostEqual(result["facility_load_mw"], 2)
        self.assertAlmostEqual(original["facility_load_mw"], 10)
        self.assertAlmostEqual(result["economics"]["Revenue_dc"], 400)
        self.assertAlmostEqual(result["economics"]["annualized_cost_dc"], 350)
        self.assertAlmostEqual(result["solver_status"]["primary_net_annual_value"], 50)
        self.assertAlmostEqual(result["solver_status"]["stages"][0]["objective"], 50)
        self.assertNotIn("ppa_price", inspect.signature(size_datacenter_with_dc_cost).parameters)

    def test_zero_cost_reduces_to_served_it_revenue(self):
        result = self.solve_toy(datacenter_fixed_opex_per_mw_it_year=0)
        self.assertAlmostEqual(result["facility_load_mw"], 10)
        economics = result["economics"]
        self.assertAlmostEqual(economics["served_it_energy_mwh"], 16)
        self.assertAlmostEqual(economics["Revenue_dc"], 1600)
        self.assertEqual(economics["annualized_cost_dc"], 0)
        self.assertAlmostEqual(economics["net_annual_value"], economics["Revenue_dc"])
        self.assertEqual(result["metrics"]["LOLH"], 1)
        self.assertLess(economics["Revenue_dc"], 100 * economics["it_capacity_mw"] * 4)

    def test_zero_revenue_selects_lower_bound(self):
        for lower in (0, 1):
            with self.subTest(lower=lower):
                result = self.solve_toy(datacenter_firm_revenue_per_mwh_it=0, facility_load_min_mw=lower)
                self.assertAlmostEqual(result["facility_load_mw"], lower)
                self.assertEqual(result["economics"]["Revenue_dc"], 0)
                self.assertAlmostEqual(result["economics"]["net_annual_value"], -350 * lower / 2)

    def test_single_stage_preserves_economic_objective_without_locks(self):
        original_solve = Model.solve
        calls = []
        def inspect_solve(mdl, **kwargs):
            self.assertIsNone(mdl.get_constraint_by_name("preserve_primary_net_annual_value"))
            load = mdl.get_var_by_name("contracted_facility_load_mw")
            self.assertLess(load.lb, load.ub)
            self.assertAlmostEqual(mdl.objective_expr.get_coef(load), -350 / 2)
            calls.append(True)
            return original_solve(mdl, **kwargs)
        with patch.object(Model, "solve", inspect_solve):
            result = self.solve_toy()
        self.assertEqual(len(calls), 1)
        status = result["solver_status"]
        self.assertEqual(len(status["stages"]), 1)
        self.assertNotIn("net_lock_tolerance_eur", status)
        self.assertAlmostEqual(result["economics"]["net_annual_value"], 50)

    def test_fluctuation_penalty_has_eur_units_and_excludes_accounting(self):
        for weight in (0, 0.0001, 0.001):
            with self.subTest(weight=weight):
                result = self.solve_toy(
                    wind_ts=[32, 0], solar_ts=[0, 0], b_P=25, b_E=200,
                    pue=1.15, reliability_target=0.5, facility_load_min_mw=16,
                    facility_load_max_mw=16.1, battery_power_fluctuation_penalty=weight,
                )
                econ, status = result["economics"], result["solver_status"]
                fluctuation = float(np.abs(np.diff(result["b_t"])).sum())
                self.assertAlmostEqual(fluctuation, 29.76, places=5)
                self.assertAlmostEqual(status["battery_penalty_eur"], weight * 100 * fluctuation)
                self.assertAlmostEqual(econ["net_annual_value"], econ["Revenue_dc"] - econ["annualized_cost_dc"])
                self.assertAlmostEqual(status["stages"][0]["objective"],
                                       econ["net_annual_value"] - status["battery_penalty_eur"])
                self.assertAlmostEqual(result["metrics"]["EENS"], 2.24, places=5)

    def test_technical_constraints_and_physical_upper_bound_are_reused(self):
        self.assertIs(economic_sizing.ems_cplex_constraints_simple_datacenter,
                      ems_cplex_constraints_simple_datacenter)
        result = self.solve_toy()
        self.assertEqual(result["solver_status"]["max_failed_hours"], 1)
        self.assertEqual(result["physical_load_upper_mw"], 10)
        self.assertLessEqual(result["metrics"]["LOLH"], 1)

    def test_battery_physics_pass_shared_validation(self):
        result = self.solve_toy(
            wind_ts=[32, 0], solar_ts=[0, 0], b_P=25, b_E=200,
            pue=1.15, reliability_target=0.5, facility_load_min_mw=16,
            facility_load_max_mw=16.1,
        )
        self.assertAlmostEqual(result["facility_load_mw"], 16)
        self.assertAlmostEqual(result["metrics"]["EENS"], 2.24, places=5)
        self.assertAlmostEqual(result["soc_mwh"][0], 100)
        self.assertAlmostEqual(result["soc_mwh"][-1], 100)
        self.assertLess(max(result["validation"].values()), 1e-5)

    def test_time_limit_candidate_keeps_net_gap(self):
        original_solve = Model.solve
        calls = []

        def report_time_limit(mdl, **kwargs):
            solution = original_solve(mdl, **kwargs)
            calls.append(True)
            if len(calls) == 1:
                details = mdl._solve_details
                details._solve_status_code = 107
                details._solve_status = "time limit exceeded"
                details._miprelgap = 0.05
                details._best_bound = 1.05 * solution.objective_value
            return solution

        with patch.object(Model, "solve", report_time_limit), self.assertWarns(RuntimeWarning):
            result = self.solve_toy(allow_time_limit_incumbent=True)
        self.assertFalse(result["solver_status"]["optimal"])
        self.assertEqual(result["solver_status"]["stages"][0]["relative_gap"], 0.05)
        self.assertAlmostEqual(result["solver_status"]["primary_net_annual_value"], 50)

    def test_no_solution_is_never_accepted(self):
        with patch.object(Model, "solve", return_value=None):
            with self.assertRaises(RuntimeError):
                self.solve_toy(allow_time_limit_incumbent=True)

    def test_invalid_revenue_is_rejected(self):
        for rate in (None, -1, np.nan, np.inf):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "datacenter_firm_revenue"):
                self.solve_toy(datacenter_firm_revenue_per_mwh_it=rate)


if __name__ == "__main__":
    unittest.main()
