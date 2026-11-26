"""Lifetime finance tests: no EMS, CPLEX or EGO evaluations."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import openmdao.api as om

from hydesign.finance.finance_simple_datacenter import (
    FINANCE_PARAMETERS, INPUT_NAMES, OUTPUT_NAMES, finance_simple_datacenter,
    finance_simple_datacenter_comp, compare_native_hpp_finance,
)
from hydesign.sizing_native_hpp_sizing_base import read_configuration


CONFIG = Path(__file__).resolve().parents[1] / "examples/thesis_simple_datacenter/simple_datacenter.yml"


class FinanceTests(unittest.TestCase):
    def setUp(self):
        self.config = read_configuration(CONFIG)
        self.inputs = dict.fromkeys(INPUT_NAMES, 0.0)
        self.inputs.update(CAPEX_w=100.0, served_it_energy_mwh=20.0,
                           datacenter_firm_revenue_per_mwh_it=1.0)

    def calculate(self, config=None, inputs=None):
        cfg = self.config if config is None else config
        model = finance_simple_datacenter(**{k: cfg[k] for k in FINANCE_PARAMETERS})
        return dict(zip(OUTPUT_NAMES, model.compute(**(self.inputs if inputs is None else inputs))))

    def test_yaml_keeps_physical_year_separate(self):
        self.assertEqual(self.config["N_life"], 1)
        self.assertEqual(self.config["hpp_economic_lifetime_years"], 25)
        self.assertEqual(self.config["datacenter_facility_lifetime_years"], 25)
        self.assertEqual(self.config["project_discount_rate"], .07)

    def test_manual_cashflow_npv_and_ratio(self):
        out = self.calculate()
        years = self.config["hpp_economic_lifetime_years"]
        expected = -100 + sum(20 / 1.07**y for y in range(1, years + 1))
        self.assertAlmostEqual(out["NPV_system"], expected)
        self.assertAlmostEqual(out["NPV_over_CAPEX_system"], expected / 100)
        np.testing.assert_allclose(out["cashflow_vec"], [-100] + [20] * years)
        self.assertAlmostEqual(out["cumulative_discounted_cashflow_vec"][-1], expected)

    def test_discounted_lifetime_capex_reuses_actual_investments(self):
        inputs=dict(self.inputs,CAPEX_b=200,CAPEX_dc_it=300)
        out=self.calculate(inputs=inputs)
        discount=1.07**np.arange(26)
        expected=600+200/1.07**15+sum(300/1.07**y for y in (5,10,15,20))
        self.assertAlmostEqual(out['CAPEX_PV_system'],expected)
        self.assertAlmostEqual(out['CAPEX_lifetime_system'],600+200+4*300)
        self.assertAlmostEqual(out['CAPEX_replacement_PV_system'],expected-600)
        self.assertAlmostEqual(out['NPV_over_PV_CAPEX_system'],out['NPV_system']/expected)
        self.assertAlmostEqual(out['NPV_over_CAPEX_system'],out['NPV_system']/600)
        self.assertEqual(out['CAPEX_system_vec'][-1],0)
        np.testing.assert_allclose(out['CAPEX_PV_system'],np.sum(out['CAPEX_system_vec']/discount))

    def test_discounted_capex_phasing_inflation_and_no_replacements(self):
        simple=self.calculate()
        self.assertEqual(simple['CAPEX_PV_system'],simple['CAPEX_system'])
        self.assertEqual(simple['CAPEX_replacement_PV_system'],0)
        cfg=dict(self.config,phasing_yr=[0,2],phasing_CAPEX=[.25,.75],inflation=[.02,.02])
        out=self.calculate(cfg)
        self.assertAlmostEqual(out['CAPEX_PV_system'],25+75*1.02**2/1.07**2)
        cashflow=out['Revenue_dc_vec']-out['OPEX_system_vec']-out['taxes_vec']-out['CAPEX_system_vec']+out['residual_value_vec']
        np.testing.assert_array_equal(out['cashflow_vec'],cashflow)
        self.assertAlmostEqual(out['NPV_system'],np.sum(cashflow/1.07**np.arange(26)))
        zero=self.calculate(inputs=dict.fromkeys(INPUT_NAMES,0.))
        self.assertEqual(zero['CAPEX_PV_system'],0)
        self.assertTrue(np.isnan(zero['NPV_over_PV_CAPEX_system']))

    def test_replacements_and_terminal_boundary(self):
        inputs = dict(self.inputs, CAPEX_b=200, CAPEX_dc_it=300)
        out = self.calculate(inputs=inputs)
        np.testing.assert_array_equal(np.flatnonzero(out["replacement_battery_vec"]), [15])
        np.testing.assert_array_equal(np.flatnonzero(out["replacement_it_vec"]), [5, 10, 15, 20])
        self.assertEqual(out["replacement_battery_vec"][15], 200)
        np.testing.assert_allclose(out["replacement_it_vec"][[5, 10, 15, 20]], 300)
        self.assertEqual(out["CAPEX_system_vec"][-1], 0)
        expected = -600 + sum(20 / 1.07**y for y in range(1, 26)) - 200 / 1.07**15
        expected -= sum(300 / 1.07**y for y in (5, 10, 15, 20))
        self.assertAlmostEqual(out["NPV_system"], expected)
        self.assertEqual(out["CAPEX_system"], 600)  # Replacements excluded from denominator.

    def test_replacement_cost_fractions(self):
        cfg = dict(self.config, battery_replacement_cost_fraction=.5, datacenter_it_replacement_cost_fraction=.75)
        out = self.calculate(cfg, dict(self.inputs, CAPEX_b=200, CAPEX_dc_it=100))
        self.assertEqual(out["replacement_battery_vec"][15], 100)
        self.assertEqual(out["replacement_it_vec"][5], 75)

    def test_battery_purchase_at_project_end_is_excluded(self):
        cfg = dict(self.config, battery_replacement_year=25, battery_life_y=25)
        out = self.calculate(cfg, dict(self.inputs, CAPEX_b=200))
        self.assertEqual(out["replacement_battery_vec"].sum(), 0)

    def test_five_year_project_has_no_it_replacement(self):
        cfg = dict(self.config, hpp_economic_lifetime_years=5)
        out = self.calculate(cfg, dict(self.inputs, CAPEX_dc_it=300))
        self.assertEqual(out["replacement_it_vec"].sum(), 0)

    def test_negative_npv_does_not_zero_positive_irr(self):
        cfg = dict(self.config, hpp_economic_lifetime_years=1, project_discount_rate=.2)
        out = self.calculate(cfg, dict(self.inputs, served_it_energy_mwh=110))
        self.assertAlmostEqual(out["NPV_system"], -100 + 110 / 1.2)
        self.assertAlmostEqual(out["IRR_system"], .1)

    def test_undefined_irr_and_payback_are_nan(self):
        out = self.calculate(inputs=dict(self.inputs, served_it_energy_mwh=0))
        self.assertTrue(np.isnan(out["IRR_system"]))
        self.assertTrue(np.isnan(out["payback_years"]))
        self.assertTrue(np.isnan(out["discounted_payback_years"]))

    def test_zero_discount_and_payback(self):
        out = self.calculate(dict(self.config, project_discount_rate=0))
        self.assertEqual(out["NPV_system"], 400)
        self.assertEqual(out["payback_years"], 5)
        self.assertEqual(out["discounted_payback_years"], 5)

    def test_zero_investment_ratio_is_undefined(self):
        out = self.calculate(inputs=dict(self.inputs, CAPEX_w=0))
        self.assertTrue(np.isnan(out["NPV_over_CAPEX_system"]))

    def test_native_finance_parity_hpp_only(self):
        out = compare_native_hpp_finance(self.inputs, self.config)
        self.assertAlmostEqual(out["difference"], 0)

    def test_native_phased_tax_policy_and_replacement_depreciation(self):
        cfg = dict(self.config, hpp_economic_lifetime_years=6, tax_rate=.22,
                   datacenter_it_replacement_cost_fraction=.5)
        inputs = dict(self.inputs, CAPEX_w=0, CAPEX_dc_it=100, served_it_energy_mwh=30)
        out = self.calculate(cfg, inputs)
        np.testing.assert_allclose(out["depreciation_vec"], [0, 20, 20, 20, 20, 20, 10])
        np.testing.assert_allclose(out["taxes_vec"], [0, 2.2, 2.2, 2.2, 2.2, 2.2, 4.4])
        self.assertAlmostEqual(out["NPV_system"], sum(
            cf / 1.07**y for y, cf in enumerate(out["cashflow_vec"])))
        loss = self.calculate(cfg, dict(inputs, served_it_energy_mwh=5))
        np.testing.assert_array_equal(loss["taxes_vec"], 0)

    def test_inflation_applied_once_to_operation_and_replacements(self):
        cfg = dict(self.config, hpp_economic_lifetime_years=6, inflation=[.1, .1])
        out = self.calculate(cfg, dict(self.inputs, CAPEX_dc_it=100, OPEX_dc=10))
        self.assertAlmostEqual(out["Revenue_dc_vec"][2], 20 * 1.1**2)
        self.assertAlmostEqual(out["OPEX_system_vec"][2], 10 * 1.1**2)
        self.assertAlmostEqual(out["replacement_it_vec"][5], 100 * 1.1**5)
        self.assertAlmostEqual(out["depreciation_vec"][6], 100 / 25 + 100 * 1.1**5 / 5)
        self.assertAlmostEqual(out["NPV_system"], out["discounted_cashflow_vec"].sum())

    def test_net_terminal_value_and_negative_irr(self):
        cfg = dict(self.config, hpp_economic_lifetime_years=1, residual_value_fraction=.5)
        out = self.calculate(cfg, dict(self.inputs, served_it_energy_mwh=0))
        np.testing.assert_allclose(out["cashflow_vec"], [-100, 50])
        self.assertAlmostEqual(out["IRR_system"], -.5)
        self.assertAlmostEqual(out["NPV_system"], -100 + 50 / 1.07)

    def test_initial_payment_phasing(self):
        cfg = dict(self.config, phasing_yr=[0, 1], phasing_CAPEX=[1, 1])
        out = self.calculate(cfg)
        np.testing.assert_allclose(out["CAPEX_w_vec"][:3], [50, 50, 0])
        self.assertEqual(out["CAPEX_system"], 100)
        self.assertAlmostEqual(out["NPV_system"], out["discounted_cashflow_vec"].sum())

    def test_no_annualized_cost_or_search_penalty_inputs(self):
        self.assertFalse(any("annualized" in name or "penalty" in name for name in INPUT_NAMES))
        cfg = dict(self.config, reliability_target=.1, joint_penalty_gamma=999,
                   wind_WACC=.99, datacenter_discount_rate=.99)
        first, second = self.calculate(), self.calculate(cfg)
        np.testing.assert_array_equal(first["cashflow_vec"], second["cashflow_vec"])

    def test_invalid_assumptions_and_uncovered_asset_lives(self):
        cases = [dict(project_discount_rate=-1), dict(tax_rate=1.1),
                 dict(hpp_economic_lifetime_years=2.5), dict(phasing_yr=[-1]),
                 dict(phasing_CAPEX=[0]), dict(inflation=[-1, 0]),
                 dict(datacenter_it_lifetime_years=0), dict(wind_life_y=20),
                 dict(ref_yr_inflation=1), dict(residual_value_fraction=2)]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.calculate(dict(self.config, **change))
        with self.assertRaises(ValueError):
            self.calculate(dict(self.config, battery_life_y=5), dict(self.inputs, CAPEX_b=100))

    def test_openmdao_promoted_connections_match_direct_model(self):
        assumptions = {k: self.config[k] for k in FINANCE_PARAMETERS}
        problem = om.Problem(reports=False)
        try:
            problem.model.add_subsystem("finance", finance_simple_datacenter_comp(**assumptions), promotes=["*"])
            problem.setup()
            for name, value in self.inputs.items():
                problem.set_val(name, value)
            problem.run_model()
            expected = self.calculate()
            for name in OUTPUT_NAMES:
                np.testing.assert_allclose(problem.get_val(name), expected[name], equal_nan=True)
        finally:
            problem.cleanup()


if __name__ == "__main__":
    unittest.main()
