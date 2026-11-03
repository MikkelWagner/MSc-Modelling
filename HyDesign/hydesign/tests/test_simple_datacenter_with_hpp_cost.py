"""Native-cost parity, annualization and unchanged technical/sizing routes."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from hydesign.assembly.hpp_assembly_simple_datacenter_with_dc_cost import hpp_model_simple_datacenter_with_dc_cost
from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost import hpp_model_simple_datacenter_with_hpp_cost
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost
from hydesign.costs.cost_simple_hpp_with_hpp_cost import (
    hpp_cost_annualized, HPP_COST_NAMES, ANNUAL_HPP_COST_NAMES,
)
import hydesign.sizing_simple_datacenter_with_hpp_cost as extension
from hydesign.sizing_simple_datacenter_with_dc_cost_strengthened import size_datacenter_with_dc_cost


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "hydesign/examples"
YAML_PATH = EXAMPLES / "thesis_simple_datacenter/simple_datacenter.yml"
PARAMETERS = yaml.safe_load(YAML_PATH.read_text())
WIND_KEYS = ("wind_turbine_cost", "wind_civil_works_cost", "wind_fixed_onm_cost",
             "wind_variable_onm_cost", "d_ref", "hh_ref", "p_rated_ref")
PV_KEYS = ("solar_PV_cost", "solar_hardware_installation_cost", "solar_inverter_cost", "solar_fixed_onm_cost")
BATTERY_KEYS = ("battery_energy_cost", "battery_power_cost", "battery_BOP_installation_commissioning_cost",
                "battery_control_system_cost", "battery_energy_onm_cost")


def parameters(keys):
    return {key: PARAMETERS[key] for key in keys}


class ConfigurationAndAnnualizationTests(unittest.TestCase):
    def test_shared_yaml_contains_all_workflows_and_native_cost_assumptions(self):
        original = yaml.safe_load((EXAMPLES / "Europe/hpp_pars.yml").read_text())
        self.assertEqual(list(YAML_PATH.parent.glob("*.yml")), [YAML_PATH])
        self.assertEqual(PARAMETERS["reliability_target"], .999)
        self.assertEqual(PARAMETERS["datacenter_discount_rate"], .075)
        for name in ("facility_load_min_mw", "sizing_time_limit_seconds",
                     "hpp_sizing_time_limit_seconds", "variables", "optimization_method"):
            self.assertIn(name, PARAMETERS)
        for name in WIND_KEYS + PV_KEYS + BATTERY_KEYS + ("wind_WACC", "solar_WACC", "battery_WACC"):
            self.assertEqual(PARAMETERS[name], original[name], name)
        self.assertEqual(PARAMETERS["hpp_economic_lifetime_years"], original["N_life"])
        self.assertEqual(PARAMETERS["N_life"], 1)
        self.assertEqual(PARAMETERS["hpp_BOS_soft_cost"], 119940)
        self.assertEqual(PARAMETERS["hpp_grid_connection_cost"], 0)
        self.assertEqual(PARAMETERS["land_cost"], 300000)
        self.assertEqual(PARAMETERS["battery_replacement_year"], 15)
        self.assertEqual(PARAMETERS["battery_replacement_cost_fraction"], 1.)
        for name in ("decommissioning_cost_w", "decommissioning_cost_s", "battery_price_reduction_per_year"):
            self.assertNotIn(name, PARAMETERS)

    def test_crfs_and_aggregation(self):
        model = hpp_cost_annualized(**parameters(("wind_WACC", "solar_WACC", "battery_WACC", "hpp_economic_lifetime_years")))
        expected = .06 * 1.06**25 / (1.06**25 - 1)
        for crf in (model.CRF_wind, model.CRF_pv, model.CRF_battery):
            self.assertAlmostEqual(crf, expected, places=14)
        out = dict(zip(ANNUAL_HPP_COST_NAMES, model.compute(100, 10, 200, 20, 300, 30)))
        self.assertEqual(out["CAPEX_hpp"], 600)
        self.assertAlmostEqual(out["annualized_CAPEX_hpp"], (600 + 300/1.06**15) * expected)
        self.assertAlmostEqual(out["annualized_cost_hpp"], (600 + 300/1.06**15) * expected + 60)
        self.assertAlmostEqual(out["annualized_cost_hpp"], sum(out['annualized_cost_'+k] for k in ('wind','pv','battery')))
        zero = hpp_cost_annualized(0, 0, 0, 25)
        self.assertEqual(zero.CRF_wind, 1 / 25)
        self.assertAlmostEqual(zero.compute(100, 0, 0, 0, 0, 0)[0], 4)

    def test_shared_annualization_included_once(self):
        model = hpp_cost_annualized(.06, .06, .06, 25)
        crf = .06 / (1 - 1.06**-25)
        out = dict(zip(ANNUAL_HPP_COST_NAMES, model.compute(100, 10, 200, 20, 300, 30, 400, 0)))
        self.assertAlmostEqual(out["annualized_cost_shared"], 400 * crf)
        self.assertEqual(out["CAPEX_hpp"], 1000)
        self.assertEqual(out["OPEX_hpp"], 60)
        self.assertAlmostEqual(out["annualized_cost_hpp"], (1000+300/1.06**15) * crf + 60)

    def test_one_full_replacement_for_40_mw_846_mwh_battery(self):
        native = battery_cost(**parameters(BATTERY_KEYS), life_y=1, battery_price_reduction_per_year=0)
        initial, opex = native.compute(846.16, 40, np.ones(8760))
        self.assertEqual(initial, 56391920)
        self.assertEqual(opex, 0)
        annual = hpp_cost_annualized(.06, .06, .06, 25, 15, 1.)
        out = dict(zip(ANNUAL_HPP_COST_NAMES, annual.compute(100, 0, 200, 0, initial, opex, 300, 0)))
        discounted = initial/1.06**15
        crf = .06/(1-1.06**-25)
        self.assertEqual(out["battery_replacement_cost"], initial)
        self.assertAlmostEqual(out["discounted_battery_replacement_cost"], discounted)
        self.assertAlmostEqual(out["annualized_CAPEX_battery"], crf*(initial+discounted))
        self.assertEqual(out["annualized_CAPEX_battery"], out["annualized_cost_battery"])
        self.assertEqual(out["CAPEX_hpp"], initial+600)
        np.testing.assert_allclose(out["annualized_cost_hpp"]-crf*(initial+600), crf*discounted, rtol=1e-12)

    def test_mandatory_replacement_validation_zero_rate_and_zero_battery(self):
        for year, fraction in ((0,1), (25,1), (30,1), (15.5,1), (np.nan,1),
                               (15,0), (15,-1), (15,np.nan)):
            with self.assertRaises(ValueError):
                hpp_cost_annualized(.06,.06,.06,25,year,fraction)
        zero = hpp_cost_annualized(0,0,0,25)
        out = dict(zip(ANNUAL_HPP_COST_NAMES, zero.compute(0,0,0,0,100,0)))
        self.assertEqual(out["discounted_battery_replacement_cost"], 100)
        self.assertEqual(out["CAPEX_hpp"], 100)
        self.assertEqual(out["annualized_cost_battery"], 8)
        out = dict(zip(ANNUAL_HPP_COST_NAMES, zero.compute(0,0,0,0,0,0)))
        self.assertEqual(out["battery_replacement_cost"], 0)
        self.assertEqual(out["annualized_cost_battery"], 0)

    def test_replacement_configuration_is_forwarded(self):
        annual = hpp_cost_annualized(.06,.06,.06,25,battery_replacement_year=10,
                                     battery_replacement_cost_fraction=.5)
        out = dict(zip(ANNUAL_HPP_COST_NAMES, annual.compute(0,0,0,0,100,0)))
        self.assertEqual(out["battery_replacement_cost"], 50)
        self.assertAlmostEqual(out["discounted_battery_replacement_cost"], 50/1.06**10)

    def test_invalid_economic_inputs(self):
        for rate, life in ((-.1, 25), (np.nan, 25), (.06, 0), (.06, np.inf)):
            with self.assertRaises(ValueError):
                hpp_cost_annualized(rate, .06, .06, life)

    def test_battery_healthy_soh_has_one_initial_investment(self):
        expected = PARAMETERS['battery_energy_cost'] * 200 + 25 * sum(
            PARAMETERS[k] for k in BATTERY_KEYS[1:4])
        for years, reduction in ((1, 0), (25, 0), (25, .1)):
            soh = np.ones(8760 * years)
            events = np.flatnonzero((soh > .99) & (np.append(1, np.diff(soh)) > 0))
            np.testing.assert_array_equal(events, [0])
            model = battery_cost(**parameters(BATTERY_KEYS), life_y=years,
                                 battery_price_reduction_per_year=reduction)
            capex, opex = model.compute(200, 25, soh)
            self.assertEqual(capex, expected)
            self.assertEqual(opex, 200 * PARAMETERS['battery_energy_onm_cost'])

    def test_notebooks_use_new_workflow_without_stale_outputs(self):
        for name in ('simple_datacenter_with_hpp_cost', 'simple_datacenter_sizing_with_hpp_cost'):
            notebook = json.loads((ROOT / 'docs/notebooks' / (name+'.ipynb')).read_text(encoding='utf8'))
            sources = []
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    compile(source, name, 'exec')
                    self.assertEqual(cell['outputs'], [])
                    sources.append(source)
            code = '\n'.join(sources)
            self.assertIn('simple_datacenter.yml', code)
            self.assertIn('hpp_assembly_simple_datacenter_with_hpp_cost', code)
            self.assertIn('net_annual_value_system', code)


class FixedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kwargs = dict(sim_pars_fn=str(YAML_PATH),
                      input_ts_fn=str(EXAMPLES / 'Europe/GWA2/input_ts_Denmark_good_solar.csv'),
                      latitude=55.39776, longitude=11.813965, altitude=42,
                      facility_load_mw=16, pue=1.15, reliability_target=0,
                      ppa_price=PARAMETERS['ppa_price'])
        cls.previous = hpp_model_simple_datacenter_with_dc_cost(**kwargs)
        cls.hpp = hpp_model_simple_datacenter_with_hpp_cost(**kwargs)
        cls.design = [35, 300, 5, 20, 7, 180, 42, 180, 1.25, 25, 8]
        cls.previous.evaluate(*cls.design)
        cls.hpp.evaluate(*cls.design)

    @classmethod
    def tearDownClass(cls):
        cls.previous.prob.cleanup()
        cls.hpp.prob.cleanup()

    def test_native_cost_parity(self):
        prob = self.hpp.prob
        scalar = lambda name: prob.get_val(name).item()
        wind = wpp_cost(**parameters(WIND_KEYS), N_time=8760).compute(
            scalar('Nwt'), scalar('hh'), scalar('d'), scalar('p_rated'), prob.get_val('wind_t'))
        pv = pvp_cost(**parameters(PV_KEYS)).compute(scalar('solar_MW'), scalar('DC_AC_ratio'))
        battery = battery_cost(**parameters(BATTERY_KEYS), life_y=1,
                               battery_price_reduction_per_year=0).compute(200, 25, np.ones(8760))
        for names, expected in ((('CAPEX_w','OPEX_w'),wind), (('CAPEX_s','OPEX_s'),pv), (('CAPEX_b','OPEX_b'),battery)):
            np.testing.assert_array_equal([self.hpp.economics[n] for n in names],
                                          [np.asarray(value).item() for value in expected])
        np.testing.assert_array_equal(prob.get_val('SoH'), np.ones(8760))

    def test_technical_and_dc_results_unchanged(self):
        for name, values in self.previous.hourly.items():
            np.testing.assert_allclose(values, self.hpp.hourly[name], rtol=0, atol=1e-8, err_msg=name)
        np.testing.assert_allclose(self.previous.soc_mwh, self.hpp.soc_mwh, rtol=0, atol=1e-8)
        self.assertEqual(self.previous.metrics, self.hpp.metrics)
        for name, value in self.previous.economics.items():
            self.assertEqual(value, self.hpp.economics[name], name)

    def test_no_excluded_components_and_system_accounting(self):
        self.assertTrue(hasattr(self.hpp.prob.model, 'shared_cost'))
        for name in ('decommissioning_cost', 'battery_degradation', 'wind_degradation', 'pv_degradation'):
            self.assertFalse(hasattr(self.hpp.prob.model, name))
        econ = self.hpp.economics
        self.assertEqual(econ['CAPEX_hpp'], econ['CAPEX_w']+econ['CAPEX_s']+econ['CAPEX_b']+econ['CAPEX_sh'])
        self.assertEqual(econ['net_annual_value'], econ['net_annual_value_dc_only'])
        self.assertEqual(econ['net_annual_value_system'], econ['Revenue_dc']-econ['annualized_cost_dc']-econ['annualized_cost_hpp'])
        self.assertEqual(len(self.hpp.outputs), len(self.hpp.list_out_vars))

    def test_shared_proxy_land_and_annual_totals(self):
        e = self.hpp.economics
        self.assertEqual(e["G_proxy_MW"], 16 + 25)
        self.assertEqual(e["CAPEX_sh_fixed"], 119940 * 41)
        self.assertAlmostEqual(e["CAPEX_sh_land"], 300000 * max(100/7, 180*.01226))
        self.assertEqual(e["CAPEX_sh"], e["CAPEX_sh_fixed"] + e["CAPEX_sh_land"])
        self.assertEqual(e["OPEX_sh"], 0)
        self.assertAlmostEqual(e["annualized_cost_shared"], e["CAPEX_sh"]*.06/(1-1.06**-25))
        self.assertAlmostEqual(e["annualized_cost_hpp"], sum(e["annualized_cost_"+k]
                               for k in ("wind", "pv", "battery", "shared")))
        # Prove rated battery MW is used directly, not b_P/eta or b_P*eta.
        self.assertNotAlmostEqual(e["G_proxy_MW"], 16 + 25/PARAMETERS["battery_charge_efficiency"])

    def test_replacement_outputs_and_initial_capex(self):
        e = self.hpp.economics
        self.assertEqual(e["battery_replacement_cost"], e["CAPEX_b"])
        self.assertAlmostEqual(e["discounted_battery_replacement_cost"], e["CAPEX_b"]/1.06**15)
        self.assertAlmostEqual(e["annualized_CAPEX_battery"],
                               (e["CAPEX_b"]+e["discounted_battery_replacement_cost"])*.06/(1-1.06**-25))
        for name in ("battery_replacement_cost", "discounted_battery_replacement_cost"):
            self.assertIn(name+" [EUR]", self.hpp.list_out_vars)

    def test_legacy_sizer_reprices_shared_costs_at_new_load(self):
        prior = {"economics": dict(self.previous.economics), "facility_load_mw": 23.}
        with patch.object(extension, "size_datacenter_with_dc_cost", return_value=prior):
            result = extension.size_datacenter_with_hpp_cost(self.hpp)
        e = result["economics"]
        self.assertEqual(e["G_proxy_MW"], 48)
        self.assertEqual(e["CAPEX_sh_fixed"], 119940*48)
        delta = 7*119940*.06/(1-1.06**-25)
        self.assertAlmostEqual(e["annualized_cost_hpp"]-self.hpp.economics["annualized_cost_hpp"], delta)
        self.assertAlmostEqual(e["net_annual_value_system"],
                               e["net_annual_value_dc_only"]-e["annualized_cost_hpp"])
        self.assertEqual(self.hpp.economics["facility_load_mw"], 16)

    def test_sizing_delegation_preserves_result_and_matching_hpp_cost(self):
        # A returned solution is decorated without rerunning or changing dispatch.
        previous = {'economics': dict(self.previous.economics), 'facility_load_mw': 16,
                    'hourly': self.previous.hourly, 'metrics': self.previous.metrics,
                    'solver_status': self.previous.solver_status}
        with patch.object(extension, 'size_datacenter_with_dc_cost', return_value=previous) as solve:
            result = extension.size_datacenter_with_hpp_cost(self.hpp, reliability_target=.999, primary_objective_scale=1e6)
        args = solve.call_args.kwargs
        np.testing.assert_array_equal(args['wind_ts'], self.hpp.hourly['wind_mw'])
        np.testing.assert_array_equal(args['solar_ts'], self.hpp.hourly['pv_mw'])
        self.assertEqual((args['b_P'], args['b_E']), (25, 200))
        self.assertEqual(args['reliability_target'], .999)
        self.assertEqual(args['primary_objective_scale'], 1e6)
        self.assertNotIn('annualized_cost_hpp', args)
        for name in HPP_COST_NAMES:
            np.testing.assert_allclose(result['economics'][name], self.hpp.economics[name], rtol=1e-12, atol=1e-8)
        for name, value in previous['economics'].items():
            self.assertEqual(result['economics'][name], value)
        self.assertNotIn('annualized_cost_hpp', previous['economics'])
        for name in ('hourly', 'metrics', 'solver_status'):
            self.assertIs(result[name], previous[name])

    def test_reject_separate_generation_or_unprepared_assembly(self):
        with self.assertRaises(ValueError):
            extension.size_datacenter_with_hpp_cost(SimpleNamespace(economics=None))
        with self.assertRaises(ValueError):
            extension.size_datacenter_with_hpp_cost(self.hpp, b_P=30)


class SizingSolveTests(unittest.TestCase):
    def test_real_strengthened_solver_optimum_and_economics_are_unchanged(self):
        self.assertIs(extension.size_datacenter_with_dc_cost, size_datacenter_with_dc_cost)
        options = dict(pue=2, reliability_target=.75, facility_load_min_mw=1, facility_load_max_mw=20,
                       time_limit_seconds=10, primary_objective_scale=1e6,
                       tighten_load_upper_bound=True, aggregate_shortage=True, hourly_strengthening=False,
                       datacenter_facility_capex_per_mw_it=0, datacenter_it_capex_per_mw_it=0,
                       datacenter_fixed_opex_per_mw_it_year=350, datacenter_discount_rate=.075,
                       datacenter_facility_lifetime_years=20, datacenter_it_lifetime_years=5,
                       datacenter_firm_revenue_per_mwh_it=100)
        # Controlled short profile with known unique optimal facility load of 2 MW.
        wind, solar = np.array([10.,10.,10.,2.]), np.zeros(4)
        previous = size_datacenter_with_dc_cost(wind, solar, 0, 0, **options)
        hpp = SimpleNamespace(hourly={'wind_mw':wind, 'pv_mw':solar}, inputs=[0,0],
                              economics=dict.fromkeys(HPP_COST_NAMES, 0.),
                              sim_pars=PARAMETERS,
                              prob=SimpleNamespace(get_val=lambda name: np.array([0.])))
        hpp.economics['OPEX_w'] = 1000.
        result = extension.size_datacenter_with_hpp_cost(hpp, **options)
        self.assertAlmostEqual(result['facility_load_mw'], 2)
        self.assertAlmostEqual(result['facility_load_mw'], previous['facility_load_mw'])
        for name in ('Revenue_dc', 'annualized_cost_dc', 'net_annual_value'):
            self.assertAlmostEqual(result['economics'][name], previous['economics'][name])
        expected_shared = 119940*2*.06/(1-1.06**-25)
        self.assertAlmostEqual(result['economics']['net_annual_value_system'], -950-expected_shared, places=3)
        self.assertAlmostEqual(result['solver_status']['primary_net_annual_value'], 50, places=3)


if __name__ == '__main__':
    unittest.main()
