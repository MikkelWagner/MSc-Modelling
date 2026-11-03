"""Exactness checks for the experimental sizing formulation (full CPLEX)."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
from docplex.mp.model import Model

from hydesign.ems.ems_simple_datacenter import ems_cplex_constraints_simple_datacenter
from hydesign.sizing_simple_datacenter_with_dc_cost import size_datacenter_with_dc_cost as baseline
from hydesign.sizing_simple_datacenter_with_dc_cost_strengthened import (
    size_datacenter_with_dc_cost as strengthened,
    add_reliability_strengthening, improving_load_lower_bound,
)


def toy(**updates):
    values = dict(
        wind_ts=[10, 10, 10, 2], solar_ts=[0]*4, b_P=0, b_E=0,
        pue=2, reliability_target=0.75, facility_load_min_mw=1,
        facility_load_max_mw=20, time_limit_seconds=10,
        datacenter_facility_capex_per_mw_it=0,
        datacenter_it_capex_per_mw_it=0,
        datacenter_fixed_opex_per_mw_it_year=350,
        datacenter_discount_rate=0.075,
        datacenter_facility_lifetime_years=20,
        datacenter_it_lifetime_years=5,
        datacenter_firm_revenue_per_mwh_it=100,
    )
    values.update(updates)
    return values


class StrengtheningTests(unittest.TestCase):
    def test_all_variants_preserve_known_optimum_and_eur_reporting(self):
        expected = baseline(**toy())
        variants = [dict(tighten_load_upper_bound=False, aggregate_shortage=False,
                         hourly_strengthening=False),
                    dict(tighten_load_upper_bound=True, aggregate_shortage=False,
                         hourly_strengthening=False),
                    dict(tighten_load_upper_bound=False, aggregate_shortage=True,
                         hourly_strengthening=False),
                    dict(tighten_load_upper_bound=False, aggregate_shortage=False,
                         hourly_strengthening=True, incumbent_objective_eur=50),
                    dict(tighten_load_upper_bound=False, aggregate_shortage=False,
                         hourly_strengthening=False, primary_objective_scale=1e6),
                    dict(incumbent_objective_eur=50),
                    dict(incumbent_objective_eur=50, primary_objective_scale=1e6)]
        for options in variants:
            with self.subTest(options=options):
                result = strengthened(**toy(), **options)
                self.assertAlmostEqual(result['facility_load_mw'], expected['facility_load_mw'], places=6)
                self.assertAlmostEqual(result['economics']['net_annual_value'], 50, places=3)
                status = result['solver_status']
                self.assertTrue(status['optimal'])
                self.assertAlmostEqual(status['stages'][0]['objective'], 50, places=5)
                self.assertAlmostEqual(status['stages'][0]['best_bound'], 50, places=5)
                self.assertEqual(len(status['stages']), 1)
                self.assertNotIn('net_lock_tolerance_eur', status)
                self.assertLess(max(result['validation'].values()), 1e-5)

    def test_pure_rows_preserve_every_small_integer_failure_pattern(self):
        # Compare feasibility AND optimized served energy for every allowed pattern,
        # with chronological storage and non-unit efficiency; not just one optimum.
        for pattern in itertools.product((0, 1), repeat=4):
            if sum(pattern) > 2:
                continue
            objectives = []
            for extra in (False, True):
                with Model() as mdl:
                    load = mdl.continuous_var(lb=1, ub=6)
                    variables = ems_cplex_constraints_simple_datacenter(
                        mdl, [0, 7, 1, 8], [0]*4, 3, 8, .9, .9, .5,
                        load, 6, 1.2, .5,
                    )
                    for z, value in zip(variables['failed_hour'], pattern):
                        z.lb = z.ub = value
                    if extra:
                        add_reliability_strengthening(mdl, variables, load, 1.2, 1)
                    mdl.maximize(100 * variables['delivered_energy_mwh'] - 3 * load)
                    sol = mdl.solve()
                    objectives.append(None if sol is None else sol.objective_value)
            self.assertEqual(objectives[0] is None, objectives[1] is None)
            if objectives[0] is not None:
                self.assertAlmostEqual(*objectives, places=5)

    def test_upper_bound_with_positive_economic_coefficient(self):
        inputs = toy(datacenter_fixed_opex_per_mw_it_year=0)
        expected = baseline(**inputs)
        result = strengthened(**inputs)
        self.assertAlmostEqual(result['economics']['net_annual_value'],
                               expected['economics']['net_annual_value'], places=4)
        details = result['formulation']['bound_details']
        self.assertGreater(details['minimum_value_per_mw'], 0)
        self.assertGreaterEqual(result['physical_load_upper_mw'], expected['facility_load_mw']-1e-7)

    def test_no_reliability_full_reliability_and_zero_revenue(self):
        for reliability, rate in ((0, 100), (1, 100), (.75, 0)):
            with self.subTest(reliability=reliability, rate=rate):
                inputs = toy(reliability_target=reliability,
                             datacenter_firm_revenue_per_mwh_it=rate)
                expected = baseline(**inputs)
                result = strengthened(**inputs)
                self.assertAlmostEqual(result['economics']['net_annual_value'],
                                       expected['economics']['net_annual_value'], places=3)

    def test_lower_bound_is_pruning_not_incumbent_load(self):
        self.assertLessEqual(improving_load_lower_bound(413046614.91, 8760, 4000,
                             4980720.163340803, 1.15, 1), 15.8022284542)
        self.assertEqual(improving_load_lower_bound(-50, 4, 0, 350, 2, 1), 1)
        self.assertEqual(improving_load_lower_bound(-50, 4, 100, 350, 2, 1), 1)

    def test_no_mip_start_or_cleanup_and_single_dispatch_solve(self):
        original = Model.solve
        solves = []
        def check(mdl, **kwargs):
            self.assertEqual(mdl.number_of_mip_starts, 0)
            self.assertIsNone(mdl.get_constraint_by_name('preserve_primary_net_annual_value'))
            solves.append(mdl.name)
            return original(mdl, **kwargs)
        with patch.object(Model, 'solve', check):
            strengthened(**toy(), primary_objective_scale=1e6, tighten_load_upper_bound=False)
        self.assertEqual(len(solves), 1)

    def test_scaled_penalty_and_certified_bound_match_unscaled_solver(self):
        for weight in (0, 0.0001, 0.001):
            inputs = toy(wind_ts=[32, 0], solar_ts=[0, 0], b_P=25, b_E=200,
                         pue=1.15, reliability_target=.5, facility_load_min_mw=16,
                         facility_load_max_mw=16.1, battery_power_fluctuation_penalty=weight)
            expected = baseline(**inputs)
            for scale in (1., 1e6):
                with self.subTest(weight=weight, scale=scale):
                    result = strengthened(**inputs, primary_objective_scale=scale,
                                          incumbent_objective_eur=expected['solver_status']['stages'][0]['objective'])
                    stage = result['solver_status']['stages'][0]
                    self.assertAlmostEqual(stage['objective'], expected['solver_status']['stages'][0]['objective'], places=4)
                    self.assertAlmostEqual(stage['solver_objective'] * scale, stage['objective'])
                    self.assertAlmostEqual(stage['best_bound'], stage['objective'], places=4)
                    self.assertAlmostEqual(stage['battery_penalty_eur'], weight * 100 * 29.76, places=5)
                    self.assertAlmostEqual(result['economics']['net_annual_value'],
                                           expected['economics']['net_annual_value'], places=4)

    def test_invalid_certificate_and_scale(self):
        for options in ({'incumbent_objective_eur': np.nan},
                        {'primary_objective_scale': 0},
                        {'incumbent_objective_eur': 1e20}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                strengthened(**toy(), **options)


if __name__ == '__main__':
    unittest.main()
