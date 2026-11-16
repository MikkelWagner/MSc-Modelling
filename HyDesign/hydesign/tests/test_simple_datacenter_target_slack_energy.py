"""Small real-CPLEX tests and integration contracts for the isolated workflow."""
import csv
from pathlib import Path
from uuid import uuid4
import unittest
from unittest.mock import patch

import numpy as np
from docplex.mp.model import Model
from hydesign.ems import ems_simple_datacenter_target_slack_energy as ems
from hydesign.assembly.hpp_assembly_simple_datacenter_hpp_sizing_target_slack_energy import (
    reliability_assessment, penalty_configuration, score_candidate,
)


class TargetSlackEnergyTests(unittest.TestCase):
    def case(self, generation, **kwargs):
        args = dict(wind_ts=generation, solar_ts=np.zeros(len(generation)),
            P_batt_MW=0, E_batt_MWh=0, battery_depth_of_discharge=.9,
            charge_efficiency=np.sqrt(.86), battery_initial_soc=.5,
            facility_load_mw=1., pue=1.15, reliability_target=.7,
            threads=1, time_limit_seconds=10)
        args.update(kwargs)
        result = ems.ems_cplex_simple_datacenter_target_slack_energy(**args)
        self.assertTrue(result[-1]['physically_validated'])
        self.assertLess(result[-1]['calibration']['secondary_bound'], 1)
        self.assertTrue(result[-1]['slack_matches_observed_excess'])
        return result

    def test_zero_generation_returns_slack(self):
        s = self.case([0]*10)[-1]
        self.assertEqual(s['metrics']['LOLH'], 10)
        self.assertAlmostEqual(s['reliability_slack'], 7)
        self.assertTrue(s['certificate']['minimum_reliability_violation_certified'])

    def test_no_battery_maximizes_partial_energy(self):
        r = self.case([1,.5,0,2], reliability_target=.5)
        np.testing.assert_allclose(r[0], [1,.5,0,1])
        self.assertAlmostEqual(r[-1]['reliability_slack'], 0)
        self.assertAlmostEqual(r[-1]['delivered_energy_mwh'], 2.5)

    def test_surplus_does_not_cycle(self):
        r = self.case([10]*8, P_batt_MW=2, E_batt_MWh=4)
        self.assertAlmostEqual(r[-1]['battery_throughput_mwh'], 0)
        self.assertEqual(r[-1]['metrics']['LOLH'], 0)

    def test_storage_physics(self):
        r = self.case([3,0], P_batt_MW=2, E_batt_MWh=4, reliability_target=1.)
        np.testing.assert_allclose(r[0], 1)
        self.assertAlmostEqual(r[4][0], 1/.86)
        self.assertAlmostEqual(r[5][1], 1)
        self.assertAlmostEqual(r[3][0], r[3][-1])

    def test_initial_soc_is_not_free_energy(self):
        r = self.case([0]*4, P_batt_MW=2, E_batt_MWh=4)
        self.assertAlmostEqual(r[-1]['delivered_energy_mwh'], 0)
        self.assertAlmostEqual(r[-1]['battery_throughput_mwh'], 0)

    def test_scarce_energy_minimum_excess_is_one(self):
        r = self.case([2.5,0,0], P_batt_MW=2, E_batt_MWh=2.5,
            battery_depth_of_discharge=1, charge_efficiency=1, reliability_target=1)
        self.assertEqual(r[-1]['metrics']['LOLH'], 1)
        self.assertAlmostEqual(r[-1]['reliability_slack'], 1)
        self.assertTrue(r[-1]['certificate']['minimum_reliability_violation_certified'])

    def test_binary_vs_epsilon_reporting(self):
        s = self.case([.9995], reliability_target=1.)[-1]
        self.assertEqual(s['binary_failed_hours'], 1)
        self.assertEqual(s['metrics']['LOLH'], 0)
        self.assertAlmostEqual(s['reliability_slack'], 1)
        self.assertEqual(s['reported_excess_failed_hours'], 0)
        assessment = reliability_assessment(dict(outer_feasible=True), s, .001)
        self.assertTrue(assessment['exact_service_target_excluded'])
        self.assertFalse(assessment['reported_target_excluded'])
        self.assertEqual(assessment['reliability_assessment'], 'VALIDATED_FEASIBLE')

    def test_single_solve_only_hourly_binaries(self):
        calls = []
        class ObservedModel(Model):
            def solve(mdl, *args, **kwargs):
                calls.append(mdl.number_of_binary_variables)
                return super().solve(*args, **kwargs)
        with patch.object(ems, 'Model', ObservedModel):
            self.case([3,0,1], P_batt_MW=1, E_batt_MWh=2)
        self.assertEqual(calls, [3])

    def test_certificate_does_not_claim_time_limited_minimum(self):
        c = ems.slack_certificate(.2, .25025, 5)
        self.assertFalse(c['minimum_reliability_violation_certified'])
        s = dict(certificate=c, termination='TIME_LIMIT_INCUMBENT')
        a = reliability_assessment(dict(outer_feasible=False), s, .001)
        self.assertEqual(a['reliability_assessment'], 'BELOW_TARGET_TIME_LIMIT_UNRESOLVED')
        self.assertFalse(a['reported_target_excluded'])
        self.assertTrue(ems.slack_certificate(5.1,.25025,5)['minimum_reliability_violation_certified'])
        self.assertTrue(ems.slack_certificate(-100,.25025,0)['minimum_reliability_violation_certified'])

    def test_secondary_priority_and_disabled_regularization(self):
        c = ems.calibrate_penalties(8760,16,80,np.sqrt(.86))
        self.assertAlmostEqual(c['secondary_bound'], .25025)
        self.assertLess(c['secondary_bound'], 1)
        self.assertEqual(ems.calibrate_penalties(4,1,2,1,kappa_battery=0)['lambda_b'], 0)
        for args in (dict(beta_energy=.9,kappa_battery=.2), dict(kappa_battery=-1)):
            with self.assertRaises(ValueError): ems.calibrate_penalties(4,1,2,1,**args)

    def test_thresholds_and_no_penalty_for_allowed_hours(self):
        for target, allowed in ((.7,2628),(.9,876),(.99,87),(.999,8),(1.,0)):
            p = penalty_configuration(target,1000,mode='cost_aware_rational')
            self.assertEqual(p['max_failed_hours'], allowed)
            self.assertEqual(score_candidate(p,valid=True,feasible=True,lcoe=50,lolh=allowed)['ego_objective_score'],50)
            self.assertGreater(score_candidate(p,valid=True,lcoe=50,lolh=allowed+1)['ego_objective_score'],50)

    def test_invalid_physics_is_rejected(self):
        with patch.object(ems,'validate_dispatch',side_effect=ValueError('Simultaneous battery charging/discharging')):
            with self.assertRaisesRegex(ValueError,'Simultaneous'): self.case([1,1])

    def test_configuration_and_component_forward_target(self):
        from hydesign.sizing_native_hpp_sizing_target_slack_energy import prepare_configuration, ROOT
        from hydesign.ems.ems_simple_datacenter_comp_target_slack_energy import ems_simple_datacenter_target_slack_energy
        cfg = prepare_configuration(ROOT/'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml')
        self.assertEqual(cfg['reliability_target'], .7)
        self.assertEqual(cfg['dispatch_policy'], 'target_slack_energy')
        self.assertEqual(cfg['hpp_sizing_rejection_mode'], 'cost_aware_rational')
        self.assertEqual(cfg['ems_time_limit_seconds'],30)
        m = ems_simple_datacenter_target_slack_energy(N_time=1,reliability_target=.999)
        m.compute(np.array([.5]),np.array([0]),0,0,.9,np.sqrt(.86),.5,1,1.15)
        self.assertEqual(m.solver_status['reliability_target'],.999)

    def test_history_exports_slack_and_never_relabels_availability(self):
        from hydesign.sizing_native_hpp_sizing_target_slack_energy import write_evaluation_history
        status = self.case([.9995], reliability_target=1)[-1]
        record = dict(design=[35,300,5,6,7,53,42,180,1.25,59,2], started_at=1,
            completed_at=2, status='BELOW_TARGET', solver_status=status,
            metrics=status['metrics'], economics={'served_facility_energy_mwh':.9995},
            lcoe_hpp_eur_per_mwh=50, annualized_cost_hpp=49.975,
            ego_objective_score=50, physical_validation_passed=True)
        invalid = dict(design=record['design'], started_at=2, status='RUNNING')
        parent = Path(__file__).resolve().parent
        directory = parent/('case_'+uuid4().hex+'_target_slack_energy')
        directory.mkdir()
        path = directory/'history_target_slack_energy.csv'
        try:
            write_evaluation_history([record,invalid],path,1,3,reliability_target=1.)
            with path.open(newline='') as stream: rows=list(csv.DictReader(stream))
        finally:
            self.assertEqual(directory.resolve().parent, parent)
            path.unlink(missing_ok=True)
            path.with_suffix('.tmp').unlink(missing_ok=True)
            directory.rmdir()
        self.assertEqual(float(rows[0]['reliability_slack']),1)
        self.assertEqual(float(rows[0]['reported_excess_failed_hours']),0)
        self.assertEqual(float(rows[0]['served_facility_energy_mwh']),.9995)
        self.assertNotIn('maximum_availability_certified',rows[0])
        self.assertEqual(rows[1]['reliability_slack'],'')
        self.assertEqual(rows[1]['failed_hours'],'')


if __name__ == '__main__':
    unittest.main()
