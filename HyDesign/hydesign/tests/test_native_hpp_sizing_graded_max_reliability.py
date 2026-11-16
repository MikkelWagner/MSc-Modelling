"""Graded scoring/validation tests and opt-in short native EGO integration runs.

python -B -m unittest hydesign.tests.test_native_hpp_sizing_graded_max_reliability -v
python -B -m hydesign.tests.test_native_hpp_sizing_graded_max_reliability --smoke
"""
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from hydesign.assembly import hpp_assembly_simple_datacenter_hpp_sizing_graded_max_reliability as adapter
from hydesign import sizing_native_hpp_sizing_graded_max_reliability as runner


class PenaltyTests(unittest.TestCase):
    def test_formula_and_feasible_first_ordering(self):
        p = adapter.penalty_configuration(.9, 342.3591147603436)
        self.assertAlmostEqual(p['base_rejection_objective'], 345.782705907947)
        self.assertEqual(p['max_failed_hours'], 876)
        self.assertAlmostEqual(p['penalty_coefficient'], 100/7884)
        feasible = adapter.score_candidate(p, valid=True, feasible=True, lcoe=342.3591147603436, lolh=876)
        near = adapter.score_candidate(p, valid=True, lolh=877, lcoe=1)
        far = adapter.score_candidate(p, valid=True, lolh=8760, lcoe=None)
        invalid = adapter.score_candidate(p, valid=False)
        self.assertAlmostEqual(near['ego_objective_score'], p['base_rejection_objective']+100/7884)
        self.assertAlmostEqual(far['penalty_contribution'], 100)
        scores = [x['ego_objective_score'] for x in (feasible, near, far, invalid)]
        self.assertTrue(all(a < b for a,b in zip(scores,scores[1:])))
        self.assertIsNone(invalid['excess_failed_hours'])

    def test_99_999_and_100_percent_boundaries(self):
        for target, allowed in ((.99,87),(.999,8),(1.,0)):
            with self.subTest(target=target):
                p = adapter.penalty_configuration(target, 100)
                self.assertEqual(p['max_failed_hours'], allowed)
                self.assertEqual(adapter.score_candidate(p, valid=True, feasible=True,
                    lcoe=50, lolh=allowed)['ego_objective_score'],50)
                s = adapter.score_candidate(p, valid=True, lolh=allowed+1)
                self.assertEqual(s['excess_failed_hours'],1)
                self.assertGreater(s['ego_objective_score'],p['base_rejection_objective'])
                with self.assertRaises(ValueError):
                    adapter.score_candidate(p, valid=True, feasible=True, lcoe=50, lolh=allowed+1)

    def test_configurable_scale_and_fixed_baseline(self):
        p = adapter.penalty_configuration(.9, 200, lcoe_ref=150, gamma=3)
        self.assertAlmostEqual(p['penalty_coefficient'],450/7884)
        p = adapter.penalty_configuration(.9,200,mode='fixed')
        for failed in (877,8760):
            self.assertEqual(adapter.score_candidate(p,valid=True,lolh=failed)['ego_objective_score'],202)
        self.assertEqual(adapter.score_candidate(p,valid=False)['ego_objective_score'],202)

    def test_invalid_settings_and_inconsistent_metrics(self):
        for kwargs in ({'gamma':0},{'gamma':float('nan')},{'lcoe_ref':-1},
                       {'gamma':True},{'mode':'other'},{'gamma':1e308,'lcoe_ref':1e308}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                adapter.penalty_configuration(.9,100,**kwargs)
        for target in (0,1.01,float('nan')):
            with self.assertRaises(ValueError): adapter.penalty_configuration(target,100)
        p = adapter.penalty_configuration(.99,100)
        for failed in (None,float('nan'),1.5,-1,8761):
            with self.assertRaises(ValueError): adapter.score_candidate(p,valid=True,lolh=failed)

    def test_certificate_scope_and_unresolved_status_are_distinct(self):
        checks = dict(outer_feasible=False,max_failed_hours=876)
        status = dict(termination='TIME_LIMIT_INCUMBENT',certificate=dict(failed_hours_lower_bound=950))
        a = adapter.reliability_assessment(checks,status,.001)
        self.assertEqual(a['reliability_assessment'],'BELOW_TARGET_EXACT_SERVICE_CERTIFIED')
        self.assertTrue(a['exact_service_target_excluded'])
        self.assertFalse(a['reported_target_excluded'])
        self.assertTrue(a['reported_target_feasibility_unresolved'])
        self.assertTrue(adapter.reliability_assessment(checks,status,0)['reported_target_excluded'])
        status['certificate']['failed_hours_lower_bound']=800
        self.assertEqual(adapter.reliability_assessment(checks,status,.001)['reliability_assessment'],
                         'BELOW_TARGET_TIME_LIMIT_UNRESOLVED')
        self.assertEqual(adapter.reliability_assessment(None,{},.001)['reliability_assessment'],
                         'INVALID_EVALUATION')
        self.assertEqual(adapter.reliability_assessment(None,{},.001,
            [dict(has_solution=False,status_code=108)])['reliability_assessment'],
            'UNRESOLVED_NO_VALID_DISPATCH')


class AdapterTests(unittest.TestCase):
    def setUp(self):
        parent = runner.ROOT/'hpp_sizing_runs/lcoe_reliability_frontier_graded_max_reliability'
        parent.mkdir(parents=True,exist_ok=True)
        # Match the existing tests: tempfile's private Windows ACL is not
        # accessible to this workspace's sandbox identity.
        self.directory = (parent/('test_graded_max_reliability_'+uuid4().hex)).resolve()
        assert self.directory.parent == parent.resolve()
        self.directory.mkdir()
        def cleanup():
            assert self.directory.parent == parent.resolve()
            assert self.directory.name.startswith('test_graded_max_reliability_')
            try:
                shutil.rmtree(self.directory)
            except PermissionError:
                pass  # OneDrive may hold a new scratch directory; preserve it.
        self.addCleanup(cleanup)

    def candidate(self, **kwargs):
        # Reuse the existing balanced synthetic dispatch fixture, not its tests.
        from hydesign.tests.test_native_hpp_sizing_base_max_reliability import NativeMaximumReliabilityTests
        model,design,context = NativeMaximumReliabilityTests.candidate(self,**kwargs)
        model.__class__ = adapter.hpp_model_simple_datacenter_hpp_sizing_graded_max_reliability
        model.penalty = adapter.penalty_configuration(model.sim_pars['reliability_target'],model.objective_upper_bound)
        model.rejection_score = -model.penalty['base_rejection_objective']
        return model,design,context

    def records(self):
        return [json.loads(p.read_text()) for p in self.directory.glob('*.json')]

    def test_below_target_archive_and_history_preserve_actual_economics(self):
        import pandas as pd
        model,design,context = self.candidate(failed=976,target=.9)
        with context: result = model.evaluate(*design)
        record = self.records()[0]
        self.assertEqual(record['status'],'BELOW_TARGET')
        self.assertEqual(record['excess_failed_hours'],100)
        self.assertEqual(record['metrics']['LOLH'],976)
        self.assertAlmostEqual(record['lcoe_hpp_eur_per_mwh'],1e7/((8760-976)*16))
        self.assertEqual(-result[-1],record['ego_objective_score'])
        self.assertTrue(record['physical_validation_passed'])
        self.assertTrue(Path(record['dispatch_path']).exists())
        runner.write_evaluation_history([record],self.directory/'history.csv',0,1,reliability_target=.9)
        row = pd.read_csv(self.directory/'history.csv').iloc[0]
        self.assertEqual(row['excess_failed_hours'],100)
        self.assertEqual(row['failed_hours'],976)
        self.assertEqual(row['annualized_cost_hpp'],1e7)
        self.assertAlmostEqual(row['ego_objective_score'],-result[-1])
        self.assertFalse(row['accepted'])

    def test_physical_violation_returns_worse_score_without_stale_metrics(self):
        model,design,context = self.candidate()
        physical = adapter.hpp_model_simple_datacenter_with_hpp_cost_max_reliability
        with context:
            original = physical.evaluate
            def corrupt(instance,*x):
                output = original(instance,*x)
                instance.hourly['curtailment_mw'][0] += 1
                return output
            with patch.object(physical,'evaluate',corrupt): result=model.evaluate(*design)
        record = self.records()[0]
        self.assertFalse(record['physical_validation_passed'])
        self.assertIn('Physical balance failure',record['error'])
        self.assertEqual(-result[-1],model.penalty['invalid_rejection_objective'])
        for field in ('metrics','economics','lcoe_hpp_eur_per_mwh','excess_failed_hours','dispatch_path'):
            self.assertIsNone(record[field])

    def test_native_objective_and_recovery_choose_lcoe_not_cost(self):
        from types import SimpleNamespace
        from hydesign.Parallel_EGO import model_evaluation
        from hydesign import sizing_native_hpp_sizing_base_max_reliability as baseline
        scores=[]
        for n,failed,cost in ((1,4000,1e7),(2,0,1.1e7)):
            model,design,context=self.candidate(failed=failed,turbines=n,cost=cost)
            inputs=dict(hpp_model=lambda **kwargs:model,scaler=SimpleNamespace(inverse_transform=lambda x:x),
                list_vars=model.list_vars,design_vars=model.list_vars,fixed_vars=[],variables=model.variables,
                opt_sign=-1,op_var_index=2)
            with context: scores.append(float(model_evaluation((np.array([design]),inputs))))
        self.assertLess(scores[1],scores[0])
        with patch.object(baseline,'annual_cost_from_dispatch',return_value=1.1e7):
            best,errors,_=runner.independently_validate(self.records(),model.sim_pars,self.directory)
        self.assertFalse(errors)
        self.assertEqual(best['design'][3],2)
        self.assertAlmostEqual(best['lcoe_hpp_eur_per_mwh'],scores[1])

    def test_no_incumbent_has_no_invented_reliability_in_history(self):
        import pandas as pd
        model,design,_ = self.candidate()
        def unresolved(instance,*x):
            instance.solve_observations.append(dict(status_code=108,status='time limit',has_solution=False))
            raise RuntimeError('No incumbent')
        with patch.object(adapter.hpp_model_simple_datacenter_with_hpp_cost_max_reliability,
                          'evaluate',unresolved):
            model.evaluate(*design)
        record=self.records()[0]
        self.assertEqual(record['reliability_assessment'],'UNRESOLVED_NO_VALID_DISPATCH')
        self.assertIsNone(record['reported_target_excluded'])
        self.assertEqual(record['ego_objective_score'],model.penalty['invalid_rejection_objective'])
        runner.write_evaluation_history([record],self.directory/'history.csv',0,1,reliability_target=.5)
        row=pd.read_csv(self.directory/'history.csv').iloc[0]
        for field in ('reliability','failed_hours','excess_failed_hours','lcoe_hpp_eur_per_mwh'):
            self.assertTrue(pd.isna(row[field]),field)

    def test_fixed_mode_matches_unchanged_adapter(self):
        from hydesign.tests.test_native_hpp_sizing_base_max_reliability import NativeMaximumReliabilityTests
        for failed in (0,1000):
            model,design,context=self.candidate(failed=failed,target=.9)
            model.penalty['rejection_mode']='fixed'
            model.penalty['invalid_rejection_objective']=model.penalty['base_rejection_objective']
            with context: actual=model.evaluate(*design).copy()
            baseline,design,context=NativeMaximumReliabilityTests.candidate(self,failed=failed,target=.9)
            baseline.rejection_score=model.rejection_score
            with context: expected=baseline.evaluate(*design)
            np.testing.assert_array_equal(actual,expected)

    def test_final_recovery_cannot_accept_below_target_even_with_forged_score(self):
        model,design,context = self.candidate(failed=88,target=.99)
        with context: model.evaluate(*design)
        record = self.records()[0]
        best,_,_=runner.independently_validate([record],model.sim_pars,self.directory)
        self.assertIsNone(best)
        # Recovery recomputes target compliance instead of trusting status/score.
        record.update(status='FEASIBLE',score=-.01)
        best,errors,_=runner.independently_validate([record],model.sim_pars,self.directory)
        self.assertIsNone(best)
        self.assertIn('below the requested target',errors[0]['error'])

    def test_adapter_acceptance_at_high_reliability_boundaries(self):
        for target,allowed in ((.99,87),(.999,8),(1.,0)):
            for failed in (allowed,allowed+1):
                model,design,context=self.candidate(failed=failed,target=target)
                with context: model.evaluate(*design)
                self.assertEqual(model.candidate_status,'FEASIBLE' if failed==allowed else 'BELOW_TARGET')

    def test_config_and_notebook(self):
        config=runner.ROOT/'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml'
        cfg=runner.prepare_configuration(config)
        self.assertEqual(cfg['hpp_sizing_lcoe_ref'],100)
        self.assertEqual(cfg['hpp_sizing_penalty_gamma'],1)
        self.assertEqual(cfg['hpp_sizing_rejection_mode'],'graded')
        self.assertIn('graded_max_reliability',cfg['hpp_sizing_output_dir'])
        cfg=runner.prepare_configuration(config,dict(hpp_sizing_lcoe_ref=120,hpp_sizing_penalty_gamma=2))
        self.assertEqual(cfg['hpp_sizing_lcoe_ref'],120)
        notebook=runner.ROOT/'docs/notebooks/simple_datacenter_lcoe_reliability_with_hpp_sizing_graded_max_reliability.ipynb'
        for cell in json.loads(notebook.read_text())['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),str(notebook),'exec')
                self.assertFalse(cell['outputs'])


def smoke_experiment():
    """Explicit opt-in: two short annual-profile runs, not the full frontier."""
    import matplotlib
    matplotlib.use('Agg')  # Save figures without opening a desktop GUI in CLI checks.
    config_path=runner.ROOT/'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml'
    config=runner.read_configuration(config_path)
    for name in ('Nwt','solar_MW [MW AC]','b_P [MW]','b_E_h [h]'):
        config['variables'][name]['var_type']='design'
    config.update(facility_load_mw=16,pue=1.15,ems_time_limit_seconds=5,ems_threads=1,
                  hpp_sizing_time_limit_seconds=150,n_doe=8,n_procs=2,n_clusters=1,
                  max_iter=1,min_conv_iter=100,npred=200,n_seed=0,
                  hpp_sizing_rejection_mode='graded',hpp_sizing_lcoe_ref=100.,hpp_sizing_penalty_gamma=1.)
    directory=runner.run_lcoe_reliability_experiment(config_path,config,[.7,.9],
        runner.ROOT/'hpp_sizing_runs/lcoe_reliability_frontier_graded_max_reliability')
    import pandas as pd
    for label in ('070','090'):
        folder=directory/('reliability_'+label)
        result=json.loads((folder/'result.json').read_text())
        assert result['termination_reason'] in ('MAX_ITERATIONS','TIME_LIMIT','NATIVE_CONVERGENCE'), result
        assert result['evaluated_designs'] > config['n_doe'], result
        assert result['best'] is not None, result
        assert result['best']['validation']['outer_feasible']
        history=pd.read_csv(folder/'evaluations.csv')
        for _, row in history[history['physical_validation_passed']==True].iterrows():
            expected=(row.lcoe_hpp_eur_per_mwh if row.outer_feasible else
                      row.base_rejection_objective+row.penalty_coefficient*row.excess_failed_hours)
            np.testing.assert_allclose(row.ego_objective_score,expected)
            np.testing.assert_allclose(row.score,-expected)
        print('VERIFIED',label,result['evaluated_designs'],result['best']['lcoe_hpp_eur_per_mwh'],flush=True)
    print('SMOKE_EXPERIMENT',directory,flush=True)
    return directory


if __name__=='__main__':
    import sys
    if '--smoke' in sys.argv: smoke_experiment()
    else: unittest.main()
