"""Small tests; full-year controlled validation lives in a separate runnable script."""
import copy
from concurrent.futures import Future
import json
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager
import time
import unittest
from unittest.mock import patch

import numpy as np
from docplex.mp.model import Model

from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_sizing import (
    failure_status, normalize_turbines, sizing_score,
)
from hydesign.ems.ems_simple_datacenter import (
    ems_cplex_constraints_simple_datacenter, ems_cplex_simple_datacenter,
)
from hydesign.sizing_simple_datacenter_with_hpp_sizing import (
    VARIABLE_NAMES, SIZABLE, BudgetExpired, RunController, BudgetedEvaluator,
    native_cost_bound, prepare_variables, read_configuration,
)
from hydesign import Parallel_EGO as native

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml'


class ConfigurationTests(unittest.TestCase):
    def test_default_999_per_mille_and_all_variable_combinations(self):
        original = read_configuration(CONFIG)
        self.assertEqual(original['facility_load_mw'], 16)
        self.assertEqual(original['reliability_target'], .999)
        self.assertEqual(int(np.floor((1-original['reliability_target'])*8760+1e-9)), 8)
        self.assertEqual(original['hpp_sizing_time_limit_seconds'], 600)
        for active in ([], ['Nwt'], list(SIZABLE[1:]), list(SIZABLE)):
            config = copy.deepcopy(original)
            for key in SIZABLE:
                config['variables'][key]['var_type'] = 'design' if key in active else 'fixed'
            variables = prepare_variables(config)
            self.assertEqual(native.get_design_vars(variables)[0], active)
            design = np.array([[variables[n]['value'] for n in active]])
            expanded = native.expand_x_for_model_eval(design, dict(list_vars=VARIABLE_NAMES,
                variables=variables, design_vars=active, fixed_vars=[n for n in VARIABLE_NAMES if n not in active]))
            np.testing.assert_array_equal(expanded[0], [35,300,5,20,7,180,42,180,1.25,25,8])
            self.assertEqual(expanded[0,-2]*expanded[0,-1], 200)

    def test_variable_names_and_types_are_strict(self):
        c = read_configuration(CONFIG)
        c['variables']['solar_MW [MW]'] = c['variables'].pop('solar_MW [MW AC]')
        with self.assertRaises(ValueError): prepare_variables(c)
        c = read_configuration(CONFIG)
        c['variables']['Nwt']['types'] = 'float'
        with self.assertRaises(ValueError): prepare_variables(c)

    def test_integer_roundoff_and_rejection(self):
        self.assertEqual(normalize_turbines(28.999999999999996), 29)
        for value in (28.5, -1, np.nan, np.inf):
            with self.assertRaises(ValueError): normalize_turbines(value)

    def test_cost_direction_and_finite_bounds(self):
        c = read_configuration(CONFIG)
        bound = native_cost_bound(c, prepare_variables(c))
        rejection = -(bound + max(1., .01*bound))/1e6
        self.assertGreater(sizing_score(100), sizing_score(200))
        for cost in (0, bound/2, bound): self.assertGreater(sizing_score(cost), rejection)
        self.assertTrue(np.isfinite(rejection))

    def test_notebook_compiles(self):
        n = json.loads((ROOT/'docs/notebooks/simple_datacenter_with_hpp_sizing.ipynb').read_text())
        for c in n['cells']:
            if c['cell_type']=='code':
                compile(''.join(c['source']), 'notebook', 'exec')
                self.assertEqual(c['outputs'], [])


class StatusAndPhysicsTests(unittest.TestCase):
    def test_status_classification(self):
        expected = {103:'PROVEN_INFEASIBLE', 107:'TIME_LIMIT_WITH_FEASIBLE_DISPATCH',
                    108:'TIME_LIMIT_UNRESOLVED', 109:'NUMERICAL_ERROR',
                    101:'CONFIGURATION_OR_PROGRAMMING_ERROR'}
        for code,status in expected.items():
            self.assertEqual(failure_status([{'status_code':code,'has_solution':code==107}], RuntimeError()), status)
        self.assertEqual(failure_status([], ValueError()), 'CONFIGURATION_OR_PROGRAMMING_ERROR')
        try:
            try: raise TimeoutError('budget')
            except TimeoutError as e: raise RuntimeError('wrapped by EMS') from e
        except RuntimeError as e:
            self.assertEqual(failure_status([],e), 'TIME_LIMIT_UNRESOLVED')

    def test_cleanup_failure_does_not_prove_physical_infeasibility(self):
        primary = {'status_code': 101, 'has_solution': True}
        cleanup = {'status_code': 103, 'has_solution': False}
        self.assertEqual(failure_status([primary, cleanup], RuntimeError()), 'NUMERICAL_ERROR')
        self.assertEqual(failure_status([primary], TimeoutError()), 'TIME_LIMIT_WITH_FEASIBLE_DISPATCH')
        cleanup['status_code'] = 108
        self.assertEqual(failure_status([primary, cleanup], RuntimeError()), 'TIME_LIMIT_WITH_FEASIBLE_DISPATCH')

    def test_real_ems_accepts_and_rejects_hard_reliability(self):
        # Short case intentionally uses 100%: no binaries may permit shortage.
        result = ems_cplex_simple_datacenter([16,16], [0,0], 0,0,.9,np.sqrt(.86),.5,16,1.15,
                                            reliability_target=1,time_limit_seconds=10)
        np.testing.assert_allclose(result[0], [16,16])
        with Model() as model:
            variables = ems_cplex_constraints_simple_datacenter(model,[0,0],[0,0],0,0,.9,np.sqrt(.86),.5,16,16,1.15,.99)
            model.minimize(0)
            self.assertIsNone(model.solve())
            self.assertIn(model.solve_details.status_code, (3,103))
            self.assertEqual(variables['max_failed_hours'],0)


class ImmediateExecutor:
    def submit(self, fn, task):
        future = Future()
        future.set_result(fn(task))
        return future


@contextmanager
def scratch_directory():
    path = ROOT / (uuid.uuid4().hex + '_with_hpp_sizing')
    path.mkdir()
    try:
        yield str(path)
    finally:
        assert path.resolve().parent == ROOT.resolve()
        shutil.rmtree(path)


class BudgetAndHistoryTests(unittest.TestCase):
    def controller(self, directory, deadline):
        c = read_configuration(CONFIG)
        kwargs = {'hpp_deadline':deadline, 'hpp_rejection_score':-100}
        return RunController(c, prepare_variables(c), kwargs, Path(directory), time.monotonic(), ImmediateExecutor())

    def test_no_candidate_launched_after_deadline(self):
        with scratch_directory() as d:
            ctl=self.controller(d,time.monotonic()-1)
            with patch('hydesign.sizing_simple_datacenter_with_hpp_sizing.evaluate_candidate') as worker:
                with self.assertRaises(BudgetExpired): ctl.evaluate([[35,300,5,20,7,180,42,180,1.25,25,8]])
                worker.assert_not_called()

    def test_history_cache_and_best_survive_budget(self):
        with scratch_directory() as d:
            ctl=self.controller(d,time.monotonic()+10)
            design=[35,300,5,20,7,180,42,180,1.25,25,8]
            candidate=dict(design=design,status='FEASIBLE',accepted=True,score=-10,error=None,
                economics={'annualized_cost_hpp':1e7},metrics={'LOLH':0,'EENS':0,'firm_reliability':1},
                hourly={'wind_mw':np.array([16.])},soc_mwh=np.array([100.,100.]))
            with patch('hydesign.sizing_simple_datacenter_with_hpp_sizing.evaluate_candidate',return_value=candidate), \
                 patch('hydesign.sizing_simple_datacenter_with_hpp_sizing.validate_selected',return_value={}):
                ctl.evaluate([design])
                ctl.evaluate([design])
            self.assertEqual(len(ctl.history),1)
            self.assertEqual(ctl.best['economics']['annualized_cost_hpp'],1e7)
            self.assertTrue((Path(d)/'history_with_hpp_sizing.csv').exists())
            self.assertEqual(json.loads((Path(d)/'best_with_hpp_sizing.json').read_text())['score'],-10)
            ctl.kwargs['hpp_deadline']=time.monotonic()-1
            with self.assertRaises(BudgetExpired): ctl.check_time()
            self.assertIsNotNone(ctl.best)

    def test_seeded_native_doe_stays_aligned_and_sign_is_correct(self):
        from sklearn.preprocessing import MinMaxScaler
        with scratch_directory() as d:
            ctl=self.controller(d,time.monotonic()+10)
            ctl.seed_designs=[[35,300,5,29,7,180,42,180,1.25,25,8]]
            evaluator=BudgetedEvaluator(ctl,1)
            x=np.array([[.5],[.8]])
            variables=ctl.variables
            kwargs=dict(scaler=MinMaxScaler().fit([[0],[400]]),design_vars=['Nwt'],
                        fixed_vars=[n for n in VARIABLE_NAMES if n!='Nwt'],list_vars=VARIABLE_NAMES,
                        variables=variables,opt_sign=-1)
            with patch.object(ctl,'evaluate',return_value=np.array([[-10.],[-20.]])) as evaluate:
                scores=evaluator.run_ydoe(None,x,**kwargs)
                self.assertAlmostEqual(evaluate.call_args.args[0][0,3],29)
                np.testing.assert_array_equal(scores,[[10],[20]])


if __name__=='__main__': unittest.main()
