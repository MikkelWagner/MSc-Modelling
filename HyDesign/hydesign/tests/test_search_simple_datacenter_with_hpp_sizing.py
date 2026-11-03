"""Lightweight search tests; no full-year outer optimization is started here."""
import copy
import itertools
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

from hydesign.search_simple_datacenter_with_hpp_sizing import (
    CandidatePolicy, FeasibilitySurrogate, SearchSpace, UNKNOWN_STATUSES,
    expected_improvement, run_feasibility_aware, search_settings, training_data,
)
from hydesign.sizing_simple_datacenter_with_hpp_sizing import (
    BudgetExpired, RunController, SIZABLE, VARIABLE_NAMES, prepare_variables, read_configuration,
)
from hydesign.tests.test_simple_datacenter_with_hpp_sizing import (
    CONFIG, ImmediateExecutor, scratch_directory,
)

RUNNER = 'hydesign.sizing_simple_datacenter_with_hpp_sizing'
SEARCH = 'hydesign.search_simple_datacenter_with_hpp_sizing'


def configuration():
    c = read_configuration(CONFIG)
    c.update(npred=40, n_doe=6, max_iter=3, fa_classifier_optimize=False)
    return c


def observation(count, status='FEASIBLE', cost=10e6):
    return dict(design=[35, 300, 5, count, 7, 180, 42, 180, 1.25, 25, 8],
                status=status, accepted=status == 'FEASIBLE', score=-1e100,
                economics={'annualized_cost_hpp': cost}, error=None)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.space = SearchSpace(prepare_variables(self.config), VARIABLE_NAMES)

    def test_real_costs_not_scores_and_unknowns_not_labels(self):
        rows = [observation(100, cost=12345678.9), observation(200, cost=22345678.9),
                observation(0, 'PROVEN_INFEASIBLE', cost=999e12)]
        rows += [observation(i+1, status, cost=999e12) for i, status in enumerate(sorted(UNKNOWN_STATUSES))]
        rows[0]['score'] = float('nan')
        xc, costs, xf, labels = training_data(rows, self.space)
        np.testing.assert_array_equal(costs, [12345678.9, 22345678.9])
        np.testing.assert_array_equal(labels, [1, 1, 0])
        np.testing.assert_allclose(xc[:, 0], [.25, .5])
        self.assertEqual(len(xf), 3)

    def test_invalid_acceptance_and_errors_are_diagnosed(self):
        cases = [observation(100), observation(1, 'PROVEN_INFEASIBLE'),
                 observation(1, 'TIME_LIMIT_UNRESOLVED'), observation(1, 'CONFIGURATION_OR_PROGRAMMING_ERROR')]
        for row in cases[:3]: row['accepted'] = not row['accepted']
        cases.append(observation(100, cost=float('nan')))
        for row in cases:
            with self.assertRaises(ValueError): training_data([row], self.space)

    def test_sparse_one_class_and_no_feasible_data(self):
        for rows in ([], [observation(1, 'TIME_LIMIT_UNRESOLVED')],
                     [observation(0, 'PROVEN_INFEASIBLE')], [observation(100)]):
            model = FeasibilitySurrogate(self.space, self.config)
            model.fit(rows)
            mean, std, p, ei, acquisition = model.predict([observation(50)['design']])
            self.assertIsNone(model.cost_model)
            self.assertIsNone(model.classifier)
            self.assertTrue(np.all((p > 0) & (p < 1)))
            np.testing.assert_array_equal(acquisition, [0])
            self.assertTrue(np.isnan(mean).all())

    def test_actual_native_cost_regression_and_classifier(self):
        rows = [observation(i, cost=(7+i*.4)*1e6) for i in (100, 200, 300, 400)]
        rows += [observation(0, 'PROVEN_INFEASIBLE')]
        model = FeasibilitySurrogate(self.space, self.config)
        model.fit(rows)
        self.assertIsNotNone(model.cost_model)
        self.assertIsNotNone(model.classifier)
        mean, std, p, ei, acquisition = model.predict([observation(150)['design']])
        self.assertAlmostEqual(mean[0]/1e6, 67, places=3)
        self.assertTrue(np.isfinite(std).all())
        self.assertTrue(np.all((p > 0) & (p < 1)))
        np.testing.assert_allclose(acquisition, ei*p)
        model.fit(rows + [observation(3, 'TIME_LIMIT_UNRESOLVED')])
        self.assertEqual(model.fit_count, 1)

    def test_constant_cost_and_numerical_training_failure(self):
        rows = [observation(i) for i in (100, 200, 300)]
        model = FeasibilitySurrogate(self.space, self.config)
        model.fit(rows)
        self.assertEqual(model.constant_cost, 10e6)
        np.testing.assert_array_equal(model.predict([observation(150)['design']])[-1], [0])
        rows[0]['economics']['annualized_cost_hpp'] = 9e6
        with patch(SEARCH+'.native_ego.get_sm', side_effect=np.linalg.LinAlgError('synthetic singularity')):
            model.fit(rows)
        self.assertIn('cost_model_error', model.diagnostics)
        self.assertIsNone(model.cost_model)

    def test_joint_four_variable_regression_and_classification(self):
        c = configuration()
        for name in SIZABLE: c['variables'][name]['var_type'] = 'design'
        space = SearchSpace(prepare_variables(c), VARIABLE_NAMES)
        rows = []
        for i, design in enumerate(space.sample(16, 42)):
            row = observation(design[3], cost=(10 + design[3]*.4 + design[5]*.02
                              + design[-2]*design[-1]*.01)*1e6)
            row['design'] = design.tolist()
            if i % 5 == 0: row.update(status='PROVEN_INFEASIBLE', accepted=False)
            rows.append(row)
        model = FeasibilitySurrogate(space, c)
        model.fit(rows)
        self.assertIsNotNone(model.cost_model, model.diagnostics)
        self.assertIsNotNone(model.classifier, model.diagnostics)
        for values in model.predict(space.sample(7, 43)):
            self.assertTrue(np.isfinite(values).all())

    def test_ei_edge_cases(self):
        answer = expected_improvement(10, [8, 12, 10, np.nan, 8, 8], [0, 0, 1, 1, -1, np.inf])
        np.testing.assert_allclose(answer, [2, 0, 1/np.sqrt(2*np.pi), 0, 0, 0])
        np.testing.assert_array_equal(expected_improvement(np.inf, [1], [2]), [0])
        self.assertTrue(np.isfinite(expected_improvement(10, [-1e100, 1e100], [1, 1])).all())

    def test_settings_validation(self):
        for key, bad in [('fa_batch_size', 0), ('fa_max_retries', -1),
                         ('fa_cost_theta_bounds', [20, 1]), ('fa_classifier_length_scale', 0),
                         ('fa_classifier_optimize', 'true'), ('n_seed', -1)]:
            c = dict(self.config, **{key: bad})
            with self.assertRaises(ValueError): search_settings(c)


class CandidateTests(unittest.TestCase):
    def test_all_sixteen_active_subsets_and_fixed_values(self):
        for mask in itertools.product((False, True), repeat=4):
            c = configuration()
            for name, active in zip(SIZABLE, mask):
                c['variables'][name]['var_type'] = 'design' if active else 'fixed'
            variables = prepare_variables(c)
            space = SearchSpace(variables, VARIABLE_NAMES)
            rows = space.sample(9, 17)
            self.assertEqual(space.dimension, sum(mask))
            for i, name in enumerate(VARIABLE_NAMES):
                if name in space.active:
                    self.assertTrue(np.all(rows[:, i] >= variables[name]['limits'][0]))
                    self.assertTrue(np.all(rows[:, i] <= variables[name]['limits'][1]))
                else:
                    np.testing.assert_array_equal(rows[:, i], variables[name]['value'])
            np.testing.assert_array_equal(rows[:, 3], np.round(rows[:, 3]))
            np.testing.assert_array_equal(rows, space.sample(9, 17))
            if not mask[2] and not mask[3]: np.testing.assert_array_equal(rows[:, -2]*rows[:, -1], 200)

    def policy(self):
        c = configuration()
        c.update(fa_batch_size=2, fa_retry_interval=1, fa_retry_cooldown_iterations=1)
        return CandidatePolicy(SearchSpace(prepare_variables(c), VARIABLE_NAMES), c)

    def test_new_candidates_only_from_global_pool_no_neighbours(self):
        policy = self.policy()
        records = {tuple(observation(100)['design']): observation(100)}
        pool = np.array([observation(n)['design'] for n in (3, 100, 250, 250, 398)])
        with patch.object(policy.space, 'sample', return_value=pool):
            chosen, proposals = policy.propose(records, {}, 0)
        self.assertEqual(len({tuple(row) for row in chosen}), 2)
        self.assertTrue(all(row[3] in (3, 250, 398) for row in chosen))
        self.assertTrue(all(p['selection_reason'] == 'sparse_or_flat_model_exploration' for p in proposals))

    def test_periodic_exploration_ignores_low_feasibility_probability(self):
        policy = self.policy()
        policy.proposed_count = 4  # Default exploration interval is five proposals.
        records = {tuple(observation(100)['design']): observation(100)}
        pool = np.array([observation(n)['design'] for n in (105, 400)])
        fake = (np.array([9e6, 1e6]), np.ones(2), np.array([.9, 0.]),
                np.array([1e6, 9e6]), np.array([9e5, 0.]))
        with patch.object(policy.space, 'sample', return_value=pool), patch.object(policy.surrogate, 'predict', return_value=fake):
            chosen, proposals = policy.propose(records, {}, 0)
        self.assertEqual(chosen[0][3], 400)
        self.assertEqual(proposals[0]['selection_reason'], 'global_exploration')

    def test_unknown_retry_is_ranked_capped_and_never_negative_label(self):
        policy = self.policy()
        unknown = observation(70, 'TIME_LIMIT_UNRESOLVED')
        key = tuple(unknown['design'])
        records = {key: unknown}
        pool = np.array([observation(300)['design']])
        def predictions(designs):
            score = np.array([100 if row[3] == 70 else 1 for row in designs])
            return np.zeros(len(score)), np.ones(len(score)), np.ones(len(score))*.5, score*2, score
        with patch.object(policy.space, 'sample', return_value=pool), patch.object(policy.surrogate, 'predict', side_effect=predictions):
            chosen, proposals = policy.propose(records, {key: 1}, 1)
            self.assertEqual(chosen[0][3], 70)
            self.assertTrue(proposals[0]['retry'])
            self.assertEqual(policy.surrogate.diagnostics['feasibility_training_count'], 0)
            chosen, _ = policy.propose(records, {key: 2}, 2)
            self.assertNotIn(70, [row[3] for row in chosen])


class ControllerTests(unittest.TestCase):
    def controller(self, path):
        c = configuration()
        return RunController(c, prepare_variables(c), {'hpp_deadline': time.monotonic()+30},
                             Path(path), time.monotonic(), ImmediateExecutor())

    def test_proposal_precedes_evaluation_duplicates_and_retry_limit(self):
        with scratch_directory() as path:
            control = self.controller(path)
            design = observation(70)['design']
            def worker(task):
                lines = (Path(path)/'proposals_with_hpp_sizing.jsonl').read_text().splitlines()
                self.assertEqual(json.loads(lines[-1])['design'], task[0])
                return observation(70, 'TIME_LIMIT_UNRESOLVED')
            with patch(RUNNER+'.evaluate_candidate', side_effect=worker) as evaluate:
                control.evaluate([design, design], proposals=[{}, {}])
                control.evaluate([design], proposals=[{}])
                self.assertEqual(evaluate.call_count, 1)
                control.evaluate([design], proposals=[{'retry': True}])
                with self.assertRaises(ValueError): control.evaluate([design], proposals=[{'retry': True}])
            self.assertEqual(len(control.history), 2)
            self.assertEqual(control.attempts[tuple(design)], 2)
            self.assertEqual(control.history[-1]['attempt'], 2)

    def test_stop_after_current_batch_preserves_best(self):
        with scratch_directory() as path:
            control = self.controller(path)
            def worker(task):
                row = observation(task[0][3])
                row.update(score=-10, hourly={'wind_mw': np.array([16.])}, soc_mwh=np.array([100.,100.]))
                control.kwargs['hpp_deadline'] = time.monotonic()-1
                return row
            with patch(RUNNER+'.evaluate_candidate', side_effect=worker) as evaluate, patch(RUNNER+'.validate_selected', return_value={}):
                with self.assertRaises(BudgetExpired):
                    control.evaluate([observation(100)['design'], observation(200)['design']], proposals=[{}, {}])
                self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(control.best['design'][3], 100)
            self.assertTrue((Path(path)/'best_with_hpp_sizing.json').exists())

    def test_configuration_error_aborts_search(self):
        with scratch_directory() as path:
            control = self.controller(path)
            row = observation(100, 'CONFIGURATION_OR_PROGRAMMING_ERROR')
            row['error'] = 'synthetic invalid parameter'
            with patch(RUNNER+'.evaluate_candidate', return_value=row):
                with self.assertRaisesRegex(RuntimeError, 'invalid parameter'):
                    control.evaluate([row['design']], proposals=[{}])
            self.assertIsNone(control.best)
            self.assertEqual(len(control.history), 1)

    def test_search_without_any_feasible_incumbent_continues(self):
        with scratch_directory() as path:
            control = self.controller(path)
            def worker(task): return observation(task[0][3], 'PROVEN_INFEASIBLE')
            with patch(RUNNER+'.evaluate_candidate', side_effect=worker):
                reason = run_feasibility_aware(control, control.config, control.variables, VARIABLE_NAMES)
            self.assertEqual(reason, 'feasibility_aware_iteration_limit')
            self.assertGreater(len(control.history), control.config['n_doe'])
            self.assertIsNone(control.best)
            self.assertEqual(control.search_summary['training']['cost_training_count'], 0)
            self.assertTrue(all(r['selection_reason'] != 'native_ego' for r in control.history))

    def test_deadline_consumed_by_surrogate_stops_dispatch_submission(self):
        with scratch_directory() as path:
            control = self.controller(path)
            def worker(task): return observation(task[0][3], 'PROVEN_INFEASIBLE')
            def proposal(policy, records, attempts, iteration):
                control.kwargs['hpp_deadline'] = time.monotonic()-1
                return [observation(123)['design']], [{}]
            with patch(RUNNER+'.evaluate_candidate', side_effect=worker) as evaluate, \
                 patch.object(CandidatePolicy, 'propose', autospec=True, side_effect=proposal):
                with self.assertRaises(BudgetExpired):
                    run_feasibility_aware(control, control.config, control.variables, VARIABLE_NAMES)
                self.assertEqual(evaluate.call_count, control.config['n_doe'])
            self.assertNotIn(tuple(observation(123)['design']), control.cache)


if __name__ == '__main__':
    unittest.main()
