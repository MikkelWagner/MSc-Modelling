"""Native refinement tests and synthetic pipeline smoke tests; no CPLEX solves."""
import itertools
import json
from contextlib import contextmanager
from pathlib import Path
import shutil
import stat
import time
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from hydesign import Parallel_EGO as native
from hydesign.metrics_simple_datacenter import firm_load_metrics
from hydesign.search_simple_datacenter_with_hpp_sizing import (
    FeasibilitySurrogate, NativeRefinement, SearchSpace, UNKNOWN_STATUSES,
    run_feasibility_aware, search_settings, training_data,
)
from hydesign.sizing_simple_datacenter_with_hpp_sizing import (
    BudgetExpired, EvaluationLimitReached, RunController, SIZABLE, VARIABLE_NAMES,
    prepare_variables, run_hpp_sizing, unique_evaluation_limit, validate_selected,
)
from hydesign.tests.test_search_simple_datacenter_with_hpp_sizing import (
    CONFIG, RUNNER, SEARCH, configuration, observation,
)
from hydesign.tests.test_simple_datacenter_with_hpp_sizing import (
    ImmediateExecutor,
)


@contextmanager
def scratch_directory():
    root = Path(__file__).resolve().parents[2]
    path = root / ('native_refinement_' + uuid.uuid4().hex + '_with_hpp_sizing')
    path.mkdir()
    try:
        yield str(path)
    finally:
        assert path.resolve().parent == root.resolve()
        def clear_readonly(function, failed_path, _error):
            failed = Path(failed_path).resolve()
            assert failed == path.resolve() or path.resolve() in failed.parents
            failed.chmod(stat.S_IWRITE)
            function(failed_path)
        # OneDrive may mark temporary files/directories read-only or briefly
        # hold a just-closed output directory. Only clean this test's own tree.
        for attempt in range(5):
            try:
                shutil.rmtree(path, onerror=clear_readonly)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.1)


def settings(four=False, **overrides):
    config = configuration()
    config.update(optimization_method='feasibility_aware', fa_native_refinement_enabled=True,
                  fa_native_refinement_interval=1, max_iter=8)
    if four:
        for name in SIZABLE:
            config['variables'][name]['var_type'] = 'design'
    config.update(overrides)
    return config


def synthetic_result(design, config, status=None, failed_hours=0):
    """Analytic test objective and invented dispatch, NOT a physical HPP run.

    Feasible test records still pass the real annual dispatch/reliability checker.
    The synthetic availability threshold is unrelated to either thesis benchmark.
    """
    count, pv, power, hours = np.asarray(design)[[3, 5, 9, 10]]
    adequate = count + .2 * pv + .02 * power * hours >= 130
    status = status or ('FEASIBLE' if adequate else 'PROVEN_INFEASIBLE')
    cost = 1e6 + count * 10000 + pv * 2000 + power * 1000 + power * hours * 200
    row = observation(count, status, cost)
    row['design'] = list(design)
    row['score'] = -cost / 1e6 if row['accepted'] else -1e10
    if row['accepted']:
        n, load, pue = 8760, config['facility_load_mw'], config['pue']
        demand = np.full(n, load)
        unmet = np.zeros(n)
        unmet[:failed_hours] = load
        served = demand - unmet
        zeros = np.zeros(n)
        row.update(
            hourly=dict(wind_mw=served, pv_mw=zeros, renewable_available_mw=served,
                        charge_mw=zeros, discharge_mw=zeros, curtailment_mw=zeros,
                        firm_demand_it_mw=demand / pue, served_firm_it_mw=served / pue,
                        unserved_firm_it_mw=unmet / pue, facility_load_mw=demand,
                        served_facility_mw=served, unserved_facility_mw=unmet),
            soc_mwh=np.full(n + 1, power * hours * config['battery_initial_soc']),
            solver_status={'feasible': True, 'backend': 'synthetic_test_only'},
            metrics=firm_load_metrics(unmet, epsilon_mw=config['epsilon_mw']),
        )
    return row


class RefinementTests(unittest.TestCase):
    def policy(self, config=None):
        c = config if config is not None else settings()
        return NativeRefinement(SearchSpace(prepare_variables(c), VARIABLE_NAMES), c)

    def test_disabled_and_schedule_do_not_call_native_helpers(self):
        for c, iteration in [(settings(fa_native_refinement_enabled=False), 0),
                             (settings(fa_native_refinement_interval=5), 3)]:
            with patch(SEARCH + '.native_ego.extreme_around_point') as helper:
                self.assertEqual(self.policy(c).propose(observation(200), {}, iteration), ([], []))
                helper.assert_not_called()

    def test_only_verified_incumbents_with_genuine_cost(self):
        p = self.policy()
        rows = [None, observation(0, 'PROVEN_INFEASIBLE')]
        rows += [observation(200, status) for status in UNKNOWN_STATUSES]
        rows.append(dict(observation(200), accepted=False))
        for row in rows:
            self.assertEqual(p.propose(row, {}, 0), ([], []))
        with self.assertRaises(ValueError):
            p.propose(observation(200, cost=np.nan), {}, 0)
        self.assertEqual(p.diagnostics['events'], 0)

    def test_one_variable_extremes_then_native_perturbation(self):
        p, best = self.policy(), observation(200)
        with patch(SEARCH + '.native_ego.extreme_around_point', wraps=native.extreme_around_point) as helper:
            designs, meta = p.propose(best, {}, 0)
            helper.assert_called_once()
            np.testing.assert_allclose(helper.call_args.args[0], [[.5]])
        self.assertEqual([x[3] for x in designs], [0, 400])
        self.assertTrue(all(m['selection_reason'] == 'native_refinement_extreme' for m in meta))
        expected_step = np.random.RandomState(1).uniform(.05, .25)
        with patch(SEARCH + '.native_ego.perturbe_around_point', wraps=native.perturbe_around_point) as helper:
            designs, meta = p.propose(best, {}, 1)
            helper.assert_called_once()
        self.assertEqual([x[3] for x in designs],
                         [round(200 + expected_step * 400), round(200 - expected_step * 400)])
        self.assertEqual(meta[0]['refinement_step'], expected_step)
        self.assertEqual(meta[0]['selection_reason'], 'native_refinement_perturbation')

    def test_all_active_subsets_preserve_bounds_and_fixed_values(self):
        for mask in itertools.product((False, True), repeat=4):
            c = settings(fa_batch_size=8)
            for name, active in zip(SIZABLE, mask):
                c['variables'][name]['var_type'] = 'design' if active else 'fixed'
            p = self.policy(c)
            best = observation(20)
            cache = {tuple(best['design']): best}
            for iteration in (0, 1):
                designs, meta = p.propose(best, cache, iteration)
                self.assertEqual(len({tuple(x) for x in designs}), len(designs))
                for x in designs:
                    self.assertNotIn(tuple(x), cache)
                    self.assertEqual(x[3], round(x[3]))
                    self.assertEqual(sum(a != b for a, b in zip(x, best['design'])), 1)
                    for i, name in enumerate(VARIABLE_NAMES):
                        if name in p.space.active:
                            self.assertGreaterEqual(x[i], p.space.variables[name]['limits'][0])
                            self.assertLessEqual(x[i], p.space.variables[name]['limits'][1])
                        else:
                            self.assertEqual(x[i], p.space.variables[name]['value'])

    def test_coordinate_rotation_does_not_starve_later_variables(self):
        p = self.policy(settings(four=True))
        best = observation(200)
        seen = set()
        for i in range(4):
            designs, meta = p.propose(best, {}, i)
            self.assertLessEqual(len(designs), 2)
            seen.update(m['refinement_coordinate'] for m in meta)
        self.assertEqual(seen, set(SIZABLE))

    def test_no_surrogate_gate_and_exact_untouched_continuous_coordinates(self):
        p = self.policy(settings(four=True, fa_batch_size=8))
        best = observation(137)
        best['design'][5] = 113.72314159265359
        best['design'][-2:] = [37.12345678912345, 13.718281828459045]
        with patch.object(FeasibilitySurrogate, 'predict', side_effect=AssertionError('Refinement must not consult predictions')):
            designs, _ = p.propose(best, {}, 0)
            for x in designs:
                self.assertEqual(sum(a != b for a, b in zip(x, best['design'])), 1)

    def test_cached_unknowns_and_clipped_duplicates_are_skipped(self):
        c = settings()
        c['variables']['Nwt'].update(value=0, limits=[0, 1])
        p = self.policy(c)
        best = observation(0)
        cache = {tuple(best['design']): best}
        designs, _ = p.propose(best, cache, 0)
        self.assertEqual([x[3] for x in designs], [1])
        cache[tuple(designs[0])] = observation(1, 'TIME_LIMIT_UNRESOLVED')
        self.assertEqual(p.propose(best, cache, 1), ([], []))
        self.assertGreater(p.diagnostics['skipped_cached_or_duplicate'], 0)

    def test_latest_center_and_improvement_restore_extremes(self):
        p = self.policy()
        p.propose(observation(200, cost=10e6), {}, 0)
        improved = observation(173, cost=9e6)
        improved['evaluation_index'] = 12
        _, meta = p.propose(improved, {}, 1)
        self.assertEqual(meta[0]['refinement_center_design'], improved['design'])
        self.assertEqual(meta[0]['refinement_center_evaluation_index'], 12)
        self.assertTrue(meta[0]['refinement_center_changed'])
        self.assertEqual(meta[0]['selection_reason'], 'native_refinement_extreme')
        self.assertEqual(p.diagnostics['center_updates'], 1)

    def test_relative_tolerance_zero_cost_and_random_state(self):
        p = self.policy()
        p.propose(observation(200, cost=10e6), {}, 0)
        np.random.seed(123)
        expected = np.random.random(4)
        np.random.seed(123)
        _, meta = p.propose(observation(200, cost=10e6 - 1), {}, 1)
        np.testing.assert_array_equal(np.random.random(4), expected)
        self.assertEqual(meta[0]['selection_reason'], 'native_refinement_perturbation')
        p.propose(observation(200, cost=0), {}, 2)
        _, meta = p.propose(observation(200, cost=0), {}, 3)
        self.assertEqual(meta[0]['selection_reason'], 'native_refinement_perturbation')

    def test_invalid_settings_fail_before_running(self):
        for key, bad in [('fa_native_refinement_enabled', 'false'),
                         ('fa_native_refinement_enabled', 1),
                         ('fa_native_refinement_interval', 0),
                         ('fa_native_refinement_interval', 1.5),
                         ('fa_native_refinement_interval', True),
                         ('fa_native_refinement_interval', '5'),
                         ('fa_native_refinement_interval', np.nan),
                         ('tol', -1), ('tol', np.inf), ('tol', 'tiny')]:
            with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                search_settings(settings(**{key: bad}))
        for bad in [0, -1, 2.5, True, np.inf, '216']:
            with self.assertRaises(ValueError):
                unique_evaluation_limit({'hpp_sizing_max_unique_evaluations': bad})
        self.assertIsNone(unique_evaluation_limit({}))
        self.assertEqual(unique_evaluation_limit({'hpp_sizing_max_unique_evaluations': 216}), 216)


class PipelineTests(unittest.TestCase):
    def controller(self, path, config=None):
        c = settings() if config is None else config
        return RunController(c, prepare_variables(c), {'hpp_deadline': time.monotonic() + 60},
                             Path(path), time.monotonic(), ImmediateExecutor())

    def test_refinement_results_keep_statuses_validation_and_best(self):
        with scratch_directory() as path:
            control = self.controller(path)
            p = NativeRefinement(SearchSpace(control.variables, VARIABLE_NAMES), control.config)
            control.search_summary['native_refinement'] = p.diagnostics
            center = observation(200)['design']
            with patch(RUNNER + '.evaluate_candidate', side_effect=lambda task: synthetic_result(task[0], control.config)):
                control.evaluate([center], proposals=[{}])
            first = control.best
            for i, status in enumerate(['PROVEN_INFEASIBLE', 'TIME_LIMIT_UNRESOLVED', 'NUMERICAL_ERROR', 'FEASIBLE']):
                design = observation(190 - i)['design']
                with patch(RUNNER + '.evaluate_candidate', side_effect=lambda task, s=status: synthetic_result(task[0], control.config, s)):
                    control.evaluate([design], proposals=[{'selection_reason': 'native_refinement_perturbation'}])
                if status != 'FEASIBLE':
                    self.assertIs(control.best, first)
            self.assertEqual(control.best['design'][3], 187)
            self.assertEqual(p.diagnostics['unique_evaluations'], 4)
            self.assertEqual(p.diagnostics['feasible_evaluations'], 1)
            self.assertEqual(p.diagnostics['proven_infeasible_evaluations'], 1)
            self.assertEqual(p.diagnostics['unknown_evaluations'], 2)
            self.assertEqual(p.diagnostics['incumbent_improvements'], 1)
            _, costs, _, labels = training_data(list(control.cache.values()), p.space)
            self.assertEqual(len(costs), 2)
            self.assertEqual(sorted(labels.tolist()), [0, 1, 1])

    def test_refinement_cannot_bypass_hourly_reliability_validation(self):
        with scratch_directory() as path:
            control = self.controller(path)
            design = observation(200)['design']
            invalid = synthetic_result(design, control.config, 'FEASIBLE', failed_hours=88)
            with patch(RUNNER + '.evaluate_candidate', return_value=invalid):
                with self.assertRaisesRegex(ValueError, 'hourly reliability'):
                    control.evaluate([design], proposals=[{'selection_reason': 'native_refinement_extreme'}])
            self.assertIsNone(control.best)
            self.assertFalse(control.cache)

    def test_partial_refinement_batch_survives_deadline(self):
        with scratch_directory() as path:
            control = self.controller(path)
            p = NativeRefinement(SearchSpace(control.variables, VARIABLE_NAMES), control.config)
            control.search_summary['native_refinement'] = p.diagnostics
            designs, meta = p.propose(observation(200), {}, 0)
            def worker(task):
                control.kwargs['hpp_deadline'] = time.monotonic() - 1
                return synthetic_result(task[0], control.config)
            with patch(RUNNER + '.evaluate_candidate', side_effect=worker) as evaluate:
                with self.assertRaises(BudgetExpired):
                    control.evaluate(designs, proposals=meta)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(p.diagnostics['unique_evaluations'], 1)
            self.assertEqual(len(control.history), 1)

    def test_deadline_after_generation_prevents_refinement_submission(self):
        with scratch_directory() as path:
            control = self.controller(path)
            original = NativeRefinement.propose
            def expire(policy, best, records, iteration):
                answer = original(policy, best, records, iteration)
                control.kwargs['hpp_deadline'] = time.monotonic() - 1
                return answer
            with patch(RUNNER + '.evaluate_candidate', side_effect=lambda task: synthetic_result(task[0], control.config)), \
                 patch.object(NativeRefinement, 'propose', autospec=True, side_effect=expire):
                with self.assertRaises(BudgetExpired):
                    run_feasibility_aware(control, control.config, control.variables, VARIABLE_NAMES)
            self.assertFalse(any(r['selection_reason'].startswith('native_refinement_') for r in control.history))

    def test_unique_cap_has_no_parallel_overshoot_and_cache_is_free(self):
        with scratch_directory() as path:
            control = self.controller(path, settings(n_procs=4, hpp_sizing_max_unique_evaluations=3))
            designs = [observation(i)['design'] for i in (140, 150, 160, 170, 180)]
            with patch(RUNNER + '.evaluate_candidate', side_effect=lambda task: synthetic_result(task[0], control.config)) as worker:
                with self.assertRaises(EvaluationLimitReached):
                    control.evaluate(designs, proposals=[{} for _ in designs])
                self.assertEqual(worker.call_count, 3)
                self.assertEqual(len(control.cache), 3)
                control.evaluate(designs[:3], proposals=[{}, {}, {}])
                self.assertEqual(worker.call_count, 3)

    def test_retries_consume_time_and_attempts_but_not_another_unique_slot(self):
        with scratch_directory() as path:
            control = self.controller(path, settings(hpp_sizing_max_unique_evaluations=2))
            x = observation(100)['design']
            with patch(RUNNER + '.evaluate_candidate', side_effect=lambda task: synthetic_result(task[0], control.config, 'TIME_LIMIT_UNRESOLVED')):
                control.evaluate([x], proposals=[{}])
                control.evaluate([x], proposals=[{'retry': True}])
                with self.assertRaises(EvaluationLimitReached):
                    control.evaluate([observation(200)['design']], proposals=[{}])
            self.assertEqual(len(control.cache), 2)
            self.assertEqual(len(control.history), 3)

    def test_no_feasible_incumbent_still_runs_global_search(self):
        with scratch_directory() as path:
            control = self.controller(path)
            with patch(RUNNER + '.evaluate_candidate', side_effect=lambda task: synthetic_result(task[0], control.config, 'PROVEN_INFEASIBLE')):
                reason = run_feasibility_aware(control, control.config, control.variables, VARIABLE_NAMES)
            self.assertEqual(reason, 'feasibility_aware_iteration_limit')
            self.assertIsNone(control.best)
            self.assertEqual(control.search_summary['native_refinement']['events'], 0)
            self.assertGreater(len(control.history), control.config['n_doe'])

    def test_disabled_matches_missing_setting_and_never_refines(self):
        histories = []
        for explicit in (False, True):
            with scratch_directory() as path:
                c = settings(fa_native_refinement_enabled=False)
                if not explicit:
                    c.pop('fa_native_refinement_enabled')
                    c.pop('fa_native_refinement_interval')
                control = self.controller(path, c)
                with patch(RUNNER + '.evaluate_candidate', side_effect=lambda task: synthetic_result(task[0], c)), \
                     patch.object(NativeRefinement, 'propose') as refine:
                    run_feasibility_aware(control, c, control.variables, VARIABLE_NAMES)
                refine.assert_not_called()
                histories.append([(r['Nwt'], r['selection_reason'], r['status'], r['annualized_cost_hpp']) for r in control.history])
                self.assertFalse(control.search_summary['native_refinement_enabled'])
        self.assertEqual(histories[0], histories[1])

    def test_full_runner_smoke_wind_and_four_variables(self):
        for four in (False, True):
            with self.subTest(four=four), scratch_directory() as path:
                c = settings(four=four, fa_native_refinement_interval=2,
                             hpp_sizing_output_dir=path, hpp_sizing_max_unique_evaluations=24)
                with patch(RUNNER + '.ProcessPoolExecutor') as pool, \
                     patch(RUNNER + '.evaluate_candidate', side_effect=lambda task: synthetic_result(task[0], c)), \
                     patch(RUNNER + '.validate_selected', wraps=validate_selected) as validate:
                    pool.return_value.__enter__.return_value = ImmediateExecutor()
                    result = run_hpp_sizing(CONFIG, c)
                self.assertIsNone(result['error'], result['error'])
                self.assertIn(result['termination_reason'], ('unique_evaluation_limit', 'feasibility_aware_iteration_limit'))
                self.assertLessEqual(result['unique_designs'], 24)
                self.assertIsNotNone(result['best'])
                self.assertGreater(validate.call_count, 1)
                self.assertEqual(result['final_validation']['metrics']['LOLH'], 0)
                summary = result['search_summary']['native_refinement']
                self.assertGreater(summary['unique_evaluations'], 0)
                self.assertGreater(summary['generation_seconds'], 0)
                self.assertGreater(summary['evaluation_wall_seconds'], 0)
                proposals = [json.loads(line) for line in Path(result['proposals_path']).read_text().splitlines()]
                refined = [p for p in proposals if p['selection_reason'].startswith('native_refinement_')]
                self.assertEqual(len(refined), summary['unique_evaluations'])
                self.assertTrue(all('refinement_center_design' in p for p in refined))
                first_refinement = proposals.index(refined[0])
                self.assertTrue(any(not p['selection_reason'].startswith('native_refinement_')
                                    for p in proposals[first_refinement + 1:]))
                saved = json.loads((Path(result['run_directory']) / 'result_with_hpp_sizing.json').read_text())
                self.assertEqual(saved['native_refinement_enabled'], True)
                print('SYNTHETIC SMOKE', json.dumps(dict(active_variables=4 if four else 1,
                      termination=result['termination_reason'], unique=result['unique_designs'],
                      elapsed_seconds=result['elapsed_seconds'], refinement=summary)), flush=True)


if __name__ == '__main__':
    unittest.main()
