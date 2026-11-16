"""Phase 2 tests and opt-in EMS time-budget benchmark (no outer search)."""
import json
from pathlib import Path
import unittest


class NativeMaximumReliabilityTests(unittest.TestCase):
    def setUp(self):
        import shutil
        from uuid import uuid4
        parent = (Path(__file__).resolve().parents[2]/'hpp_sizing_runs/lcoe_reliability_frontier_max_reliability').resolve()
        parent.mkdir(parents=True,exist_ok=True)
        self.directory = (parent/('test_max_reliability_'+uuid4().hex)).resolve()
        assert self.directory.parent == parent and self.directory.name.startswith('test_max_reliability_')
        self.directory.mkdir()
        def cleanup():
            try:
                shutil.rmtree(self.directory)
            except PermissionError:
                # OneDrive/Windows may still hold this new scratch directory.
                # Keep it rather than changing ACLs or touching other results.
                pass
        self.addCleanup(cleanup)

    def candidate(self, failed=0, target=.5, turbines=1, cost=1e7):
        from unittest.mock import patch
        import numpy as np
        from hydesign.tests.test_native_hpp_sizing_base import physical_result
        from hydesign.assembly import hpp_assembly_simple_datacenter_hpp_sizing_base_max_reliability as module
        from hydesign.metrics_simple_datacenter import firm_load_metrics
        physical = physical_result(failed)
        physical.sim_pars['reliability_target'] = target
        h = physical.hourly
        supplied = h['served_facility_mw'].copy()
        supplied[:failed] = 0
        h.update(wind_mw=supplied, renewable_available_mw=supplied,
                 served_facility_mw=supplied, unserved_facility_mw=16-supplied,
                 served_firm_it_mw=supplied/1.15, unserved_firm_it_mw=(16-supplied)/1.15)
        physical.metrics = firm_load_metrics(16-supplied)
        physical.economics.update(annualized_cost_hpp=cost, served_facility_energy_mwh=float(supplied.sum()))
        physical.solver_status.update(feasible=True, binary_failed_hours=failed,
            certificate=dict(failed_hours_lower_bound=0,failed_hours_upper_bound=failed,
                             maximum_availability_certified=failed==0,
                             exact_reliability_lower_bound=1-failed/8760,exact_reliability_upper_bound=1),
            termination='TIME_LIMIT_INCUMBENT')
        model = object.__new__(module.hpp_model_simple_datacenter_hpp_sizing_base_max_reliability)
        model.sim_pars = physical.sim_pars
        model.list_vars = list(module.VARIABLE_NAMES)
        model.list_out_vars = ['cost',module.VALID_NAME,module.objective_name(module.LCOE_OBJECTIVE)]
        design = [35,300,5,turbines,7,180,42,180,1.25,0,8]
        model.variables = {k:dict(var_type='fixed',value=v) for k,v in zip(model.list_vars,design)}
        model.objective = module.LCOE_OBJECTIVE
        model.objective_scale = 1
        model.cost_upper_bound = 30e6
        model.objective_upper_bound = module.objective_bound(model.sim_pars,30e6,module.LCOE_OBJECTIVE)
        model.rejection_score = -model.objective_upper_bound-1
        model.baseline_log_dir = self.directory
        def evaluate(instance,*args):
            instance.__dict__.update(physical.__dict__)
            return np.array([cost])
        context = patch.object(module.hpp_model_simple_datacenter_with_hpp_cost_max_reliability,
                               'evaluate',evaluate)
        return model,design,context

    def test_below_target_retains_physics_economics_bounds_and_archive(self):
        import pandas as pd
        from hydesign import sizing_native_hpp_sizing_base_max_reliability as runner
        model,design,context = self.candidate(failed=2000,target=.9)
        with context:
            out = model.evaluate(*design)
        self.assertEqual(model.candidate_status,'BELOW_TARGET')
        self.assertEqual(out[-1],model.rejection_score)
        record = json.loads(next(self.directory.glob('*.json')).read_text())
        self.assertTrue(record['physical_validation_passed'])
        self.assertFalse(record['outer_feasible'])
        self.assertEqual(record['metrics']['LOLH'],2000)
        self.assertTrue(Path(record['dispatch_path']).exists())
        self.assertIsNotNone(record['solver_status']['certificate'])
        runner.write_evaluation_history([record],self.directory/'history.csv',0,1,reliability_target=.9)
        row = pd.read_csv(self.directory/'history.csv').iloc[0]
        self.assertFalse(row['accepted'])
        self.assertEqual(row['annualized_cost_hpp'],1e7)
        self.assertGreater(row['lcoe_hpp_eur_per_mwh'],0)
        self.assertEqual(row['failed_hours'],2000)
        self.assertFalse(row['maximum_availability_certified'])

    def test_native_ego_and_saved_recovery_choose_lcoe_not_cost(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from hydesign.Parallel_EGO import model_evaluation
        from hydesign import sizing_native_hpp_sizing_base_max_reliability as runner
        scores=[]
        for n,failed,cost in [(1,4000,1e7),(2,0,1.1e7)]:
            model,design,context=self.candidate(failed=failed,turbines=n,cost=cost)
            inputs=dict(hpp_model=lambda **kwargs:model,scaler=SimpleNamespace(inverse_transform=lambda x:x),
                list_vars=model.list_vars,design_vars=model.list_vars,fixed_vars=[],variables=model.variables,
                opt_sign=-1,op_var_index=2)
            with context:
                scores.append(float(model_evaluation((np.array([design]),inputs))))
            self.assertEqual(model.candidate_status,'FEASIBLE')
        self.assertLess(scores[1],scores[0])
        records=[json.loads(p.read_text()) for p in self.directory.glob('*.json')]
        with patch.object(runner,'annual_cost_from_dispatch',return_value=1.1e7), patch(
                'hydesign.ems.ems_simple_datacenter_max_reliability.ems_cplex_simple_datacenter_max_reliability',
                side_effect=AssertionError('Recovery must not resample dispatch')):
            best,errors,_=runner.independently_validate(records,model.sim_pars,self.directory)
        self.assertFalse(errors)
        self.assertEqual(best['design'][3],2)
        self.assertAlmostEqual(best['lcoe_hpp_eur_per_mwh'],1.1e7/140160)

    def test_recovery_rejects_corrupt_archive(self):
        from hydesign import sizing_native_hpp_sizing_base_max_reliability as runner
        model,design,context=self.candidate()
        with context:
            model.evaluate(*design)
        record=json.loads(next(self.directory.glob('*.json')).read_text())
        Path(record['dispatch_path']).write_bytes(b'corrupt')
        best,errors,_=runner.independently_validate([record],model.sim_pars,self.directory)
        self.assertIsNone(best)
        self.assertIn('checksum',errors[0]['error'])

    def test_failed_model_keeps_finite_score_without_stale_results(self):
        from unittest.mock import patch
        import numpy as np
        from hydesign.assembly import hpp_assembly_simple_datacenter_hpp_sizing_base_max_reliability as module
        model,design,context=self.candidate()
        with context:
            model.evaluate(*design)
        with patch.object(module.hpp_model_simple_datacenter_with_hpp_cost_max_reliability,
                          'evaluate',side_effect=ValueError('Invalid dispatch')):
            result=model.evaluate(*design)
        self.assertTrue(np.isfinite(result[-1]))
        self.assertEqual(result[-1],model.rejection_score)
        self.assertIsNone(model.economics)
        self.assertIsNone(model.hourly)
        records=[json.loads(p.read_text()) for p in self.directory.glob('*.json')]
        rejected=next(r for r in records if r['status']!='FEASIBLE')
        self.assertFalse(rejected['physical_validation_passed'])

    def test_below_target_is_not_recovered_as_best(self):
        from hydesign import sizing_native_hpp_sizing_base_max_reliability as runner
        model,design,context=self.candidate(failed=1000,target=.99)
        with context:
            model.evaluate(*design)
        record=json.loads(next(self.directory.glob('*.json')).read_text())
        best,errors,_=runner.independently_validate([record],model.sim_pars,self.directory)
        self.assertIsNone(best)
        self.assertFalse(errors)

    def test_configuration_defaults_and_simple_notebook(self):
        from hydesign import sizing_native_hpp_sizing_base_max_reliability as runner
        cfg=runner.prepare_configuration(runner.ROOT/'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml')
        self.assertEqual(cfg['hpp_sizing_objective'],'minimize_lcoe')
        self.assertEqual(cfg['ems_time_limit_seconds'],30)
        self.assertIn('_max_reliability',cfg['hpp_sizing_output_dir'])
        nb=runner.ROOT/'docs/notebooks/simple_datacenter_lcoe_reliability_with_hpp_sizing_max_reliability.ipynb'
        content=json.loads(nb.read_text())
        code='\n'.join(''.join(c['source']) for c in content['cells'] if c['cell_type']=='code')
        compile(code,str(nb),'exec')
        self.assertIn('reliability_targets = [0.70, 0.90]',code)
        self.assertNotIn('def ',code)
        self.assertLess(len(code.splitlines()),70)

    def test_component_has_no_reliability_target_input(self):
        from hydesign.ems.ems_simple_datacenter_comp_max_reliability import ems_simple_datacenter_comp_max_reliability
        component=ems_simple_datacenter_comp_max_reliability(N_time=3)
        self.assertNotIn('reliability_target',component.input_keys)
        self.assertNotIn('battery_power_fluctuation_penalty',component.input_keys)


def benchmark_ems_time_limits():
    from datetime import datetime, timezone
    import hashlib
    import time
    from uuid import uuid4
    import numpy as np
    import pandas as pd
    from hydesign.ems.ems_simple_datacenter_max_reliability import ems_cplex_simple_datacenter_max_reliability
    root = Path(__file__).resolve().parents[2]
    saved = root / 'hpp_sizing_runs/lcoe_reliability_frontier'
    directory = root / 'hpp_sizing_runs/lcoe_reliability_frontier_max_reliability' / (
        datetime.now(timezone.utc).strftime('benchmark_%Y%m%d_%H%M%SZ_') + uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    sources = [('shared', 'experiment_20261105_131955Z_5fe42393', '080'),
               ('low70', 'experiment_20261105_231423Z_85c2d00e', '070'),
               ('mid90', 'experiment_20261105_231423Z_85c2d00e', '090'),
               ('full100', 'experiment_20261105_231423Z_85c2d00e', '100')]
    rows = []
    for case, experiment, target in sources:
        folder = saved / experiment / ('reliability_' + target)
        source = json.loads((folder/'result.json').read_text())
        cfg, design = source['configuration'], source['best']['design']
        path = next(folder.glob('*/hourly_hpp_sizing_base.csv'))
        h = pd.read_csv(path)
        for limit in (10, 30, 60):
            start = time.perf_counter()
            label = case + '_' + str(limit)
            row = dict(case=case, time_limit_seconds=limit, threads=1, seed=0,
                       source=str(folder), design=design,
                       generation_sha256=hashlib.sha256(h[['wind_mw','pv_mw']].to_numpy().tobytes()).hexdigest())
            print('EMS benchmark:', label, flush=True)
            try:
                result = ems_cplex_simple_datacenter_max_reliability(
                    h.wind_mw.to_numpy(), h.pv_mw.to_numpy(), design[9], design[9]*design[10],
                    cfg['battery_depth_of_discharge'], cfg['battery_charge_efficiency'],
                    cfg['battery_initial_soc'], cfg['facility_load_mw'], cfg['pue'],
                    time_limit_seconds=limit, threads=1)
                status = result[-1]
                row.update(status['certificate'])
                row.update(status['metrics'],
                           physical_validation=True, delivered_energy_mwh=float(result[0].sum()),
                           curtailment_mwh=float(result[1].sum()),
                           battery_throughput_mwh=float((result[4]+result[5]).sum()),
                           binary_failed_hours=status['binary_failed_hours'],
                           cplex_seconds=status['stages'][0]['seconds'],
                           relative_gap=status['stages'][0]['relative_gap'],
                           solver_termination=status['termination'])
                (directory/(label+'_status_max_reliability.json')).write_text(json.dumps(status,indent=2))
                np.savez_compressed(directory/(label+'_dispatch_max_reliability.npz'),
                                    supplied=result[0], soc=result[3], charge=result[4], discharge=result[5], unmet=result[8])
            except Exception as exc:
                row.update(physical_validation=False, error=repr(exc))
            row['wall_seconds'] = time.perf_counter()-start
            rows.append(row)
            (directory/'results_max_reliability.json').write_text(json.dumps(rows,indent=2))
            pd.DataFrame(rows).to_csv(directory/'summary_max_reliability.csv',index=False)
            print(json.dumps(row), flush=True)
    print('BENCHMARK_DIRECTORY', directory, flush=True)
    return directory


if __name__ == '__main__':
    import sys
    if '--benchmark' in sys.argv:
        benchmark_ems_time_limits()
    else:
        unittest.main()
