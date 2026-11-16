"""Short real-process timeout tests; no EGO/CPLEX optimization in this suite."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from hydesign import sizing_native_hpp_sizing_graded_max_reliability as runner
from hydesign import process_control_graded_max_reliability as control

MODULE='hydesign.tests.test_timeout_graded_max_reliability'


def _blocked(directory):
    (Path(directory)/f'worker_{os.getpid()}.json').write_text(json.dumps(dict(pid=os.getpid())))
    time.sleep(300)


def _child(mode,gate,directory):
    directory=Path(directory)
    while not Path(gate).exists(): time.sleep(.01)
    (directory/'ready.json').write_text(json.dumps(dict(pid=os.getpid())))
    if mode=='pool':
        import multiprocessing
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        (directory/'external.json').write_text(json.dumps(dict(pid=child.pid)))
        with multiprocessing.Pool(2) as pool: pool.map(_blocked,[str(directory)]*2)
    elif mode=='hang': time.sleep(300)
    elif mode=='normal':
        (directory/'completion_hpp_sizing_base.json').write_text(json.dumps(dict(status='NATIVE_COMPLETED')))


class TimeoutTests(unittest.TestCase):
    def setUp(self):
        parent=runner.ROOT/'hpp_sizing_runs/timeout_checks_graded_max_reliability'
        parent.mkdir(exist_ok=True)
        self.directory=(parent/uuid4().hex).resolve()
        self.directory.mkdir()
        def cleanup():
            assert self.directory.parent==parent.resolve() and len(self.directory.name)==32
            target=Path('\\\\?\\'+str(self.directory)) if os.name=='nt' else self.directory
            try: shutil.rmtree(target)
            except PermissionError: pass
        self.addCleanup(cleanup)

    def command(self,mode,gate,directory=None):
        return [sys.executable,'-B','-m',MODULE,'--child',mode,str(gate),str(directory or self.directory)]

    def run_child(self,mode,budget=1.5,**kwargs):
        gate=self.directory/'start.flag'
        return control.run_contained(self.command(mode,gate),self.directory,budget,gate,
            cleanup_seconds=3,heartbeat_seconds=.2,**kwargs)

    def test_A_normal_completion(self):
        r=self.run_child('normal',5)
        self.assertEqual(r['termination_reason'],'NORMAL_EXIT')
        self.assertEqual(r['returncode'],0)
        self.assertTrue(r['process_cleanup']['verified'])
        self.assertLess(r['optimization_seconds'],5)
        _,cfg=self.configuration()
        candidate,_=self.save_candidate(cfg)
        recovered=self.recover(cfg,r)
        self.assertEqual(recovered['final_status'],'VALIDATED_FEASIBLE_INCUMBENT')
        self.assertAlmostEqual(recovered['best']['lcoe_hpp_eur_per_mwh'],candidate['lcoe_hpp_eur_per_mwh'])

    def test_B_unresponsive_optimizer(self):
        start=time.monotonic()
        r=self.run_child('hang')
        self.assertEqual(r['termination_reason'],'TIME_LIMIT')
        self.assertLess(time.monotonic()-start,3)
        self.assertTrue(r['process_cleanup']['verified'])

    def test_C_hanging_pool_and_descendant_tree(self):
        import psutil
        r=self.run_child('pool',3)
        self.assertEqual(r['termination_reason'],'TIME_LIMIT')
        self.assertTrue(r['process_cleanup']['verified'])
        paths=[self.directory/'ready.json',self.directory/'external.json']+list(self.directory.glob('worker_*.json'))
        self.assertGreaterEqual(len(paths),4)
        for path in paths:
            pid=json.loads(path.read_text())['pid']
            self.assertFalse(psutil.pid_exists(pid),f'Orphan {pid}')
        if os.name=='nt': self.assertEqual(r['process_cleanup']['active_processes_after_stop'],0)

    def test_sleep_elapsed_and_backwards_clock(self):
        with patch.object(control.time,'time',return_value=100) as wall, patch.object(control.time,'monotonic',return_value=20) as mono:
            d=control.Deadline(10)
            wall.return_value=1000  # Simulate resume: no active CPU time passed.
            self.assertEqual(d.remaining,0)
            wall.return_value=50
            mono.return_value=31  # A wall-clock correction cannot extend a run.
            self.assertEqual(d.remaining,0)

    def test_slow_telemetry_cannot_block_deadline(self):
        import threading
        release=threading.Event()
        try:
            with patch.object(control,'_telemetry',side_effect=lambda *args:release.wait(30)):
                started=time.monotonic()
                r=self.run_child('hang',.8)
                self.assertLess(time.monotonic()-started,2)
                self.assertTrue(r['process_cleanup']['verified'])
        finally: release.set()

    @unittest.skipUnless(os.name=='nt','Windows job cleanup')
    def test_cleanup_has_its_own_bound(self):
        import win32job
        original=win32job.QueryInformationJobObject
        def still_active(job,kind):
            value=original(job,kind)
            if kind==win32job.JobObjectBasicAccountingInformation: value['ActiveProcesses']=1
            return value
        gate=self.directory/'gate.flag'
        with patch.object(win32job,'QueryInformationJobObject',side_effect=still_active):
            started=time.monotonic()
            r=control.run_contained(self.command('hang',gate),self.directory,.5,gate,cleanup_seconds=.3)
        self.assertLess(time.monotonic()-started,1.6)
        self.assertFalse(r['process_cleanup']['verified'])
        self.assertTrue(r['process_cleanup']['root_stopped'])

    def configuration(self,**overrides):
        path=runner.ROOT/'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml'
        settings=dict(hpp_sizing_time_limit_seconds=1.,hpp_sizing_output_dir=str(self.directory),
            hpp_sizing_cleanup_seconds=3.,hpp_sizing_recovery_time_limit_seconds=25.,
            hpp_sizing_reporting_time_limit_seconds=25.,hpp_sizing_heartbeat_seconds=.2,
            reliability_target=.9)
        settings.update(overrides)
        return path,runner.prepare_configuration(path,settings)

    def save_candidate(self,cfg,count=20,failed=0,status='FEASIBLE'):
        import numpy as np
        from hydesign.tests.test_native_hpp_sizing_base import physical_result
        from hydesign.metrics_simple_datacenter import firm_load_metrics
        from hydesign.sizing_native_hpp_sizing_base_max_reliability import annual_cost_from_dispatch
        physical=physical_result(failed)
        design=[35,300,5,count,7,180,42,180,1.25,0,8]
        cost=annual_cost_from_dispatch(cfg,design,physical.hourly['wind_mw'])
        energy=float(physical.hourly['served_facility_mw'].sum())
        economics=dict(CAPEX_hpp=1e8,annualized_cost_hpp=cost,served_facility_energy_mwh=energy)
        metrics=firm_load_metrics(physical.hourly['unserved_facility_mw'])
        destination=self.directory/'evaluations'
        destination.mkdir(exist_ok=True)
        file=destination/(uuid4().hex+'_hpp_sizing_base.json')
        archive=file.with_suffix('.npz')
        np.savez_compressed(archive,**physical.hourly,soc_mwh=physical.soc_mwh)
        record=dict(design=design,status=status,started_at=time.time(),completed_at=time.time(),
            score=-cost/energy,lcoe_hpp_eur_per_mwh=cost/energy,annualized_cost_hpp=cost,
            economics=economics,metrics=metrics,validation=dict(metrics=metrics),
            physical_validation_passed=True,outer_feasible=status=='FEASIBLE',solver_status=physical.solver_status,
            dispatch_path=str(archive),dispatch_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        file.write_text(json.dumps(record))
        return record,file

    def recover(self,cfg,outcome):
        request=dict(configuration=cfg,physical_inputs=cfg,run_directory=str(self.directory),outcome=outcome)
        result=runner._run_phase('recover',request,cfg,self.directory)
        self.assertEqual(result['termination_reason'],'NORMAL_EXIT')
        return json.loads((self.directory/'result_hpp_sizing_base.json').read_text())

    def test_D_timeout_recovers_cheapest_validated_lcoe_and_skips_corruption(self):
        _,cfg=self.configuration()
        best,source=self.save_candidate(cfg,20)
        self.save_candidate(cfg,40)
        damaged,_=self.save_candidate(cfg,10)
        Path(damaged['dispatch_path']).write_bytes(b'bad archive')
        (self.directory/'evaluations/truncated_hpp_sizing_base.json').write_text('{')
        before=source.read_bytes()
        outcome=self.run_child('hang',.6)
        r=self.recover(cfg,outcome)
        self.assertEqual(r['final_status'],'VALIDATED_FEASIBLE_INCUMBENT')
        self.assertEqual(r['best']['design'][3],20)
        self.assertAlmostEqual(r['best']['lcoe_hpp_eur_per_mwh'],best['lcoe_hpp_eur_per_mwh'])
        self.assertTrue(r['checkpoint_errors'])
        self.assertTrue(r['validation_errors'])
        self.assertEqual(source.read_bytes(),before)

    def test_E_timeout_without_feasible_and_interrupted_checkpoint(self):
        _,cfg=self.configuration()
        self.save_candidate(cfg,20,failed=2000,status='BELOW_TARGET')
        file=self.directory/'evaluations/running_hpp_sizing_base.json'
        file.write_text(json.dumps(dict(design=[0]*11,status='RUNNING',started_at=time.time())))
        before=file.read_bytes()
        r=self.recover(cfg,self.run_child('hang',.6))
        self.assertEqual(r['final_status'],'NO_VALIDATED_FEASIBLE_SOLUTION_FOUND')
        self.assertIsNone(r['best'])
        self.assertEqual(r['unresolved_evaluations'],1)
        self.assertEqual(file.read_bytes(),before)

    def test_recovery_itself_is_bounded(self):
        _,cfg=self.configuration(hpp_sizing_recovery_time_limit_seconds=.5)
        # Simulates a blocked recovery worker without changing recovery mathematics.
        actual=runner.run_contained
        def hanging(command,directory,limit,gate,**kwargs):
            return actual(self.command('hang',gate,directory),directory,limit,gate,**kwargs)
        with patch.object(runner,'run_contained',side_effect=hanging):
            outcome=runner._run_phase('recover',{},cfg,self.directory)
        self.assertEqual(outcome['termination_reason'],'TIME_LIMIT')
        self.assertTrue(outcome['process_cleanup']['verified'])

    def test_reporting_itself_is_bounded(self):
        _,cfg=self.configuration(hpp_sizing_reporting_time_limit_seconds=.5)
        actual=runner.run_contained
        def hanging(command,directory,limit,gate,**kwargs):
            return actual(self.command('hang',gate,directory),directory,limit,gate,**kwargs)
        with patch.object(runner,'run_contained',side_effect=hanging):
            outcome=runner._run_phase('report',{},cfg,self.directory)
        self.assertEqual(outcome['termination_reason'],'TIME_LIMIT')
        self.assertTrue(outcome['process_cleanup']['verified'])

    def test_F_sequence_continues_after_hanging_target(self):
        path,cfg=self.configuration()
        for v in ('Nwt','solar_MW [MW AC]','b_P [MW]','b_E_h [h]'):cfg['variables'][v]['var_type']='design'
        def commands(request,gate):
            r=json.loads(Path(request).read_text())
            mode='hang' if r['configuration']['reliability_target']==.7 else 'normal'
            return self.command(mode,gate,Path(request).parent)
        with patch.object(runner,'_search_command',side_effect=commands):
            directory=runner.run_lcoe_reliability_experiment(path,cfg,[.7,.9],self.directory)
        results=[json.loads((directory/f'reliability_{label}'/'result.json').read_text()) for label in ('070','090')]
        self.assertEqual(results[0]['termination_reason'],'TIME_LIMIT')
        self.assertEqual(results[1]['termination_reason'],'MAX_ITERATIONS')
        self.assertTrue(all(r['recovery_complete'] for r in results))
        self.assertTrue((directory/'summary.csv').exists())

    def test_G_resume_preserves_completed_results_and_restarts_incomplete(self):
        path,cfg=self.configuration()
        for v in ('Nwt','solar_MW [MW AC]','b_P [MW]','b_E_h [h]'):cfg['variables'][v]['var_type']='design'
        source=self.directory/'old';source.mkdir()
        cases=[dict(reliability_target=.4,folder='reliability_040'),dict(reliability_target=.7,folder='reliability_070')]
        (source/'experiment_config.json').write_text(json.dumps(dict(configuration=cfg,config_path=str(path),cases=cases)))
        done=source/'reliability_040';done.mkdir()
        (done/'result.json').write_text(json.dumps(dict(best=None,configuration=cfg,
            final_status='NO_VALIDATED_FEASIBLE_SOLUTION_FOUND',termination_reason='TIME_LIMIT')))
        incomplete=source/'reliability_070';incomplete.mkdir();(incomplete/'untouched.txt').write_text('saved work')
        before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
        calls=[]
        def commands(request,gate):
            r=json.loads(Path(request).read_text());calls.append(r['configuration']['reliability_target'])
            return self.command('normal',gate,Path(request).parent)
        with patch.object(runner,'_search_command',side_effect=commands):
            destination=runner.resume_lcoe_reliability_experiment(source,self.directory)
        self.assertEqual(calls,[.7])
        self.assertNotEqual(destination,source)
        self.assertEqual((destination/'reliability_040/result.json').read_bytes(),before[str(done/'result.json')])
        self.assertEqual(before,{str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_resume_refuses_changed_model_assumptions(self):
        path,cfg=self.configuration()
        source=self.directory/'old';source.mkdir()
        (source/'experiment_config.json').write_text(json.dumps(dict(configuration=cfg)))
        changed=dict(cfg,facility_load_mw=17)
        with self.assertRaisesRegex(ValueError,'saved model/search assumptions'):
            runner.run_lcoe_reliability_experiment(path,changed,[.7],self.directory,resume_from=source)
        with self.assertRaisesRegex(ValueError,'execution controls only'):
            runner.resume_lcoe_reliability_experiment(source,execution_overrides={'ems_time_limit_seconds':60})


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--child': _child(*sys.argv[2:])
    else: unittest.main()
