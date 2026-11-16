"""Joint objective/recovery tests; opt-in --integration and --smoke use CPLEX."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from hydesign.assembly import hpp_assembly_simple_datacenter_joint_hpp_dc_sizing as adapter
from hydesign import sizing_native_joint_hpp_dc_sizing as runner
from hydesign.metrics_simple_datacenter import firm_load_metrics

CONFIG=runner.ROOT/'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml'


def configuration():
    cfg=runner.prepare_configuration(CONFIG,dict(reliability_target=.7,pue=1.15))
    for name in ('Nwt','solar_MW [MW AC]','b_P [MW]','b_E_h [h]'):
        cfg['variables'][name]['var_type']='design'
    return cfg


class JointTests(unittest.TestCase):
    def setUp(self):
        self.cfg=configuration()
        # Ranking fixtures must not depend on the user's current YAML revenue scenario.
        self.cfg['datacenter_firm_revenue_per_mwh_it']=4000
        self.parent=(runner.ROOT/'hpp_sizing_runs/joint_hpp_dc_sizing').resolve()
        self.directory=self.parent/('test_'+uuid4().hex)
        self.directory.mkdir(parents=True)
        def cleanup():
            import shutil
            assert self.directory.resolve().parent==self.parent and self.directory.name.startswith('test_')
            try: shutil.rmtree(self.directory)
            except PermissionError: pass
        self.addCleanup(cleanup)
        self.bounds=adapter.economic_bounds(self.cfg,self.cfg['variables'])

    def candidate(self,load=16,failed=0,turbines=20,solar=180):
        from hydesign.tests.test_native_hpp_sizing_base import physical_result
        from hydesign.sizing_native_hpp_sizing_base_max_reliability import annual_cost_from_dispatch
        physical=physical_result(0)
        physical.sim_pars=dict(self.cfg,facility_load_mw=load)
        h=physical.hourly
        supplied=np.full(8760,float(load)); supplied[:failed]=0
        h.update(wind_mw=supplied,renewable_available_mw=supplied,facility_load_mw=np.full(8760,load),
            served_facility_mw=supplied,unserved_facility_mw=load-supplied,
            firm_demand_it_mw=np.full(8760,load/1.15),served_firm_it_mw=supplied/1.15,
            unserved_firm_it_mw=(load-supplied)/1.15)
        d=[35,300,5,turbines,7,solar,42,180,1.25,0,8,load/1.15]
        cost=annual_cost_from_dispatch(physical.sim_pars,d[:-1],supplied)
        dc=adapter.datacenter_cost_annualized(**{k:self.cfg[k] for k in adapter.DC_COST_PARAMETERS}).compute(d[-1])
        physical.economics.update(dict(zip(('CAPEX_dc_facility','CAPEX_dc_it','CAPEX_dc_total','OPEX_dc',
            'annualized_CAPEX_dc_facility','annualized_CAPEX_dc_it','annualized_CAPEX_dc_total','annualized_cost_dc'),dc)))
        revenue=float((supplied/1.15).sum()*self.cfg['datacenter_firm_revenue_per_mwh_it'])
        physical.economics.update(annualized_cost_hpp=cost,Revenue_dc=revenue,net_annual_value_system=revenue-cost-dc[-1],
            it_capacity_mw=d[-1],facility_load_mw=load,served_it_energy_mwh=float((supplied/1.15).sum()),
            served_facility_energy_mwh=float(supplied.sum()))
        physical.metrics=firm_load_metrics(load-supplied)
        physical.solver_status=dict(feasible=True,optimal=False,termination='TIME_LIMIT_INCUMBENT',
            certificate=dict(failed_hours_lower_bound=0,failed_hours_upper_bound=failed,
                             maximum_availability_certified=failed==0))
        model=object.__new__(adapter.hpp_model_simple_datacenter_joint_hpp_dc_sizing)
        model.sim_pars=physical.sim_pars
        model.prob=SimpleNamespace(set_val=lambda *a:None)
        model.list_vars=list(adapter.VARIABLE_NAMES)
        model.list_out_vars=['cost','penalty',adapter.SCORE_NAME]
        model.variables=deepcopy(self.cfg['variables']); model.bounds=self.bounds
        model.rejection_score=adapter.invalid_objective(self.bounds,self.cfg['reliability_target'])
        model.baseline_log_dir=self.directory
        def evaluate(instance,*args):
            instance.__dict__.update(physical.__dict__)
            return np.array([cost])
        return model,d,patch.object(adapter.PhysicalModel,'evaluate',evaluate)

    def test_penalty_thresholds_dominance_negative_values(self):
        b=self.bounds
        self.assertAlmostEqual(adapter.rational_penalty(1,b),1.2001)
        penalties=[adapter.rational_penalty(v,b) for v in range(8761)]
        self.assertTrue(np.all(np.diff(penalties)>0))
        lower=dict(Revenue_dc=0,annualized_cost_hpp=b['hpp_cost_upper_bound_eur_per_year'],
            annualized_cost_dc=b['dc_cost_upper_bound_eur_per_year'],net_annual_value_system=-b['cost_upper_bound_eur_per_year'])
        upper=dict(Revenue_dc=b['revenue_upper_bound_eur_per_year'],annualized_cost_hpp=0,
            annualized_cost_dc=0,net_annual_value_system=b['revenue_upper_bound_eur_per_year'])
        for target,hmax in ((.7,2628),(.99,87),(.999,8),(1.,0)):
            feasible=adapter.objective_values(lower,hmax,target,b)
            below=adapter.objective_values(upper,hmax+1,target,b)
            self.assertEqual(feasible['reliability_penalty_eur_per_year'],0)
            self.assertEqual(feasible['max_failed_hours'],hmax)
            self.assertGreater(feasible['penalized_annual_value'],below['penalized_annual_value'])
            self.assertLess(adapter.invalid_objective(b,target),adapter.objective_values(lower,8760,target,b)['penalized_annual_value'])
        with self.assertRaises(ValueError): adapter.penalty_parameters(dict(joint_penalty_A=.1))
        with self.assertRaises(ValueError): adapter.penalty_parameters(dict(joint_value_scale_factor=.5))
        with self.assertRaises(ValueError):
            adapter.objective_values(dict(upper,Revenue_dc=1e20,net_annual_value_system=1e20),0,.7,b)

    def test_native_evaluator_receives_complete_penalized_objective(self):
        from hydesign.Parallel_EGO import model_evaluation
        scores=[]
        for failed in (0,3000,4000):
            model,d,context=self.candidate(failed=failed)
            evaluation_errors=[]
            evaluate=model.evaluate
            def checked_evaluate(*args):
                try: return evaluate(*args)
                except Exception as exc:
                    evaluation_errors.append(exc)
                    raise
            model.evaluate=checked_evaluate
            inputs=dict(hpp_model=lambda **kwargs:model,scaler=SimpleNamespace(inverse_transform=lambda x:x),
                list_vars=model.list_vars,design_vars=model.list_vars,fixed_vars=[],variables=model.variables,
                opt_sign=-1,op_var_index=2)
            with context: value=model_evaluation((np.array([d]),inputs))
            if evaluation_errors: raise evaluation_errors[0]
            value=float(value)
            expected=model.economics['net_annual_value_system']-model.scoring['reliability_penalty_eur_per_year']
            self.assertEqual(value,-expected)
            scores.append(value)
            if failed:
                self.assertEqual(model.candidate_status,'BELOW_TARGET')
                self.assertEqual(model.record['reliability_assessment'],'BELOW_TARGET_TIME_LIMIT_UNRESOLVED')
                self.assertTrue(Path(model.record['dispatch_path']).exists())
        self.assertLess(scores[0],scores[1]); self.assertLess(scores[1],scores[2])

    def test_recovery_chooses_value_not_lcoe_and_never_solves(self):
        records=[]
        for load,turbines,solar in ((8,6,0),(16,60,250)):
            model,d,context=self.candidate(load=load,turbines=turbines,solar=solar)
            with context: model.evaluate(*d)
            records.append(model.record)
        self.assertGreater(records[1]['lcoe_hpp_eur_per_mwh'],records[0]['lcoe_hpp_eur_per_mwh'])
        self.assertGreater(records[1]['net_annual_value_system'],records[0]['net_annual_value_system'])
        with patch.object(adapter.dc_ems,'ems_cplex_simple_datacenter_max_reliability',
                          side_effect=AssertionError('Recovery must not solve EMS')):
            best,errors,_=runner.independently_validate(records,self.cfg,self.cfg['variables'],self.directory)
        self.assertEqual(errors,[])
        self.assertEqual(best['design'][3],60)
        Path(records[1]['dispatch_path']).write_bytes(b'corrupt')
        best,errors,_=runner.independently_validate(records,self.cfg,self.cfg['variables'],self.directory)
        self.assertEqual(best['design'][3],6); self.assertIn('checksum',errors[0]['error'])

    def test_invalid_physics_and_below_target_cannot_be_recovered(self):
        model,d,context=self.candidate(failed=4000)
        with context: model.evaluate(*d)
        best,errors,_=runner.independently_validate([model.record],self.cfg,self.cfg['variables'],self.directory)
        self.assertIsNone(best)
        with patch.object(adapter.PhysicalModel,'evaluate',side_effect=ValueError('invalid physics')):
            out=model.evaluate(*d)
        self.assertEqual(out[-1],model.rejection_score)
        self.assertFalse(model.record['physical_validation_passed'])
        self.assertIsNone(model.record['metrics']); self.assertIsNone(model.record['economics'])
        self.assertIsNone(model.record['dispatch_path'])
        model,d,context=self.candidate()
        with context: model.evaluate(*d)
        with np.load(model.record['dispatch_path']) as archive: arrays={k:archive[k].copy() for k in archive.files}
        arrays['served_facility_mw'][0]+=1
        np.savez_compressed(model.record['dispatch_path'],**arrays)
        model.record['dispatch_sha256']=hashlib.sha256(Path(model.record['dispatch_path']).read_bytes()).hexdigest()
        best,errors,_=runner.independently_validate([model.record],self.cfg,self.cfg['variables'],self.directory)
        self.assertIsNone(best); self.assertIn('balance',errors[0]['error'].lower())

    def test_bounds_config_and_notebook(self):
        cfg=deepcopy(self.cfg); cfg['variables'][adapter.IT_NAME]['limits'][1]*=2
        larger=adapter.economic_bounds(cfg,cfg['variables'])
        self.assertGreater(larger['revenue_upper_bound_eur_per_year'],self.bounds['revenue_upper_bound_eur_per_year'])
        self.assertGreater(larger['cost_upper_bound_eur_per_year'],self.bounds['cost_upper_bound_eur_per_year'])
        notebook=runner.ROOT/'docs/notebooks/simple_datacenter_joint_hpp_dc_sizing.ipynb'
        nb=json.loads(notebook.read_text(encoding='utf8'))
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        compile(code,str(notebook),'exec')
        self.assertNotIn('def ',code)
        self.assertIn("joint_penalty_scaling='reference_revenue'",code)
        self.assertIn('joint_penalty_gamma=0.25',code)
        self.assertIn('saved_run is None',code)
        self.assertEqual(self.cfg['hpp_sizing_objective'],runner.OBJECTIVE)

    def test_recovery_rejects_changed_score_or_reported_reliability(self):
        model,d,context=self.candidate()
        with context: model.evaluate(*d)
        for key in ('score','LOLH'):
            record=deepcopy(model.record)
            if key=='score': record[key]+=1e7
            else: record['metrics'][key]=123
            best,errors,_=runner.independently_validate([record],self.cfg,self.cfg['variables'],self.directory)
            self.assertIsNone(best); self.assertTrue(errors)

    def test_saved_loading_and_figures_do_not_solve(self):
        model,d,context=self.candidate()
        with context: model.evaluate(*d)
        best,errors,_=runner.independently_validate([model.record],self.cfg,self.cfg['variables'],self.directory)
        self.assertFalse(errors)
        runner.write_evaluation_history([model.record],self.directory/'evaluations.csv',model.record['started_at'])
        adapter.write_json_atomic(self.directory/'result.json',dict(best=best,configuration=self.cfg,
            final_status='VALIDATED_FEASIBLE_INCUMBENT',evaluated_designs=1,feasible_evaluations=1))
        before=hashlib.sha256((self.directory/'result.json').read_bytes()).hexdigest()
        with patch.object(adapter.PhysicalModel,'evaluate',side_effect=AssertionError('Must not evaluate')):
            loaded,history=runner.load_joint_hpp_dc_sizing(self.directory,save_figures=True)
        self.assertEqual(len(history),1)
        figures=Path(loaded['figure_directory'])
        self.assertTrue((figures/'joint_hpp_dc_sizing.png').exists())
        self.assertTrue((figures/'joint_hpp_dc_sizing.pdf').exists())
        self.assertTrue((figures/'summary.csv').exists())
        self.assertEqual(before,hashlib.sha256((self.directory/'result.json').read_bytes()).hexdigest())
        # Execute the financial presentation with old/new histories without Styler dependencies.
        import pandas as pd
        import contextlib
        import io
        notebook=json.loads((runner.ROOT/'docs/notebooks/simple_datacenter_joint_hpp_dc_sizing.ipynb').read_text(encoding='utf8'))
        source=next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type']=='code' and 'summary_table =' in ''.join(c['source']))
        original=history.copy(deep=True)
        displays=[]
        namespace=dict(pd=pd,result=loaded,history=history,run_directory=str(self.directory),display=displays.append)
        with contextlib.redirect_stdout(io.StringIO()): exec(source,namespace)
        pd.testing.assert_frame_equal(history,original)
        self.assertEqual(displays[0].loc['Net annual system value [MEUR/year]','Validated best design'],
                         f"{best['economics']['net_annual_value_system']/1e6:,.2f}")
        self.assertEqual(namespace['history_view']['Net annual system value [MEUR/year]'].iloc[0],
                         history['net_annual_value_system'].iloc[0]/1e6)

    def use_scaling(self,mode,gamma=.25):
        self.cfg.update(joint_penalty_scaling=mode,joint_penalty_gamma=gamma)
        self.bounds=adapter.economic_bounds(self.cfg,self.cfg['variables'])

    def test_legacy_equation_and_fallback_are_unchanged(self):
        model,d,context=self.candidate()
        with context: model.evaluate(*d)
        for explicit in (False,True):
            cfg=dict(self.cfg)
            if explicit: cfg['joint_penalty_scaling']='economic_bound'
            b=adapter.economic_bounds(cfg,cfg['variables'])
            for failed in (0,2629,4000,8760):
                result=adapter.objective_values(model.economics,failed,.7,b)
                expected=b['value_scale_eur_per_year']*adapter.rational_penalty(max(0,failed-2628),b)
                self.assertEqual(result['reliability_penalty_eur_per_year'],expected)
                self.assertEqual(result['penalized_annual_value'],model.economics['net_annual_value_system']-expected)
            expected=-b['cost_upper_bound_eur_per_year']-b['value_scale_eur_per_year']*adapter.rational_penalty(8760-2628,b)
            expected-=max(1.,.01*b['value_scale_eur_per_year'])
            self.assertEqual(adapter.invalid_objective(b,.7),expected)

    def test_reference_equation_gamma_and_fixed_domain_scale(self):
        self.use_scaling('reference_revenue')
        b=self.bounds
        reference=8760*self.cfg['datacenter_firm_revenue_per_mwh_it']*self.cfg['variables'][adapter.IT_NAME]['limits'][1]
        self.assertEqual(b['reference_revenue_eur_per_year'],reference)
        for load in (8,16):
            model,d,context=self.candidate(load=load)
            with context: model.evaluate(*d)
            self.assertEqual(model.record['reference_revenue_eur_per_year'],reference)
            for failed in (0,2628,2629,2640,8760):
                score=adapter.objective_values(model.economics,failed,.7,b)
                expected=.25*reference*adapter.rational_penalty(max(0,failed-2628),b)/b['joint_penalty_A']
                self.assertAlmostEqual(score['reliability_penalty_eur_per_year'],expected,places=6)
                doubled=adapter.economic_bounds(dict(self.cfg,joint_penalty_gamma=.5),self.cfg['variables'])
                self.assertEqual(adapter.objective_values(model.economics,failed,.7,doubled)['reliability_penalty_eur_per_year'],
                                 2*score['reliability_penalty_eur_per_year'])
        for change in ('revenue','domain'):
            cfg=deepcopy(self.cfg)
            if change=='revenue': cfg['datacenter_firm_revenue_per_mwh_it']*=2
            else: cfg['variables'][adapter.IT_NAME]['limits'][1]*=2
            bound=adapter.economic_bounds(cfg,cfg['variables'])
            self.assertEqual(bound['reference_revenue_eur_per_year'],2*reference)
            self.assertEqual(bound['penalty_scale_eur_per_year'],2*b['penalty_scale_eur_per_year'])
        inflated=adapter.economic_bounds(dict(self.cfg,joint_economic_bound_factor=2,joint_value_scale_factor=2),self.cfg['variables'])
        self.assertEqual(inflated['penalty_scale_eur_per_year'],b['penalty_scale_eur_per_year'])
        self.assertEqual(inflated['cost_upper_bound_eur_per_year'],2*b['cost_upper_bound_eur_per_year'])

    def test_reference_monotonicity_fallback_and_configuration(self):
        self.use_scaling('reference_revenue'); b=self.bounds
        self.assertTrue(np.all(np.diff([b['penalty_scale_eur_per_year']*adapter.rational_penalty(v,b) for v in range(8761)])>0))
        lower=dict(Revenue_dc=0,annualized_cost_hpp=b['hpp_cost_upper_bound_eur_per_year'],
            annualized_cost_dc=b['dc_cost_upper_bound_eur_per_year'],net_annual_value_system=-b['cost_upper_bound_eur_per_year'])
        for target in (.7,.99,.999,1.):
            fallback=adapter.invalid_objective(b,target)
            self.assertTrue(np.isfinite(fallback))
            self.assertLess(fallback,adapter.objective_values(lower,8760,target,b)['penalized_annual_value'])
        passed=runner.prepare_configuration(CONFIG,dict(joint_penalty_scaling='reference_revenue',joint_penalty_gamma=.4))
        inputs=runner.physical_inputs(CONFIG,passed,self.directory)
        self.assertEqual(inputs['joint_penalty_scaling'],'reference_revenue')
        self.assertEqual(inputs['joint_penalty_gamma'],.4)
        for params in (dict(joint_penalty_scaling='wrong'),dict(joint_penalty_gamma=0),dict(joint_penalty_gamma=float('nan'))):
            with self.assertRaises(ValueError): adapter.penalty_parameters(params)
        with self.assertRaises(ValueError):
            adapter.economic_bounds(dict(self.cfg,datacenter_firm_revenue_per_mwh_it=0),self.cfg['variables'])
        # Zero revenue remains allowed by the original scaling.
        adapter.economic_bounds(dict(self.cfg,joint_penalty_scaling='economic_bound',datacenter_firm_revenue_per_mwh_it=0),self.cfg['variables'])

    def test_reference_native_ego_sign_and_time_limited_records(self):
        self.use_scaling('reference_revenue')
        self.test_native_evaluator_receives_complete_penalized_objective()

    def test_reference_invalid_physics_still_rejected(self):
        self.use_scaling('reference_revenue')
        self.test_invalid_physics_and_below_target_cannot_be_recovered()

    def test_reference_recovery_rejects_higher_scoring_below_target(self):
        self.use_scaling('reference_revenue',gamma=.001)
        records=[]
        for load,failed in ((8,0),(40,3000)):
            model,d,context=self.candidate(load=load,failed=failed)
            with context: model.evaluate(*d)
            records.append(model.record)
        self.assertGreater(records[1]['score'],records[0]['score'])
        with patch.object(adapter.PhysicalModel,'evaluate',side_effect=AssertionError('Must not solve')):
            best,errors,_=runner.independently_validate(records,self.cfg,self.cfg['variables'],self.directory)
        self.assertFalse(errors)
        self.assertEqual(best['design'][-1],8/1.15)
        runner.write_evaluation_history(records,self.directory/'evaluations.csv',records[0]['started_at'])
        import pandas as pd
        history=pd.read_csv(self.directory/'evaluations.csv')
        self.assertTrue((history['joint_penalty_scaling']=='reference_revenue').all())
        self.assertTrue((history['joint_penalty_gamma']==.001).all())
        self.assertEqual(history.iloc[1]['reliability_penalty_eur_per_year'],records[1]['reliability_penalty_eur_per_year'])

    def test_saved_legacy_records_without_scaling_metadata_recover(self):
        model,d,context=self.candidate()
        with context: model.evaluate(*d)
        record=deepcopy(model.record)
        for key in ('joint_penalty_scaling','joint_penalty_gamma','reference_revenue_eur_per_year','penalty_scale_eur_per_year'):
            record.pop(key,None); record['bounds'].pop(key,None)
        with patch.object(adapter.PhysicalModel,'evaluate',side_effect=AssertionError('Must not solve')):
            best,errors,_=runner.independently_validate([record],self.cfg,self.cfg['variables'],self.directory)
        self.assertFalse(errors); self.assertIsNotNone(best)
        changed=dict(self.cfg,joint_penalty_scaling='reference_revenue')
        best,errors,_=runner.independently_validate([record],changed,self.cfg['variables'],self.directory)
        self.assertIsNone(best); self.assertIn('scaling',errors[0]['error'])

    def test_revenue_batch_saves_incremental_summary_and_separate_evaluations(self):
        import pandas as pd
        rates=[500,1000,2000,3000,4000]
        cfg=dict(self.cfg,hpp_sizing_output_dir=str(self.directory))
        original=deepcopy(cfg); seen=[]
        def fake_run(path,overrides):
            scenario=runner._output_path(overrides['hpp_sizing_output_dir'])
            current=pd.read_csv(scenario.parent/'summary.csv')
            self.assertEqual(len(current),5)
            self.assertEqual(current.iloc[len(seen)]['status'],'RUNNING')
            self.assertTrue((current.iloc[:len(seen)]['status']=='NO_VALIDATED_FEASIBLE_SOLUTION_FOUND').all())
            seen.append(overrides['datacenter_firm_revenue_per_mwh_it'])
            directory=runner._output_path(scenario/'saved_run'); directory.mkdir(parents=True)
            pd.DataFrame([dict(evaluation_index=1,status='BELOW_TARGET',score=-123.)]).to_csv(directory/'evaluations.csv',index=False)
            result=dict(run_directory=str(directory),final_status='NO_VALIDATED_FEASIBLE_SOLUTION_FOUND',
                configuration=overrides,best=None,evaluated_designs=1,feasible_evaluations=0,
                optimization_seconds=0,termination_reason='NATIVE_COMPLETED')
            adapter.write_json_atomic(directory/'result.json',result)
            return result
        with patch.object(runner,'run_native_joint_hpp_dc_sizing',side_effect=fake_run):
            batch,summary=runner.run_joint_revenue_scenarios(CONFIG,cfg,rates)
        self.assertEqual(seen,rates);self.assertEqual(cfg,original)
        saved=pd.read_csv(batch/'summary.csv')
        self.assertEqual(saved['revenue_eur_per_mwh_it'].tolist(),rates)
        self.assertTrue(all(Path(p).is_file() for p in saved['evaluations_csv']))
        first=(batch/'summary.csv').read_bytes()
        seen.clear()
        with patch.object(runner,'run_native_joint_hpp_dc_sizing',side_effect=fake_run):
            second,_=runner.run_joint_revenue_scenarios(CONFIG,cfg,rates)
        self.assertNotEqual(batch,second);self.assertEqual(first,(batch/'summary.csv').read_bytes())

    def test_revenue_batch_retains_prior_results_after_interruption(self):
        import pandas as pd
        cfg=dict(self.cfg,hpp_sizing_output_dir=str(self.directory))
        with patch.object(runner,'run_native_joint_hpp_dc_sizing',side_effect=[ValueError('test failure'),KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_joint_revenue_scenarios(CONFIG,cfg,[500,1000,2000])
        batch=next(self.directory.glob('revenue_batch_*'))
        saved=pd.read_csv(batch/'summary.csv')
        self.assertEqual(saved['status'].tolist(),['ERROR','INTERRUPTED','PENDING'])
        self.assertIn('test failure',saved.iloc[0]['error'])


def integration():
    """Real 8760-hour parity and variable-demand checks, including reference battery."""
    cfg=configuration()
    directory=runner._output_path(runner.ROOT/'hpp_sizing_runs/joint_hpp_dc_sizing'/('validation_'+uuid4().hex[:8]))
    directory.mkdir(parents=True)
    inputs=runner.physical_inputs(CONFIG,cfg,directory)
    rows=[]
    # No-battery parity is deterministic and proves the complete wiring. The
    # battery reference below separately exercises unchanged storage physics.
    d=[35,300,5,20,7,180,42,180,1.25,0,8]
    original=adapter.PhysicalModel(**inputs)
    try:
        original.evaluate(*d)
        joint=adapter.hpp_model_simple_datacenter_joint_hpp_dc_sizing(**inputs,variables=cfg['variables'],
                    baseline_log_dir=str(directory/'evaluations'))
        try:
            joint.evaluate(*d,16/1.15)
            assert joint.final_checks is not None,joint.candidate_error
            for k in original.hourly: np.testing.assert_allclose(joint.hourly[k],original.hourly[k],atol=1e-5,rtol=0)
            for k in original.economics: np.testing.assert_allclose(joint.economics[k],original.economics[k],rtol=1e-9,atol=1e-5)
            rows.append(dict(test='fixed_16MW_no_battery_parity',passed=True,metrics=joint.metrics,economics=joint.economics))
            for load in (8,24):
                joint.evaluate(*d,load/1.15)
                assert joint.final_checks is not None,joint.candidate_error
                np.testing.assert_allclose(joint.prob.get_val('facility_load_mw'),load)
                np.testing.assert_allclose(joint.sim_pars['facility_load_mw'],load)
                np.testing.assert_allclose(joint.economics['G_proxy_MW'],load)
                np.testing.assert_allclose(joint.economics['annualized_cost_dc'],
                                           original.economics['annualized_cost_dc']*load/16)
                rows.append(dict(test=f'variable_facility_{load}MW',passed=True,metrics=joint.metrics,economics=joint.economics))
            d[-2]=25
            joint.evaluate(*d,16/1.15)
            assert joint.final_checks is not None,joint.candidate_error
            rows.append(dict(test='reference_25MW_200MWh',passed=True,metrics=joint.metrics,
                economics=joint.economics,validation=joint.final_checks,solver_status=joint.solver_status))
        finally: joint.prob.cleanup()
    finally: original.prob.cleanup()
    adapter.write_json_atomic(directory/'validation.json',dict(tests=rows,configuration=cfg))
    print('VALIDATION PASSED:',directory,flush=True)
    for row in rows: print(row['test'],row['metrics'],flush=True)


def smoke():
    cfg=configuration()
    cfg.update(n_doe=8,max_iter=1,n_clusters=1,npred=1000,n_procs=1,n_seed=0,min_conv_iter=100,
               hpp_sizing_time_limit_seconds=300,ems_time_limit_seconds=30,ems_threads=1)
    result=runner.run_native_joint_hpp_dc_sizing(CONFIG,cfg)
    assert result.get('recovery_complete'),result
    assert result.get('evaluated_designs',0)>0,result
    assert result['termination_reason']!='NATIVE_ERROR',result.get('error')
    if result['best']:
        assert result['best']['validation']['outer_feasible']
        assert result['best']['scoring']['reliability_penalty_eur_per_year']==0
    best=result['best']
    print(json.dumps(dict(directory=result['run_directory'],status=result['final_status'],
        evaluated_designs=result['evaluated_designs'],
        best=None if best is None else {k:best[k] for k in ('design','metrics','economics','scoring','validation')}),
        indent=2),flush=True)


if __name__=='__main__':
    if '--integration' in sys.argv: integration()
    elif '--smoke' in sys.argv: smoke()
    else: unittest.main()
