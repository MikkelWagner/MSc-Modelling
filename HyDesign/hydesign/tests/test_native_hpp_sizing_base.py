"""Baseline interface/process checks; --integration runs timed and normal CPLEX cases."""

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import yaml

from hydesign.assembly import hpp_assembly_simple_datacenter_hpp_sizing_base as baseline
from hydesign import sizing_native_hpp_sizing_base as execution


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "docs/notebooks/simple_datacenter_hpp_sizing_base.ipynb"
CONFIG = ROOT / "hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml"


@contextmanager
def scratch_directory():
    parent = (ROOT / "hpp_sizing_runs").resolve()
    parent.mkdir(exist_ok=True)
    directory = (parent / ("test_hpp_sizing_base_" + uuid4().hex)).resolve()
    assert directory.parent == parent and directory.name.startswith("test_hpp_sizing_base_")
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        shutil.rmtree(directory)


def physical_result(failed_hours=0):
    """A balanced full-year fixture with no battery and known failed-hour count."""
    load = np.full(8760, 16.0)
    supplied = load.copy()
    supplied[:failed_hours] = 15.0
    zero = np.zeros(8760)
    hourly = {
        "wind_mw": supplied, "pv_mw": zero, "renewable_available_mw": supplied,
        "facility_load_mw": load, "served_facility_mw": supplied,
        "unserved_facility_mw": load - supplied, "firm_demand_it_mw": load / 1.15,
        "served_firm_it_mw": supplied / 1.15, "unserved_firm_it_mw": (load - supplied) / 1.15,
        "charge_mw": zero, "discharge_mw": zero, "curtailment_mw": zero,
    }
    return SimpleNamespace(
        sim_pars={"facility_load_mw": 16, "reliability_target": .99, "pue": 1.15,
                  "battery_depth_of_discharge": .9, "battery_charge_efficiency": np.sqrt(.86),
                  "battery_initial_soc": .5, "epsilon_mw": .001},
        hourly=hourly, soc_mwh=np.zeros(8761), solver_status={"optimal": True},
        economics={"CAPEX_hpp": 1e8, "annualized_cost_hpp": 1e7},
    )


class BaselineTests(unittest.TestCase):
    def test_simple_notebook_and_unmodified_native_driver_call(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf8"))
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        compile(code, str(NOTEBOOK), "exec")
        runner = Path(execution.__file__).read_text(encoding="utf8")
        self.assertIn("ego = EfficientGlobalOptimizationDriver(**inputs)", runner)
        self.assertIn("ego.run()", runner)
        self.assertIn("run_native_hpp_sizing(config_path, overrides=config)", code)
        self.assertLess(len(code.splitlines()), 115)
        self.assertNotIn("def ", code)
        self.assertNotIn("optimization_method", code)
        adapter = Path(baseline.__file__).read_text(encoding="utf8")
        for forbidden in ("sizing_simple_datacenter_with_hpp_sizing", "search_simple_datacenter_with_hpp_sizing",
                          "hpp_assembly_simple_datacenter_with_hpp_sizing", "BudgetedEvaluator"):
            self.assertNotIn(forbidden, code + adapter + runner)
        self.assertNotIn("patch.object(native_ego", adapter)

    def test_cost_bound_uses_native_classes_and_exceeds_reference_cost(self):
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
        variables = deepcopy(cfg["variables"])
        fixed_cost_bound = baseline.annual_cost_bound(cfg, variables)
        for name in baseline.CAPACITY_VARIABLES:
            variables[name]["var_type"] = "design"
        full_bound = baseline.annual_cost_bound(cfg, variables)
        self.assertGreater(full_bound, fixed_cost_bound)
        rejection = -(full_bound + max(1, .01 * full_bound)) / 1e6
        self.assertTrue(np.isfinite(rejection))
        self.assertLess(rejection, -full_bound / 1e6)
        cfg["wind_variable_onm_cost"] = -1
        with self.assertRaises(ValueError):
            baseline.annual_cost_bound(cfg, variables)

    def test_rejections_preserve_distinct_reasons(self):
        for code, solution, expected in (
            (103, False, "PROVEN_INFEASIBLE"),
            (108, False, "TIME_LIMIT_UNRESOLVED"),
            (107, True, "TIME_LIMIT_WITH_INCUMBENT"),
            (102, True, "VALIDATION_FAILURE"),
            (119, False, "SOLVER_UNRESOLVED_OR_ERROR"),
        ):
            self.assertEqual(baseline.rejected_status([
                {"status_code": code, "has_solution": solution}]), expected)
        self.assertEqual(baseline.rejected_status([]), "MODEL_ERROR")

    def test_bound_includes_replacement_at_maximum_battery_size(self):
        cfg = execution.read_configuration(CONFIG)
        variables = deepcopy(cfg["variables"])
        for name in baseline.CAPACITY_VARIABLES:
            variables[name]["var_type"] = "design"
        first = baseline.annual_cost_bound(cfg, variables)
        second = baseline.annual_cost_bound(dict(cfg, battery_replacement_cost_fraction=2), variables)
        power = variables["b_P [MW]"]["limits"][1]
        energy = power*variables["b_E_h [h]"]["limits"][1]
        capex = cfg["battery_energy_cost"]*energy + power*sum(cfg[k] for k in (
            "battery_power_cost", "battery_BOP_installation_commissioning_cost", "battery_control_system_cost"))
        expected_increment = capex/1.06**15*.06/(1-1.06**-25)
        np.testing.assert_allclose(second-first, expected_increment, rtol=1e-12)

    def test_bound_includes_maximum_demand_battery_and_land(self):
        cfg = execution.read_configuration(CONFIG)
        variables = deepcopy(cfg["variables"])
        for name in baseline.CAPACITY_VARIABLES:
            variables[name]["var_type"] = "design"
        actual = baseline.annual_cost_bound(cfg, variables, facility_load_max_mw=90)
        no_shared = dict(cfg, hpp_BOS_soft_cost=0, hpp_grid_connection_cost=0, land_cost=0)
        asset_bound = baseline.annual_cost_bound(no_shared, variables, facility_load_max_mw=90)
        power = variables["b_P [MW]"]["limits"][1]
        area = max(variables["Nwt"]["limits"][1]*5/7,
                   variables["solar_MW [MW AC]"]["limits"][1]*.01226)
        expected = (119940*(90+power)+300000*area)*.06/(1-1.06**-25)
        self.assertAlmostEqual(actual-asset_bound, expected)

    def test_final_check_counts_hours_at_99_percent(self):
        result = baseline.validate_result(physical_result(87), [0, 8])
        self.assertEqual(result["max_failed_hours"], 87)
        self.assertEqual(result["metrics"]["LOLH"], 87)
        self.assertEqual(result["metrics"]["EENS"], 87)
        with self.assertRaisesRegex(ValueError, "reliability"):
            baseline.validate_result(physical_result(88), [0, 8])
        model = physical_result(9)
        model.sim_pars["reliability_target"] = .999
        with self.assertRaisesRegex(ValueError, "reliability"):
            baseline.validate_result(model, [0, 8])

    def test_final_check_rejects_invalid_physics_and_cost(self):
        model = physical_result()
        model.hourly["curtailment_mw"] = np.ones(8760)
        with self.assertRaisesRegex(ValueError, "balance"):
            baseline.validate_result(model, [0, 8])
        model = physical_result()
        model.economics["annualized_cost_hpp"] = np.nan
        with self.assertRaisesRegex(ValueError, "annualized_cost"):
            baseline.validate_result(model, [0, 8])

    def test_final_check_rejects_soc_and_simultaneous_charge_discharge(self):
        model = physical_result()
        model.soc_mwh = np.full(8761, 8.0)
        model.soc_mwh[-1] = 9.0
        with self.assertRaises(ValueError):
            baseline.validate_result(model, [2, 8])
        model.soc_mwh[-1] = 8.0
        model.hourly["charge_mw"] = np.ones(8760)
        model.hourly["discharge_mw"] = np.full(8760, .86)
        model.hourly["wind_mw"] = model.hourly["wind_mw"] + .14
        model.hourly["renewable_available_mw"] = model.hourly["wind_mw"].copy()
        with self.assertRaisesRegex(ValueError, "Simultaneous"):
            baseline.validate_result(model, [2, 8])

    def test_recovery_ignores_rejected_candidates_and_selects_lowest_cost(self):
        model = physical_result()
        model.evaluate = lambda *args: None
        model.prob = SimpleNamespace(cleanup=lambda: None)
        records = [{"status": "PROVEN_INFEASIBLE", "annualized_cost_hpp": 0, "design": [0, 8]},
                   {"status": "TIME_LIMIT_UNRESOLVED", "annualized_cost_hpp": 1, "design": [0, 8]},
                   {"status": "FEASIBLE", "annualized_cost_hpp": 2e7, "design": [0, 9], "validation": {}},
                   {"status": "FEASIBLE", "annualized_cost_hpp": 1e7, "design": [0, 8], "validation": {}}]
        with scratch_directory() as directory:
            with patch("hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost.hpp_model_simple_datacenter_with_hpp_cost", return_value=model):
                best, errors, _ = execution.independently_validate(records, {}, directory)
            self.assertEqual(best["design"], [0, 8])
            self.assertEqual(errors, [])
            best, _, _ = execution.independently_validate(records[:2], {}, directory)
            self.assertIsNone(best)

    def test_atomic_records_and_deadline_classification(self):
        with scratch_directory() as directory:
            directory = Path(directory)
            path = directory / "one_hpp_sizing_base.json"
            baseline.write_json_atomic(path, {"status": "RUNNING", "started_at": 1, "design": [20]})
            (directory / "one_hpp_sizing_base.tmp").write_text('{"partial":')
            records = execution.read_evaluations(directory, "TIME_LIMIT")
            self.assertEqual(records[0]["status"], "TIME_LIMIT_INTERRUPTED")
            baseline.write_json_atomic(path, {"status": "FEASIBLE", "started_at": 1,
                                             "completed_at": 2, "annualized_cost_hpp": 10, "design": [20]})
            self.assertEqual(execution.read_evaluations(directory, "TIME_LIMIT")[0]["status"], "FEASIBLE")

    def test_overrides_flags_bounds_and_physical_settings_are_preserved(self):
        config = execution.read_configuration(CONFIG)
        config.update(facility_load_mw=10, reliability_target=.99, ems_time_limit_seconds=33,
                      pue=1.2, battery_initial_soc=.6, wind_turbine_cost=123456,
                      hpp_sizing_time_limit_seconds=None, n_procs=2, n_seed=3)
        config["variables"]["Nwt"].update(var_type="design", limits=[12, 50])
        config["variables"]["solar_MW [MW AC]"].update(var_type="fixed", value=172)
        prepared = execution.prepare_configuration(CONFIG, config)
        self.assertEqual(prepared["variables"], config["variables"])
        inputs = execution.physical_inputs(CONFIG, prepared, ROOT)
        for key in ("facility_load_mw", "reliability_target", "ems_time_limit_seconds", "pue",
                    "battery_initial_soc", "wind_turbine_cost"):
            self.assertEqual(inputs[key], config[key])
        self.assertEqual(prepared["n_procs"], 2)
        self.assertIsNone(prepared["hpp_sizing_time_limit_seconds"])
        with self.assertRaises(ValueError):
            execution.prepare_configuration(CONFIG, {"hpp_sizing_time_limit_seconds": 0})

    @unittest.skipUnless(os.name == "nt", "Windows job-object process containment")
    def test_deadline_kills_child_tree_and_disabled_limit_completes(self):
        child_code = (
            "import pathlib,sys,time,subprocess; gate=pathlib.Path(sys.argv[1]); "
            "\nwhile not gate.exists(): time.sleep(.01)\n"
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); time.sleep(120)"
        )
        with scratch_directory() as directory:
            gate = Path(directory) / "go.flag"
            result = execution.run_contained([sys.executable, "-c", child_code, str(gate)], directory, 1.5, gate)
            self.assertEqual(result["termination_reason"], "TIME_LIMIT")
            self.assertEqual(result["process_cleanup"]["active_processes_after_stop"], 0)
            self.assertGreaterEqual(result["process_cleanup"]["total_processes"], 2)
            gate.unlink()
            result = execution.run_contained([sys.executable, "-c", "pass"], directory, None, gate)
            self.assertEqual(result["termination_reason"], "NORMAL_EXIT")
            self.assertEqual(result["process_cleanup"]["active_processes_after_stop"], 0)

    def test_adapter_sign_and_finite_rejection_clear_previous_results(self):
        # Isolate the interface; the integration run below exercises real physics.
        model = object.__new__(baseline.hpp_model_simple_datacenter_hpp_sizing_base)
        model.list_vars = ["a", "b", "c", "Nwt"]
        model.list_out_vars = ["cost", baseline.VALID_NAME, baseline.SCORE_NAME]
        model.variables = {name: {"var_type": "fixed", "value": value}
                           for name, value in zip(model.list_vars, [1, 2, 3, 4])}
        model.cost_upper_bound = 2e7
        model.objective = baseline.COST_OBJECTIVE
        model.objective_scale = 1e6
        model.rejection_score = -20.2
        model.baseline_log_dir = None
        model.economics = {"annualized_cost_hpp": 1e7}
        with patch.object(baseline.hpp_model_simple_datacenter_with_hpp_cost, "evaluate", return_value=[1e7]), \
                patch.object(baseline, "validate_result", return_value={}):
            np.testing.assert_array_equal(model.evaluate(1, 2, 3, 4), [1e7, 1, -10])
        with patch.object(baseline.hpp_model_simple_datacenter_with_hpp_cost, "evaluate", side_effect=RuntimeError("solver failure")):
            values = model.evaluate(1, 2, 3, 4)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[-1], -20.2)
        self.assertEqual(values[-2], 0)
        self.assertEqual(model.candidate_status, "MODEL_ERROR")
        self.assertIn("solver failure", model.candidate_error)
        self.assertIsNone(model.economics)


class LcoeFrontierTests(unittest.TestCase):
    def test_long_nested_checkpoint_paths(self):
        with scratch_directory() as directory:
            path = execution._output_path(Path(directory)/('nested_'+'a'*70))
            path.mkdir()
            record = path/('b'*32+'_hpp_sizing_base.json')
            baseline.write_json_atomic(record, {"status":"FEASIBLE"})
            self.assertEqual(json.loads(record.read_text())["status"], "FEASIBLE")
            record.unlink()
            path.rmdir()

    def test_ratio_default_objective_and_invalid_energy(self):
        economics = {"annualized_cost_hpp": 12e6, "served_facility_energy_mwh": 120000}
        self.assertEqual(baseline.objective_value(economics, baseline.LCOE_OBJECTIVE), 100)
        self.assertEqual(baseline.objective_value(economics), 12e6)
        self.assertEqual(execution.prepare_configuration(CONFIG)["hpp_sizing_objective"], baseline.COST_OBJECTIVE)
        for energy in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                baseline.objective_value(dict(economics, served_facility_energy_mwh=energy), baseline.LCOE_OBJECTIVE)
        with self.assertRaises(ValueError):
            execution.prepare_configuration(CONFIG, {"hpp_sizing_objective": "typo"})

    def test_lcoe_bound_and_all_reliability_targets(self):
        cfg = execution.read_configuration(CONFIG)
        targets = [.4,.5,.6,.7,.8,.9,.95,.97,.99,.999,1.]
        failures = [5256,4380,3504,2628,1752,876,438,262,87,8,0]
        for target, failed in zip(targets, failures):
            cfg["reliability_target"] = target
            bound = baseline.objective_bound(cfg, 30e6, baseline.LCOE_OBJECTIVE)
            self.assertAlmostEqual(bound, 30e6/((8760-failed)*(16-.001)))
            rejection = -(bound+max(1,.01*bound))
            self.assertLess(rejection, -bound)
        with self.assertRaises(ValueError):
            baseline.objective_bound(dict(cfg, reliability_target=0), 30e6, baseline.LCOE_OBJECTIVE)

    @staticmethod
    def candidate(turbines):
        model = physical_result()
        model.sim_pars["reliability_target"] = .7
        if turbines == 1:
            for key in ("wind_mw", "renewable_available_mw", "served_facility_mw", "served_firm_it_mw"):
                model.hourly[key][:1500] = 0
            model.hourly["unserved_facility_mw"][:1500] = 16
            model.hourly["unserved_firm_it_mw"][:1500] = 16/1.15
        model.economics.update(annualized_cost_hpp=1e7 if turbines == 1 else 1.1e7,
                              served_facility_energy_mwh=float(model.hourly["served_facility_mw"].sum()))
        return model

    def test_native_ego_evaluation_and_recovery_reverse_cost_ranking(self):
        from hydesign.Parallel_EGO import model_evaluation
        designs = [[35,300,5,n,7,180,42,180,1.25,0,8] for n in (1,2)]
        records = []
        for design in designs:
            model = self.candidate(design[3])
            checks = baseline.validate_result(model, design)
            records.append(dict(status="FEASIBLE", design=design, validation=checks,
                                economics=model.economics, annualized_cost_hpp=model.economics["annualized_cost_hpp"],
                                started_at=1, completed_at=design[3]+1))
        for objective, expected in ((baseline.COST_OBJECTIVE,1), (baseline.LCOE_OBJECTIVE,2)):
            scores = []
            for design in designs:
                adapter = object.__new__(baseline.hpp_model_simple_datacenter_hpp_sizing_base)
                adapter.__dict__.update(self.candidate(design[3]).__dict__)
                adapter.list_vars = list(baseline.VARIABLE_NAMES)
                adapter.list_out_vars = ["cost", baseline.VALID_NAME, baseline.objective_name(objective)]
                adapter.variables = {k:dict(var_type="fixed",value=v) for k,v in zip(adapter.list_vars,design)}
                adapter.objective, adapter.objective_scale = objective, 1 if objective == baseline.LCOE_OBJECTIVE else 1e6
                adapter.cost_upper_bound = 30e6
                adapter.objective_upper_bound = baseline.objective_bound(adapter.sim_pars, 30e6, objective)
                adapter.rejection_score = -(adapter.objective_upper_bound+1)/adapter.objective_scale
                adapter.baseline_log_dir = None
                inputs = dict(hpp_model=lambda **kwargs: adapter, scaler=SimpleNamespace(inverse_transform=lambda x:x),
                              list_vars=adapter.list_vars, design_vars=adapter.list_vars, fixed_vars=[],
                              variables=adapter.variables, opt_sign=-1, op_var_index=2)
                with patch.object(baseline.hpp_model_simple_datacenter_with_hpp_cost,"evaluate",return_value=[1]):
                    scores.append(float(model_evaluation((np.array([design]), inputs))))
                self.assertEqual(adapter.candidate_status, "FEASIBLE")
                if objective == baseline.LCOE_OBJECTIVE:
                    self.assertAlmostEqual(scores[-1], adapter.economics["annualized_cost_hpp"]/adapter.economics["served_facility_energy_mwh"])
            self.assertEqual(int(np.argmin(scores))+1, expected)
            model = SimpleNamespace(prob=SimpleNamespace(cleanup=lambda:None))
            def evaluate(*design):
                model.__dict__.update(self.candidate(design[3]).__dict__)
            model.evaluate = evaluate
            with scratch_directory() as directory, patch(
                    "hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost.hpp_model_simple_datacenter_with_hpp_cost", return_value=model):
                best, errors, _ = execution.independently_validate(records, {}, directory, objective)
                self.assertEqual(best["design"][3], expected)
                self.assertFalse(errors)
                execution.write_evaluation_history(records, Path(directory)/"history.csv", 0, 4, objective,.7)
                import pandas as pd
                history = pd.read_csv(Path(directory)/"history.csv")
                self.assertEqual(json.loads(history.iloc[-1]["best_feasible_design"])[3], expected)

    def test_report_handles_dominated_missing_and_duplicate_points_without_optimizer(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with scratch_directory() as directory:
            directory = Path(directory)
            cases = [{"reliability_target": t,"folder":str(i)} for i,t in enumerate([.4,.5,.6,.7,1.])]
            baseline.write_json_atomic(directory/"experiment_config.json", {"cases":cases})
            for case, reliability, lcoe in zip(cases,[.7,.8,.9,.9],[100,90,110,110]):
                p = directory/case['folder'];p.mkdir()
                baseline.write_json_atomic(p/"result.json", {"best":{
                    "design":[35,300,5,20,7,180,42,180,1.25,25,8], "validation":{},
                    "economics":{},"metrics":{"firm_reliability":reliability},"lcoe_hpp_eur_per_mwh":lcoe}})
            snapshot = {p: p.read_bytes() for p in directory.rglob('*.json')}
            with patch.object(execution, "run_native_hpp_sizing", side_effect=AssertionError("Must not run EGO")):
                table,fig = execution.load_lcoe_reliability_experiment(directory,save_figures=True)
            self.assertEqual(table['nondominated'].tolist(),[False,True,True,True,False])
            self.assertEqual(table.iloc[-1]['feasibility_status'], 'PENDING')
            self.assertEqual(snapshot, {p:p.read_bytes() for p in snapshot})
            self.assertEqual(len(list(directory.glob('reports_*/pareto_frontier.pdf'))),1)
            plt.close(fig)

    def test_experiment_storage_and_new_folder_per_run(self):
        cfg = execution.read_configuration(CONFIG)
        cfg.update(facility_load_mw=16,pue=1.15,battery_power_fluctuation_penalty=.0001)
        for name in baseline.CAPACITY_VARIABLES:
            cfg['variables'][name]['var_type']='design'
        def fake_run(path, overrides):
            target = overrides['reliability_target']
            self.assertEqual(overrides['hpp_sizing_objective'], baseline.LCOE_OBJECTIVE)
            if target == 1.:
                raise RuntimeError('Synthetic unresolved run')
            model = self.candidate(2)
            design = [35,300,5,2,7,180,42,180,1.25,0,8]
            checks = baseline.validate_result(model,design)
            history = Path(overrides['hpp_sizing_output_dir'])/'native_history.csv'
            record = dict(status='FEASIBLE',design=design,validation=checks,economics=model.economics,
                          annualized_cost_hpp=1.1e7,started_at=0,completed_at=1)
            execution.write_evaluation_history([record],history,0,1,baseline.LCOE_OBJECTIVE,target)
            return dict(best=dict(design=design,validation=checks,metrics=checks['metrics'],economics=model.economics,
                                  lcoe_hpp_eur_per_mwh=baseline.objective_value(model.economics,baseline.LCOE_OBJECTIVE)),
                        history_path=str(history),configuration=overrides,final_status='VALIDATED_FEASIBLE_INCUMBENT',
                        termination_reason='TIME_LIMIT',optimization_seconds=1)
        with scratch_directory() as directory, patch.object(execution,'run_native_hpp_sizing',side_effect=fake_run):
            first = execution.run_lcoe_reliability_experiment(CONFIG,cfg,[.4,.999,1.],directory)
            snapshot = {p:p.read_bytes() for p in first.rglob('*') if p.is_file()}
            for name in ('reliability_040','reliability_099p9','reliability_100'):
                for file in ('result.json','configuration.json','evaluations.csv'):
                    self.assertTrue((first/name/file).exists())
            import pandas as pd
            summary = pd.read_csv(first/'summary.csv')
            self.assertEqual(len(summary),3)
            self.assertEqual(summary.iloc[-1]['feasibility_status'],'RUN_ERROR')
            self.assertTrue(pd.isna(summary.iloc[-1]['lcoe_hpp_eur_per_mwh']))
            second = execution.run_lcoe_reliability_experiment(CONFIG,cfg,[.7],directory)
            self.assertNotEqual(first,second)
            self.assertEqual(snapshot,{p:p.read_bytes() for p in snapshot})

    def test_frontier_notebook_compiles_and_load_mode_is_explicit(self):
        path = ROOT/'docs/notebooks/simple_datacenter_lcoe_reliability_with_hpp_sizing.ipynb'
        nb=json.loads(path.read_text(encoding='utf8'))
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        compile(code,str(path),'exec')
        self.assertLess(len(code.splitlines()),70)
        self.assertNotIn('def ',code)
        self.assertIn('if saved_experiment is None:',code)
        self.assertIn('load_lcoe_reliability_experiment(',code)
        self.assertIn('0.999, 1.00',code)
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])


def run_short_notebook():
    """Explicit 60-second test overrides, then two-worker normal completion."""
    import matplotlib
    from IPython.display import display
    matplotlib.use("Agg")
    source = ROOT / "hydesign/Parallel_EGO.py"
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf8"))
    namespace = {"__name__": "__main__", "display": display}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            print(f"Running baseline notebook cell {i}", flush=True)
            if "result = run_native_hpp_sizing(" in "".join(cell["source"]):
                # Never execute a user's production time budget from this test.
                config = namespace["config"]
                config.update(facility_load_mw=16, reliability_target=.99,
                              hpp_sizing_time_limit_seconds=60, n_doe=8, n_procs=2,
                              n_clusters=1, npred=100, n_seed=1, max_iter=30,
                              ems_threads=1, ems_time_limit_seconds=30)
                for name, limits in (("Nwt", [40,42]), ("solar_MW [MW AC]", [180,190]),
                                     ("b_P [MW]", [25,30]), ("b_E_h [h]", [8,10])):
                    config["variables"][name].update(var_type="design", limits=limits)
            exec(compile("".join(cell["source"]), f"baseline_cell_{i}", "exec"), namespace)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    timed = namespace["result"]
    assert timed["termination_reason"] == "TIME_LIMIT", timed
    assert 59 <= timed["optimization_seconds"] < 65, timed["optimization_seconds"]
    assert timed["best"] is not None, timed["validation_errors"]
    assert timed["process_cleanup"]["active_processes_after_stop"] == 0
    assert timed["process_cleanup"]["total_processes"] >= 2
    import pandas as pd
    history = pd.read_csv(timed["history_path"])
    accepted = history[history["accepted"]]
    np.testing.assert_allclose(accepted["G_proxy_MW"], 16+accepted["b_P"])
    np.testing.assert_allclose(accepted["CAPEX_sh_fixed"], 119940*accepted["G_proxy_MW"])
    np.testing.assert_allclose(accepted["CAPEX_sh"], accepted["CAPEX_sh_fixed"]+accepted["CAPEX_sh_land"])
    np.testing.assert_allclose(accepted["annualized_cost_shared"], accepted["CAPEX_sh"]*.06/(1-1.06**-25))
    np.testing.assert_allclose(accepted["battery_replacement_cost"], accepted["CAPEX_b"])
    np.testing.assert_allclose(accepted["discounted_battery_replacement_cost"], accepted["CAPEX_b"]/1.06**15)
    np.testing.assert_allclose(accepted["annualized_CAPEX_battery"],
        (accepted["CAPEX_b"]+accepted["discounted_battery_replacement_cost"])*.06/(1-1.06**-25))
    np.testing.assert_allclose(timed["best"]["economics"]["annualized_cost_hpp"],
                               accepted["annualized_cost_hpp"].min(), rtol=1e-9)
    config = execution.read_configuration(CONFIG)
    config.update(facility_load_mw=10, reliability_target=.99, hpp_sizing_time_limit_seconds=120,
                  hpp_sizing_output_dir=str(ROOT / "hpp_sizing_runs"), n_doe=2, n_procs=2,
                  ems_threads=1, max_iter=0)
    config["variables"]["Nwt"].update(var_type="design", limits=[40, 42])
    normal = execution.run_native_hpp_sizing(CONFIG, config)
    assert normal["termination_reason"] == "MAX_ITERATIONS", normal
    assert normal["optimization_seconds"] < 120
    assert normal["best"] is not None, normal["validation_errors"]
    assert normal["process_cleanup"]["active_processes_after_stop"] == 0
    assert normal["process_cleanup"]["total_processes"] >= 3
    for p in (Path(normal["run_directory"]) / "evaluations").glob("*.json"):
        record = json.loads(p.read_text())
        assert 40 <= record["design"][3] <= 42 and record["design"][3].is_integer()
        assert record["design"][5] == 180 and record["design"][-2:] == [25, 8]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    print("TIMED + NORMAL NATIVE EGO VALIDATION PASSED", flush=True)
    print("Timed result:", timed["run_directory"], flush=True)
    print("Normal result:", normal["run_directory"], flush=True)


if __name__ == "__main__":
    if "--integration" in sys.argv:
        run_short_notebook()
    else:
        unittest.main()
