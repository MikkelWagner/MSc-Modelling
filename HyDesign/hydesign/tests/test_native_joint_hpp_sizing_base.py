"""Focused joint-interface tests; --integration adds annual CPLEX parity and short EGO."""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hydesign.assembly import hpp_assembly_simple_datacenter_joint_hpp_sizing_base as joint
from hydesign import sizing_native_joint_hpp_sizing_base as execution
from hydesign.tests.test_native_hpp_sizing_base import scratch_directory, CONFIG
from hydesign.costs.cost_simple_datacenter import datacenter_cost_annualized


def configuration():
    cfg = execution.prepare_configuration(CONFIG)
    cfg["variables"]["Nwt"]["value"] = 40
    cfg["variables"][joint.IT_NAME].update(var_type="fixed", value=16/1.15)
    cfg.update(reliability_target=.99, ems_threads=1)
    return cfg


def design_vector(cfg):
    return [cfg["variables"][name]["value"] for name in joint.VARIABLE_NAMES]


class FakeHPP:
    """Only isolates joint plumbing; real dispatch parity is tested separately."""
    def __init__(self, sim_pars_fn, variables, **kwargs):
        self.sim_pars = execution.read_configuration(sim_pars_fn)
        self.sim_pars.update(kwargs)
        self.cost_upper_bound = 1e7
        self.input_ts_fn, self.altitude = "weather.csv", 0
        self.list_out_vars = ["physical", "dispatch_validated [-]", "old score"]
        self.values = {"facility_load_mw": self.sim_pars["facility_load_mw"], "pue": self.sim_pars["pue"]}
        self.prob = SimpleNamespace(set_val=self.values.__setitem__)
        self.calls = []

    def evaluate(self, *design):
        self.calls.append(design)
        assert len(design) == 11
        load = self.values["facility_load_mw"]
        assert load == self.sim_pars["facility_load_mw"]
        it = load/self.values["pue"]
        cost = datacenter_cost_annualized(**{k: self.sim_pars[k] for k in joint.DC_COST_PARAMETERS}).compute(it)[-1]
        revenue = self.sim_pars["datacenter_firm_revenue_per_mwh_it"] * it * 8760
        self.economics = dict(annualized_cost_hpp=1e6, annualized_cost_dc=cost,
                             Revenue_dc=revenue, net_annual_value_system=revenue-cost-1e6)
        self.candidate_status, self.candidate_error, self.solve_observations = "FEASIBLE", None, []
        self.metrics = {"LOLH": 0, "firm_reliability": 1., "EENS": 0.}
        self.final_checks = {"metrics": self.metrics}
        self.validation, self.solver_status, self.hourly, self.soc_mwh = {}, {}, {}, np.zeros(8761)
        return np.array([1., 1., -1.])


def record(value, cost=1e6, it=16/1.15, status="FEASIBLE", when=1):
    cfg = configuration()
    design = design_vector(cfg)
    design[-1] = it
    e = dict(net_annual_value_system=value, annualized_cost_hpp=cost,
             annualized_cost_dc=2e6, Revenue_dc=value+cost+2e6)
    return dict(design=design, status=status, started_at=0, completed_at=when,
                facility_load_mw=it*1.15, annualized_cost_hpp=cost, economics=e,
                validation={"metrics": {"EENS": 1., "firm_reliability": .999, "LOLH": 8}})


class JointTests(unittest.TestCase):
    def test_configuration_and_notebook(self):
        original = execution.read_configuration(CONFIG)
        cfg = execution.prepare_configuration(CONFIG, original)
        self.assertNotIn(joint.IT_NAME, original["variables"])
        self.assertEqual(len(cfg["variables"]), 12)
        self.assertAlmostEqual(cfg["variables"][joint.IT_NAME]["limits"][1], 50/1.15)
        cfg["variables"][joint.IT_NAME]["limits"] = [0, 10]
        with self.assertRaises(ValueError):
            execution.prepare_configuration(CONFIG, cfg)
        path = execution.ROOT / "docs/notebooks/simple_datacenter_joint_hpp_sizing_base.ipynb"
        nb = json.loads(path.read_text())
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        compile(code, str(path), "exec")
        self.assertLess(len(code.splitlines()), 90)
        self.assertNotIn("def ", code)
        self.assertIn("config['reliability_target'] = 0.99", code)

    def test_negative_value_scoring_and_maximization(self):
        lower = dict(Revenue_dc=0., annualized_cost_hpp=4e6, annualized_cost_dc=6e6,
                     net_annual_value_system=-1e7)
        higher = dict(lower, Revenue_dc=2e6, net_annual_value_system=-8e6)
        self.assertEqual(joint.system_value_score(lower, 1e7), 1.)
        self.assertGreater(joint.system_value_score(higher, 1e7), joint.system_value_score(lower, 1e7))
        self.assertGreater(joint.system_value_score(lower, 1e7), -1.)
        with self.assertRaises(ValueError):
            joint.system_value_score(lower, 1e6)
        # Native EGO default minimizes the negative of the supplied scalar.
        source = (execution.ROOT / "hydesign/Parallel_EGO.py").read_text()
        self.assertIn("opt_sign = -1", source)
        self.assertNotEqual(joint.SCORE_NAME, "LCOE [Euro/MWh]")

    def test_candidate_sync_indexing_and_economic_response(self):
        cfg = configuration()
        cfg["variables"][joint.IT_NAME].update(var_type="design", limits=[1, 30])
        with patch.object(joint, "hpp_model_simple_datacenter_hpp_sizing_base", FakeHPP):
            model = joint.hpp_model_simple_datacenter_joint_hpp_sizing_base(
                CONFIG, cfg["variables"], pue=1.15)
            design = design_vector(cfg)
            model.evaluate(*design)
            first = model.economics.copy()
            self.assertAlmostEqual(model.sim_pars["facility_load_mw"], 16.)
            self.assertEqual(model.hpp.calls[-1][-2:], (25., 8.))
            design[-1] *= 2
            model.evaluate(*design)
            self.assertAlmostEqual(model.prob.set_val.__self__["facility_load_mw"], 32.)
            self.assertAlmostEqual(model.economics["Revenue_dc"], 2*first["Revenue_dc"])
            self.assertAlmostEqual(model.economics["annualized_cost_dc"], 2*first["annualized_cost_dc"])
            self.assertGreater(model.outputs[-1], model.rejection_score)

    def test_rejection_cannot_expose_stale_economics(self):
        cfg = configuration()
        with scratch_directory() as directory, patch.object(joint, "hpp_model_simple_datacenter_hpp_sizing_base", FakeHPP):
            model = joint.hpp_model_simple_datacenter_joint_hpp_sizing_base(
                CONFIG, cfg["variables"], baseline_log_dir=directory)
            d = design_vector(cfg)
            model.evaluate(*d)
            self.assertEqual(model.candidate_status, "FEASIBLE")
            def reject(*args):
                model.hpp.candidate_status = "PROVEN_INFEASIBLE"
                model.hpp.candidate_error = "infeasible"
                model.hpp.solve_observations = [{"status_code": 103, "has_solution": False}]
                return np.zeros(3)
            model.hpp.evaluate = reject
            model.evaluate(*d)
            self.assertEqual(model.candidate_status, "PROVEN_INFEASIBLE")
            self.assertIsNone(model.economics)
            self.assertEqual(model.outputs[-1], -1.)
            records = execution.read_evaluations(directory, "NORMAL_EXIT")
            self.assertEqual(sum(execution.accepted_record(r) for r in records), 1)

    def test_inner_bound_covers_it_upper_limit_above_yaml_load(self):
        cfg = configuration()
        cfg["variables"][joint.IT_NAME].update(var_type="design", limits=[1, 80])
        with patch.object(joint, "hpp_model_simple_datacenter_hpp_sizing_base", FakeHPP):
            model = joint.hpp_model_simple_datacenter_joint_hpp_sizing_base(CONFIG, cfg["variables"])
        variables = {k: cfg["variables"][k] for k in joint.HPP_VARIABLE_NAMES}
        expected = joint.annual_cost_bound(model.sim_pars, variables, facility_load_max_mw=80*1.15)
        original = joint.annual_cost_bound(model.sim_pars, variables)
        self.assertGreater(expected, original)
        self.assertEqual(model.hpp.cost_upper_bound, expected)
        self.assertGreater(model.total_cost_bound, expected)

    def test_history_tracks_maximum_value_not_minimum_cost(self):
        records = [record(-5e6, 1e6, when=1), record(7e6, 3e6, it=20, when=2),
                   record(1e12, status="PROVEN_INFEASIBLE", when=3)]
        with scratch_directory() as directory:
            path = Path(directory)/"history.csv"
            execution.write_evaluation_history(records, path, 0, 3)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[-1]["best_feasible_net_annual_value_system"]), 7e6)
            self.assertEqual(float(rows[-1]["best_feasible_cost"]), 3e6)
            self.assertEqual(float(rows[-1]["b_P"]), 25)
            self.assertEqual(float(rows[-1]["b_E_h"]), 8)
            self.assertEqual(float(rows[1]["it_capacity_mw"]), 20)
            self.assertEqual(rows[-1]["net_annual_value_system"], "")

    def test_recovery_passes_actual_load_and_exports_it_capacity(self):
        import pandas as pd
        calls = []
        def validate(records, inputs, directory):
            r = records[0]
            calls.append((r["design"], inputs["facility_load_mw"]))
            pd.DataFrame({"facility_load_mw": [inputs["facility_load_mw"]]*8760}).to_csv(
                Path(directory)/"hourly_hpp_sizing_base.csv", index=False)
            return dict(economics=dict(r["economics"], it_capacity_mw=inputs["facility_load_mw"]/inputs["pue"]),
                        design=r["design"]), [], .1
        candidates = [record(1, 1, when=1), record(2, 10, it=20, when=2),
                      record(1e10, status="TIME_LIMIT_WITH_INCUMBENT", when=3)]
        with scratch_directory() as directory, patch.object(execution, "validate_fixed_candidate", validate):
            best, errors, _ = execution.independently_validate(candidates, {"pue": 1.15}, directory)
            self.assertEqual(best["economics"]["net_annual_value_system"], 2)
            self.assertEqual(len(calls[0][0]), 11)
            self.assertAlmostEqual(calls[0][1], 23.)
            hourly = pd.read_csv(Path(directory)/"hourly_joint_hpp_sizing_base.csv")
            np.testing.assert_allclose(hourly["facility_load_mw"], 23.)
            np.testing.assert_allclose(hourly["it_capacity_mw"], 20.)
            self.assertFalse(errors)

    def test_no_feasible_candidate_means_no_best(self):
        with scratch_directory() as directory, patch.object(execution, "validate_fixed_candidate") as validate:
            best, _, _ = execution.independently_validate([record(1e20, status="PROVEN_INFEASIBLE")], {}, directory)
            self.assertIsNone(best)
            validate.assert_not_called()


def run_integration():
    """Real annual parity, then at most 60 s of five-variable native EGO search."""
    from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost import hpp_model_simple_datacenter_with_hpp_cost
    import pandas as pd
    cfg = configuration()
    with scratch_directory() as directory:
        inputs = execution.physical_inputs(CONFIG, cfg, Path(directory))
        fixed = hpp_model_simple_datacenter_with_hpp_cost(**inputs, verbose=False)
        model = joint.hpp_model_simple_datacenter_joint_hpp_sizing_base(
            **inputs, variables=cfg["variables"], verbose=False)
        try:
            d = design_vector(cfg)
            fixed.evaluate(*d[:-1])
            model.evaluate(*d)
            assert model.candidate_status == "FEASIBLE", model.candidate_error
            for key in fixed.hourly:
                np.testing.assert_allclose(model.hourly[key], fixed.hourly[key], atol=1e-5, rtol=1e-8)
            np.testing.assert_allclose(model.soc_mwh, fixed.soc_mwh, atol=1e-5, rtol=1e-8)
            for key in fixed.economics:
                np.testing.assert_allclose(model.economics[key], fixed.economics[key], atol=1e-4, rtol=1e-9)
            assert model.metrics["LOLH"] == fixed.metrics["LOLH"]
            print("Annual fixed-load parity PASSED:", model.metrics["firm_reliability"], flush=True)
        finally:
            fixed.prob.cleanup()
            model.prob.cleanup()
    # Narrow test bounds exercise all five native variable types without spending
    # the production budget in mostly infeasible regions. Notebook bounds stay broad.
    cfg.update(hpp_sizing_time_limit_seconds=60, n_doe=8, n_seed=1, n_clusters=1,
               n_procs=2, npred=100, max_iter=1, ems_time_limit_seconds=30,
               hpp_sizing_output_dir=str(execution.ROOT/"joint_hpp_dc_sizing_runs"))
    for name, limits in (("Nwt", [40, 42]), ("solar_MW [MW AC]", [180, 190]),
                         ("b_P [MW]", [25, 30]), ("b_E_h [h]", [8, 10]),
                         (joint.IT_NAME, [8, 14])):
        cfg["variables"][name].update(var_type="design", limits=limits,
                                     types="int" if name == "Nwt" else "float")
    result = execution.run_native_joint_hpp_sizing(CONFIG, cfg)
    assert result["best"] is not None, result
    assert result["best"]["metrics"]["LOLH"] <= 87
    assert result["process_cleanup"]["active_processes_after_stop"] == 0
    history = pd.read_csv(result["history_path"])
    assert history["it_capacity_mw"].nunique() > 1
    assert history["Nwt"].map(float.is_integer).all()
    best = result["best"]
    accepted = history[history["accepted"]]
    np.testing.assert_allclose(accepted["G_proxy_MW"], accepted["facility_load_mw"]+accepted["b_P"])
    np.testing.assert_allclose(accepted["CAPEX_sh_fixed"], 119940*accepted["G_proxy_MW"])
    np.testing.assert_allclose(accepted["CAPEX_sh"], accepted["CAPEX_sh_fixed"]+accepted["CAPEX_sh_land"])
    np.testing.assert_allclose(accepted["annualized_cost_shared"], accepted["CAPEX_sh"]*.06/(1-1.06**-25))
    np.testing.assert_allclose(accepted["battery_replacement_cost"], accepted["CAPEX_b"])
    np.testing.assert_allclose(accepted["discounted_battery_replacement_cost"], accepted["CAPEX_b"]/1.06**15)
    np.testing.assert_allclose(accepted["annualized_CAPEX_battery"],
        (accepted["CAPEX_b"]+accepted["discounted_battery_replacement_cost"])*.06/(1-1.06**-25))
    hourly = pd.read_csv(Path(result["run_directory"])/"hourly_joint_hpp_sizing_base.csv")
    np.testing.assert_allclose(hourly["facility_load_mw"], best["design"][-1]*cfg["pue"])
    np.testing.assert_allclose(hourly["it_capacity_mw"], best["design"][-1])
    values = history.loc[history["accepted"], "net_annual_value_system"]
    np.testing.assert_allclose(best["economics"]["net_annual_value_system"], values.max(), rtol=1e-6, atol=1.)
    assert (Path(result["run_directory"])/"history_joint_hpp_sizing_base.png").exists()
    execution.print_results(result)
    print("Native five-variable integration PASSED", flush=True)


if __name__ == "__main__":
    if "--integration" in sys.argv:
        run_integration()
    else:
        unittest.main()

