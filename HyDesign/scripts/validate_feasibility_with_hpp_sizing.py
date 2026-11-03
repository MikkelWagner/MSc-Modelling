"""Short integration comparison. Explicitly never runs the 600-second benchmark."""
import contextlib
import io
import json
import multiprocessing
import os
from pathlib import Path

os.environ['OPENMDAO_REPORTS'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pandas as pd

from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost import hpp_model_simple_datacenter_with_hpp_cost
from hydesign.examples import examples_filepath
from hydesign.sizing_simple_datacenter_with_hpp_sizing import read_configuration, run_hpp_sizing, json_write

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml'


def main():
    settings = dict(hpp_sizing_time_limit_seconds=90, ems_time_limit_seconds=5,
                    n_doe=6, max_iter=2, npred=1000, n_procs=1, n_seed=0,
                    hpp_sizing_output_dir=str(ROOT/'hpp_sizing_runs'))
    report = {'purpose': 'short workflow test; not a convergence benchmark', 'overrides': settings, 'runs': {}}
    results = {}
    reliability_target = read_configuration(CONFIG)['reliability_target']
    for method in ('native_ego', 'feasibility_aware'):
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            result = run_hpp_sizing(CONFIG, dict(settings, optimization_method=method))
        results[method] = result
        directory = Path(result['run_directory'])
        (directory/'console_with_hpp_sizing.txt').write_text(console.getvalue(), encoding='utf8')
        history = pd.read_csv(result['history_path'])
        best = result['best']
        summary = {key: result[key] for key in ('termination_reason', 'error', 'elapsed_seconds',
                    'evaluated_designs', 'unique_designs', 'feasible_designs', 'run_directory', 'search_summary')}
        summary.update(best_Nwt=None if best is None else best['design'][3],
                       best_cost=None if best is None else best['economics']['annualized_cost_hpp'],
                       metrics=None if best is None else best['metrics'],
                       statuses=history.status.value_counts().to_dict(),
                       selection_reasons=history.selection_reason.value_counts().to_dict())
        report['runs'][method] = summary
        print(method, json.dumps(summary, default=str), flush=True)
        assert result['error'] is None, result['error']
        assert best is not None, 'Smoke test did not complete any feasible evaluation'
        assert result['final_validation']['max_failed_hours'] == int(np.floor((1-reliability_target)*8760 + 1e-9))
        assert result['final_validation']['metrics']['firm_reliability'] >= reliability_target - 1e-12
        assert len(history) == len(history.drop_duplicates(['Nwt', 'solar_MW', 'b_P', 'b_E_h']))

    # Compare the same selected candidate with the existing assembly directly.
    result = results['feasibility_aware']
    config = read_configuration(CONFIG)
    config.update(settings)
    sites = pd.read_csv(Path(examples_filepath)/'examples_sites.csv', sep=';')
    site = sites.loc[sites.name == config['hpp_sizing_site']].iloc[0]
    kwargs = dict(config, latitude=float(site.latitude), longitude=float(site.longitude),
                  altitude=float(site.altitude), input_ts_fn=str(Path(examples_filepath)/site.input_ts_fn), verbose=False)
    model = hpp_model_simple_datacenter_with_hpp_cost(sim_pars_fn=str(CONFIG), **kwargs)
    try:
        model.evaluate(*result['best']['design'])
        saved = np.load(Path(result['run_directory'])/'dispatch_with_hpp_sizing.npz')
        max_difference = {}
        for name, values in dict(model.hourly, soc_mwh=model.soc_mwh).items():
            np.testing.assert_allclose(values, saved[name], rtol=0, atol=1e-6)
            max_difference[name] = float(np.max(np.abs(np.asarray(values)-saved[name])))
        for name, value in model.economics.items():
            np.testing.assert_allclose(value, result['best']['economics'][name], rtol=1e-12, atol=1e-6)
        for name, value in model.metrics.items():
            np.testing.assert_allclose(value, result['best']['metrics'][name], rtol=1e-12, atol=1e-6)
        report['unchanged_assembly_comparison'] = dict(passed=True, design=result['best']['design'],
            max_hourly_differences=max_difference, economic_fields=len(model.economics), metric_fields=len(model.metrics))
    finally:
        model.prob.cleanup()
    path = ROOT/'docs/benchmarks/feasibility_validation_with_hpp_sizing.json'
    json_write(path, report)
    print('Validation report:', path, flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
