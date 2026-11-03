"""Controlled, serial CPLEX formulation benchmarks; no MIP starts or tuning.

Run from the repository root with hydesign_clean Python. Results, source hashes,
inputs, CPLEX logs and validated dispatch arrays are written to --output.
The incumbent scalar is a caller-certified objective for these exact inputs,
not a supplied solution; it is used ONLY by pruning variants C/F/G.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

os.environ.setdefault('OPENMDAO_REPORTS', '0')
os.environ.setdefault('MPLBACKEND', 'Agg')

import cplex
import numpy as np
import pandas as pd
from docplex.mp.progress import ProgressListener
from docplex.mp.relax_linear import LinearRelaxer

from hydesign.assembly.hpp_assembly_simple_datacenter import hpp_model_simple_datacenter
from hydesign.examples import examples_filepath
import hydesign.sizing_simple_datacenter_with_dc_cost as baseline
import hydesign.sizing_simple_datacenter_with_dc_cost_strengthened as experimental


ROOT = Path(__file__).resolve().parents[1]
COST_KEYS = (
    'datacenter_facility_capex_per_mw_it', 'datacenter_it_capex_per_mw_it',
    'datacenter_fixed_opex_per_mw_it_year', 'datacenter_discount_rate',
    'datacenter_facility_lifetime_years', 'datacenter_it_lifetime_years',
    'datacenter_firm_revenue_per_mwh_it',
)
BASELINE_FILES = (
    'hydesign/sizing_simple_datacenter_with_dc_cost.py',
    'hydesign/sizing_simple_datacenter.py',
    'hydesign/assembly/hpp_assembly_simple_datacenter.py',
    'hydesign/assembly/hpp_assembly_simple_datacenter_with_dc_cost.py',
    'hydesign/ems/ems_simple_datacenter.py',
    'hydesign/metrics_simple_datacenter.py',
    'hydesign/costs/cost_simple_datacenter.py',
    'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml',
    'docs/notebooks/simple_datacenter_sizing_with_dc_cost.ipynb',
    'docs/notebooks/simple_datacenter_with_dc_cost.ipynb',
)


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in BASELINE_FILES}


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class PrimaryProgress(ProgressListener):
    """Read-only information listener; identical instrumentation in every run."""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.phase = 0
        self.tree_started = False
        self.values = dict(first_incumbent_seconds=None, root_bound_after_cuts_eur=None,
                           time_to_5pct=None, time_to_2pct=None, time_to_1pct=None)

    def notify_start(self):
        super().notify_start()
        self.phase += 1

    def notify_progress(self, data):
        if self.phase != 1:
            return
        if data.current_nb_nodes > 0:
            self.tree_started = True
        if not self.tree_started and abs(data.best_bound) < 1e20:
            self.values['root_bound_after_cuts_eur'] = data.best_bound * self.scale
        if data.has_incumbent:
            if self.values['first_incumbent_seconds'] is None:
                self.values['first_incumbent_seconds'] = data.time
            for threshold, key in ((.05, 'time_to_5pct'), (.02, 'time_to_2pct'), (.01, 'time_to_1pct')):
                if data.mip_gap <= threshold and self.values[key] is None:
                    self.values[key] = data.time


def run_variant(name, inputs, options, output):
    module = baseline if name == 'baseline' else experimental
    real_solver = module.solve_cplex_simple_datacenter_with_dc_cost
    observation = {'variant': name, 'options': options, 'time_limit_seconds': inputs['time_limit_seconds']}
    log = io.StringIO()
    beginning = time.perf_counter()

    def observed_solver(mdl, variables, economics, load_variable, time_limit_seconds=600,
                        threads=4, log_output=False, mip_relative_gap=1e-4,
                        allow_time_limit_incumbent=False, **extra):
        observation.update(
            construction_and_bound_seconds=time.perf_counter() - beginning,
            continuous_variables=mdl.number_of_continuous_variables,
            binary_variables=mdl.number_of_binary_variables,
            constraints=mdl.number_of_constraints,
            fixed_failure_binaries=sum(z.ub == 0 for z in variables['failed_hour']),
            big_m_mw=load_variable.ub,
            effective_lower_mw=load_variable.lb,
            explicit_mip_starts=mdl.number_of_mip_starts,
        )
        assert mdl.number_of_mip_starts == 0
        scale = extra.get('primary_objective_scale', 1.0)
        mdl.maximize(economics['net_annual_value'] / scale)  # Source expression for the clone.
        # A separate clone supplies the LP diagnostic; it never initializes the MIP.
        with LinearRelaxer.make_relaxed_model(mdl) as lp:
            # Add ramp auxiliaries only to the diagnostic clone here. The real
            # solver builds its own once; do not change/duplicate physical rows.
            ramp_variables = {
                key: [lp.get_var_by_name(v.name) for v in variables[key]]
                for key in ('P_charge', 'P_discharge')
            }
            ramp = module.battery_power_fluctuation_expression(
                lp, ramp_variables, extra.get('battery_power_fluctuation_penalty', 0.0))
            ramp_price = (extra.get('battery_power_fluctuation_penalty', 0.0)
                          * extra.get('datacenter_firm_revenue_per_mwh_it', 0.0))
            lp.maximize(lp.objective_expr - ramp_price * ramp / scale)
            lp.parameters.threads = threads
            lp.parameters.timelimit = time_limit_seconds
            lp.parameters.simplex.tolerances.feasibility = 1e-9
            lp.parameters.simplex.tolerances.optimality = 1e-9
            sol = lp.solve()
            observation['root_lp_status'] = lp.solve_details.status
            observation['root_lp_seconds'] = lp.solve_details.time
            observation['root_lp_eur'] = sol.objective_value * scale if sol is not None else None
        progress = PrimaryProgress(scale)
        mdl.add_progress_listener(progress)
        try:
            return real_solver(mdl, variables, economics, load_variable, time_limit_seconds,
                               threads, log, mip_relative_gap, allow_time_limit_incumbent, **extra)
        except Exception:
            # Retain the API bound even when no incumbent exists and the original
            # solver deliberately raises. No dispatch is fabricated or accepted.
            details = mdl.solve_details
            observation['last_solve_failure'] = {
                'status': details.status, 'status_code': details.status_code,
                'seconds': details.time, 'best_bound_solver_units': details.best_bound,
            }
            if progress.phase == 1:
                observation['best_bound_eur'] = details.best_bound * scale
            raise
        finally:
            observation.update(progress.values)
            observation.update(
                continuous_variables=mdl.number_of_continuous_variables,
                binary_variables=mdl.number_of_binary_variables,
                constraints=mdl.number_of_constraints,
            )

    print(f'START {name}: {options}', flush=True)
    try:
        with patch.object(module, 'solve_cplex_simple_datacenter_with_dc_cost', observed_solver):
            result = module.size_datacenter_with_dc_cost(**inputs, **options)
        primary = result['solver_status']['stages'][0]
        observation.update(
            primary=primary,
            primary_status=primary['status'], primary_seconds=primary['seconds'],
            incumbent_eur=primary['objective'], best_bound_eur=primary.get('best_bound'),
            gap=primary.get('relative_gap'), facility_load_mw=result['facility_load_mw'],
            it_capacity_mw=result['it_capacity_mw'], failed_hours=result['metrics']['LOLH'],
            reliability=result['metrics']['firm_reliability'],
            validation=result['validation'], validation_passed=True,
            optimal=result['solver_status']['optimal'], metrics=result['metrics'],
            economics=result['economics'], formulation=result.get('formulation'),
        )
        bound_seconds = (result.get('formulation') or {}).get('bound_details', {}).get('seconds', 0)
        observation['bound_presolve_seconds'] = bound_seconds
        observation['construction_seconds'] = observation['construction_and_bound_seconds'] - bound_seconds
        if result['solver_status']['optimal']:
            observation['time_to_optimal_seconds'] = primary['seconds']
        # Progress callbacks may miss the final bound update. Distinguish a known
        # upper bound on crossing time from an actually observed crossing time.
        for threshold, key in ((.05, 'time_to_5pct'), (.02, 'time_to_2pct'), (.01, 'time_to_1pct')):
            if observation[key] is None and primary.get('relative_gap', float('inf')) <= threshold:
                observation[key + '_upper_bound_seconds'] = primary['seconds']
        np.savez_compressed(output / f'{name}_dispatch.npz', **result['hourly'],
                            soc_mwh=result['soc_mwh'], permitted_failure=result['permitted_failure'])
    except Exception as exc:
        observation.update(error=repr(exc), validation_passed=False)
    observation['wall_seconds'] = time.perf_counter() - beginning
    log_text = log.getvalue()
    # Read the single dispatch MIP log.
    primary_log = log_text.split('Version identifier:')[1] if 'Version identifier:' in log_text else log_text
    counts = re.findall(r'Reduced MIP has (\d+) binaries', primary_log)
    observation['presolved_binaries'] = int(counts[-1]) if counts else None
    # The log gives the first-incumbent timestamp more precisely than sparse callbacks.
    first = re.search(r'Found incumbent .*? after ([\d.]+) sec', primary_log)
    if first:
        observation['first_incumbent_seconds'] = float(first.group(1))
    observation['dynamic_search_logged'] = 'MIP search method: dynamic search' in primary_log
    observation['mip_start_log_present'] = 'MIP start' in primary_log
    (output / f'{name}.log').write_text(log_text, encoding='utf-8')
    observation = clean(observation)
    (output / f'{name}.json').write_text(json.dumps(observation, indent=2), encoding='utf-8')
    print('RESULT ' + json.dumps({k: observation.get(k) for k in (
        'variant', 'root_lp_eur', 'incumbent_eur', 'best_bound_eur', 'gap',
        'facility_load_mw', 'failed_hours', 'presolved_binaries', 'validation_passed', 'error')}), flush=True)
    return observation


def save_summary(output, records):
    (output / 'results.json').write_text(json.dumps(clean(records), indent=2), encoding='utf-8')
    columns = ('variant', 'construction_and_bound_seconds', 'continuous_variables',
               'binary_variables', 'constraints', 'fixed_failure_binaries', 'presolved_binaries',
               'big_m_mw', 'effective_lower_mw', 'root_lp_eur', 'root_bound_after_cuts_eur',
               'first_incumbent_seconds', 'incumbent_eur', 'best_bound_eur', 'gap',
               'facility_load_mw', 'it_capacity_mw', 'failed_hours', 'reliability',
               'validation_passed', 'optimal', 'time_to_5pct', 'time_to_2pct', 'time_to_1pct',
               'time_to_optimal_seconds', 'wall_seconds', 'construction_seconds',
               'bound_presolve_seconds', 'primary_status', 'primary_seconds')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--incumbent-eur', type=float, default=None,
                        help='Certified same-case feasible objective; required for C/F/G (not a MIP start).')
    parser.add_argument('--variants', nargs='+', default=['baseline', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])
    parser.add_argument('--long-best', type=float, default=0,
                        help='Optional longer cold solve of the lowest-gap validated short variant.')
    args = parser.parse_args()
    if any(v in args.variants for v in ('C', 'F', 'G')) and args.incumbent_eur is None:
        parser.error('C/F/G require --incumbent-eur for the exact current case; no default certificate is assumed')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        parser.error('Choose a fresh output directory to preserve earlier benchmark evidence')
    before = hashes()
    site_table = pd.read_csv(examples_filepath + 'examples_sites.csv', sep=';', index_col=0)
    site = site_table.loc[site_table.name == 'Denmark_good_solar'].iloc[0]
    hpp = hpp_model_simple_datacenter(
        latitude=site.latitude, longitude=site.longitude, altitude=site.altitude,
        work_dir='./', sim_pars_fn=examples_filepath+'thesis_simple_datacenter/simple_datacenter.yml',
        input_ts_fn=examples_filepath+site.input_ts_fn, facility_load_mw=16, pue=1.15, ppa_price=70,
    )
    design = [35, 300, 5, 20, 7, 180, 42, 180, 1.25, 25, 8]
    hpp.prob.set_val('reliability_target', 0)
    hpp.evaluate(*design)
    pars = hpp.sim_pars
    inputs = dict(wind_ts=hpp.prob.get_val('wind_t').copy(), solar_ts=hpp.prob.get_val('solar_t').copy(),
                  b_P=25, b_E=200, pue=1.15, time_limit_seconds=args.seconds,
                  threads=pars['ems_threads'], mip_relative_gap=pars['sizing_mip_relative_gap'],
                  allow_time_limit_incumbent=True,
                  battery_power_fluctuation_penalty=pars.get("battery_power_fluctuation_penalty", 0.0),
                  **{k: pars[k] for k in COST_KEYS},
                  **{k: pars[k] for k in ('battery_depth_of_discharge', 'battery_charge_efficiency',
                     'battery_initial_soc', 'reliability_target', 'facility_load_min_mw',
                     'facility_load_max_mw', 'epsilon_mw')})
    hpp.prob.cleanup()
    scalar_inputs = {k: v for k, v in inputs.items() if not isinstance(v, np.ndarray)}
    manifest = dict(source_hashes_before=before, cplex_version=cplex.Cplex().get_version(),
                    design=design, inputs=scalar_inputs, incumbent_certificate_eur=args.incumbent_eur,
                    generation_sha256={k: hashlib.sha256(inputs[k].tobytes()).hexdigest()
                                       for k in ('wind_ts', 'solar_ts')},
                    explicit_mip_starts=False,
                    benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    experimental_source_sha256=hashlib.sha256(Path(experimental.__file__).read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(clean(manifest), indent=2), encoding='utf-8')
    off = dict(tighten_load_upper_bound=False, aggregate_shortage=False, hourly_strengthening=False)
    variants = {
        'baseline': {}, 'A': dict(off, tighten_load_upper_bound=True),
        'B': dict(off, aggregate_shortage=True),
        'C': dict(off, incumbent_objective_eur=args.incumbent_eur, hourly_strengthening=True),
        'D': dict(off, primary_objective_scale=1e6),
        'E': dict(off, tighten_load_upper_bound=True, aggregate_shortage=True),
        'F': dict(tighten_load_upper_bound=True, aggregate_shortage=True,
                  hourly_strengthening=True, incumbent_objective_eur=args.incumbent_eur),
    }
    records = []
    for name in args.variants:
        if name == 'G':
            # Add scaling to the strongest tested unscaled combination, rather than
            # assuming every proposed inequality improves branch-and-bound behavior.
            combinations = [r for r in records if r['variant'] in ('E', 'F')
                            and r.get('validation_passed') and r.get('gap') is not None]
            chosen = min(combinations, key=lambda r: r['gap'])['variant'] if combinations else 'F'
            variants['G'] = dict(variants[chosen], primary_objective_scale=1e6)
            print('G scales combination ' + chosen, flush=True)
        if name not in variants:
            parser.error('Unknown variant: ' + name)
        records.append(run_variant(name, inputs, variants[name], output))
        save_summary(output, records)
    if args.long_best:
        candidates = [r for r in records if r['variant'] != 'baseline' and r.get('validation_passed')
                      and r.get('gap') is not None]
        if candidates:
            best = min(candidates, key=lambda r: r['gap'])
            records.append(run_variant(best['variant']+'_long',
                           dict(inputs, time_limit_seconds=args.long_best), best['options'], output))
            save_summary(output, records)
    after = hashes()
    manifest.update(source_hashes_after=after, original_files_unchanged=before == after)
    (output / 'manifest.json').write_text(json.dumps(clean(manifest), indent=2), encoding='utf-8')
    if before != after:
        raise RuntimeError('Original source/input files changed during the benchmark')


if __name__ == '__main__':
    main()
