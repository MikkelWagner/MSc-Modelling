"""Controlled full-year smoke checks, deliberately shorter than the 600 s run."""
from pathlib import Path
import json
import copy
import os
import time
from unittest.mock import patch

os.environ['OPENMDAO_REPORTS'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
from docplex.mp.model import Model
from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost import hpp_model_simple_datacenter_with_hpp_cost
from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_sizing import hpp_model_simple_datacenter_with_hpp_sizing
from hydesign.sizing_simple_datacenter_with_hpp_sizing import (
    run_hpp_sizing, read_configuration, prepare_variables, native_cost_bound,
)


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root/'hydesign/examples/thesis_simple_datacenter/simple_datacenter.yml'
    config = read_configuration(config_path)
    bound = native_cost_bound(config, prepare_variables(config))
    kwargs = dict(sim_pars_fn=str(config_path), latitude=55.39776, longitude=11.813965, altitude=42,
                  input_ts_fn=str(root/'hydesign/examples/Europe/GWA2/input_ts_Denmark_good_solar.csv'),
                  facility_load_mw=16, pue=1.15, reliability_target=0, verbose=False)
    design = [35,300,5,20,7,180,42,180,1.25,25,8]
    report = {}
    # Explicit unconstrained regression check only; never seeds the sizing search.
    old = hpp_model_simple_datacenter_with_hpp_cost(**kwargs)
    new = hpp_model_simple_datacenter_with_hpp_sizing(**kwargs, hpp_cost_upper_bound=bound,
                                                     hpp_rejection_score=-(bound*1.01)/1e6)
    old.evaluate(*design)
    new.evaluate(*design)
    assert new.candidate_status == 'FEASIBLE', new.candidate_error
    for key in old.hourly: np.testing.assert_allclose(old.hourly[key],new.hourly[key],rtol=0,atol=1e-8)
    for key in old.economics: assert old.economics[key] == new.economics[key], key
    report['regression_diagnostic_only'] = {'reliability':new.metrics['firm_reliability'],
        'identical_economics':True,'identical_dispatch':True,'used_as_sizing_incumbent':False}
    print('A: original reference dispatch/economics reproduced',flush=True)
    old.prob.cleanup();new.prob.cleanup()
    kwargs['reliability_target'] = config['reliability_target']
    bad = hpp_model_simple_datacenter_with_hpp_sizing(**kwargs,hpp_cost_upper_bound=bound,
                                                     hpp_rejection_score=-(bound*1.01)/1e6)
    empty = list(design);empty[3]=empty[5]=empty[-2]=0
    bad.evaluate(*empty)
    assert bad.candidate_status == 'PROVEN_INFEASIBLE', bad.candidate_error
    report['zero_generation']={'status':bad.candidate_status,'observations':bad.solve_observations}
    print('B: zero-generation case correctly rejected',flush=True)
    bad.prob.cleanup()
    # Controlled timeout exercises the real assembly/EMS exception path and structured observer.
    timeout = hpp_model_simple_datacenter_with_hpp_sizing(**kwargs,hpp_cost_upper_bound=bound,
                                                         hpp_rejection_score=-(bound*1.01)/1e6)
    original_solve = Model.solve
    def report_timeout(model,*args,**kw):
        sol = original_solve(model,*args,**kw)
        model._solve_details._solve_status_code = 107 if sol is not None else 108
        model._solve_details._solve_status = 'controlled time limit'
        return sol
    rich = list(design);rich[3]=400
    with patch.object(Model,'solve',report_timeout): timeout.evaluate(*rich)
    assert timeout.candidate_status in ('TIME_LIMIT_WITH_FEASIBLE_DISPATCH','TIME_LIMIT_UNRESOLVED')
    report['controlled_timeout']={'status':timeout.candidate_status,'accepted':False}
    print('C: controlled timeout is not physical infeasibility',flush=True)
    timeout.prob.cleanup()
    result = run_hpp_sizing(config_path,dict(hpp_sizing_time_limit_seconds=90, ems_time_limit_seconds=15,
        n_doe=3,max_iter=1,n_clusters=1,npred=300,n_procs=1,
        hpp_sizing_output_dir=str(root/'docs/benchmarks/runs_with_hpp_sizing')))
    report['short_search']=result
    print('D: short search',result['termination_reason'],result['error'],flush=True)
    assert result['termination_reason'] != 'error', result['error']
    short = run_hpp_sizing(config_path,dict(hpp_sizing_time_limit_seconds=.01,
        hpp_sizing_output_dir=str(root/'docs/benchmarks/runs_with_hpp_sizing')))
    assert short['termination_reason']=='outer_time_limit'
    assert short['best'] is None and short['evaluated_designs']==0
    report['short_deadline']=short
    # A real two-worker deadline test starts with an explicitly revalidated rich
    # design. Reference and upper scout may coincide; parent recording remains safe.
    variables = copy.deepcopy(config['variables'])
    variables['Nwt']['value'] = 400
    timed = run_hpp_sizing(config_path,dict(hpp_sizing_time_limit_seconds=25,ems_time_limit_seconds=5,
        variables=variables,n_doe=8,max_iter=20,n_clusters=1,npred=300,n_procs=2,
        hpp_sizing_output_dir=str(root/'docs/benchmarks/runs_with_hpp_sizing')))
    assert timed['termination_reason']=='outer_time_limit',timed['error']
    assert timed['best'] is not None
    report['deadline_with_incumbent']=timed
    # Wind-only domain with no PV/storage; all tested cases genuinely infeasible.
    variables = copy.deepcopy(config['variables'])
    variables['Nwt'].update(value=0,limits=[0,1])
    variables['solar_MW [MW AC]']['value'] = 0
    variables['b_P [MW]']['value'] = 0
    none = run_hpp_sizing(config_path,dict(hpp_sizing_time_limit_seconds=30,ems_time_limit_seconds=5,
        variables=variables,n_doe=3,max_iter=1,n_clusters=1,npred=300,n_procs=1,
        hpp_sizing_output_dir=str(root/'docs/benchmarks/runs_with_hpp_sizing')))
    assert none['best'] is None and none['termination_reason'] != 'error',none['error']
    report['no_feasible_search']=none
    print('E: deadline stops launches safely',flush=True)
    destination = root/'docs/benchmarks/validation_with_hpp_sizing.json'
    destination.write_text(json.dumps(report,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),encoding='utf8')
    print('Report:',destination,flush=True)


if __name__=='__main__': main()
