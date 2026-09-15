"""Small off-grid wind/PV/battery sizing test with a hard reliability filter."""

import argparse
import os
from pathlib import Path
from time import perf_counter

# Prevent OpenMDAO from creating unrelated report files during this test.
os.environ["OPENMDAO_REPORTS"] = "0"

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from hydesign.examples import examples_filepath
from hydesign.hpp_evaluator_test_simple_datacenter import SimpleDatacenterHPP


WORKSPACE_DIR = Path(__file__).resolve().parents[3]


def build_evaluator(reliability_target, ems_time_limit_s=None):
    sites = pd.read_csv(
        Path(examples_filepath) / "examples_sites.csv", index_col=0, sep=";"
    )
    site = sites.loc[sites["name"] == "France_good_wind"].iloc[0]
    return SimpleDatacenterHPP(
        sim_pars_fn=Path(examples_filepath) / site["sim_pars_fn"],
        input_ts_fn=Path(examples_filepath) / site["input_ts_fn"],
        reliability_target=reliability_target,
        load_mw=10.0,
        ems_time_limit_s=ems_time_limit_s,
        work_dir=WORKSPACE_DIR / "results" / "simple_datacenter_test",
    )


def make_design(candidate):
    wind_turbines, solar_mw, battery_power_mw, battery_duration_h = candidate
    return [
        35.0,  # clearance [m], fixed for this first test
        300.0,  # specific power [W/m2], fixed
        5.0,  # rated turbine power [MW], fixed
        round(wind_turbines),
        7.0,  # wind installation density [MW/km2], fixed
        solar_mw,
        25.0,  # PV tilt [deg], fixed
        180.0,  # PV azimuth [deg], fixed
        1.0,  # DC/AC ratio, fixed
        battery_power_mw,
        battery_duration_h,
        0.0,  # market battery fluctuation cost, unused by the new EMS
    ]


def run(reliability_target, maxiter, popsize, ems_time_limit_s=None):
    started = perf_counter()
    evaluator = build_evaluator(reliability_target, ems_time_limit_s)
    bounds = [
        (0, 200),  # number of 5 MW wind turbines
        (0, 500),  # PV capacity [MW]
        (0, 200),  # battery power [MW]
        (1, 24),  # battery duration [h]
    ]

    def objective(candidate):
        evaluator.evaluate(*make_design(candidate))
        print(f"Candidate {sum(evaluator.solve_counts.values())}: "
              f"{evaluator.operation['status']}, CAPEX={evaluator.outputs[0]:.3f} MEuro, "
              f"LOLH={evaluator.operation['lolh']}, "
              f"EMS={evaluator.operation['runtime_s']:.2f} s", flush=True)
        return evaluator.objective

    result = differential_evolution(
        objective,
        bounds=bounds,
        integrality=[True, False, False, False],
        maxiter=maxiter,
        popsize=popsize,
        seed=0,
        polish=False,
        workers=1,
        updating="immediate",
    )

    design = make_design(result.x)
    outputs = evaluator.evaluate(*design)
    output = pd.Series(outputs, index=evaluator.list_out_vars)
    output["Reliability target [-]"] = reliability_target
    output["Status"] = evaluator.operation["status"]
    output["Solver status"] = evaluator.operation["solver_status"]
    output["EMS runtime [s]"] = evaluator.operation["runtime_s"]
    output["Sizing runtime [s]"] = perf_counter() - started
    output["Outer optimizer converged"] = bool(result.success)
    output["Outer optimizer message"] = str(result.message)
    for status in ("feasible", "infeasible", "inconclusive"):
        output[f"EMS solves: {status}"] = evaluator.solve_counts[status]
    output["Simultaneous charge/discharge hours"] = evaluator.operation["simultaneous_hours"]
    output["Maximum simultaneous charge/discharge [MW]"] = evaluator.operation["max_simultaneous_mw"]
    output["Candidates with simultaneous charge/discharge"] = evaluator.simultaneous_candidates
    output["Maximum simultaneous charge/discharge across candidates [MW]"] = evaluator.max_simultaneous_mw
    output_path = WORKSPACE_DIR / "results" / "result_test_simple_datacenter.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, header=["value"])

    print("Best feasible design found:" if np.isfinite(evaluator.objective)
          else "No feasible design found; last selected candidate:")
    print(pd.Series(design, index=evaluator.list_vars))
    print("\nResults:")
    print(output)
    print(f"Firm reliability: {100 * evaluator.operation['reliability']:.4f} %")
    print(f"CPLEX statuses: {dict(evaluator.solver_status_counts)}")
    print(f"\nSaved result to: {output_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reliability-target", type=float, default=0.999)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument("--ems-time-limit-s", type=float, default=120,
                        help="Optional limit per EMS solve; unresolved solves are reported separately")
    args = parser.parse_args()
    if not 0 < args.reliability_target <= 1:
        raise ValueError("reliability target must be in the interval (0, 1]")
    run(args.reliability_target, args.maxiter, args.popsize, args.ems_time_limit_s)
