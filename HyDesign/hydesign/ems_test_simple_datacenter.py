"""Off-grid EMS for the first simple data-center sizing test."""

from decimal import Decimal, ROUND_FLOOR
from time import perf_counter

import numpy as np
import cplex
from scipy.sparse import lil_matrix


class OffgridReliabilityEMS:
    """Minimize EENS, then battery throughput, within the hourly outage budget."""

    def __init__(self, load_mw=10.0, reliability_target=0.90,
                 epsilon_mw=1e-3, time_limit_s=None):
        self.load_mw = float(load_mw)
        self.reliability_target = float(reliability_target)
        self.epsilon_mw = float(epsilon_mw)
        self.time_limit_s = time_limit_s
        if not 0 < self.reliability_target <= 1:
            raise ValueError("Reliability target must be in (0, 1]")
        if not 0 <= self.epsilon_mw < self.load_mw:
            raise ValueError("Require 0 <= epsilon < load")
        if time_limit_s is not None and time_limit_s <= 0:
            raise ValueError("CPLEX time limit must be positive")

    def outage_budget(self, n_hours):
        # Decimal avoids floor(875.9999999999998) for a 90% target.
        return int(((Decimal(1) - Decimal(str(self.reliability_target)))
                    * n_hours).to_integral_value(rounding=ROUND_FLOOR))

    def solve(
        self,
        wind_mw,
        solar_mw,
        battery_power_mw,
        battery_energy_mwh,
        battery_depth_of_discharge=0.9,
        charge_efficiency=0.985,
    ):
        started = perf_counter()
        wind_mw = np.asarray(wind_mw, dtype=float).reshape(-1)
        solar_mw = np.asarray(solar_mw, dtype=float).reshape(-1)
        if len(wind_mw) != len(solar_mw):
            raise ValueError("Wind and solar profiles must have the same length")

        n_time = len(wind_mw)
        if n_time == 0 or not all(np.all(np.isfinite(p)) and np.all(p >= 0)
                                  for p in (wind_mw, solar_mw)):
            raise ValueError("Generation profiles must be nonempty, finite and nonnegative")
        if not (np.isfinite(battery_power_mw) and battery_power_mw >= 0
                and np.isfinite(battery_energy_mwh) and battery_energy_mwh >= 0
                and 0.5 <= battery_depth_of_discharge <= 1
                and 0 < charge_efficiency <= 1):
            raise ValueError("Invalid battery parameters (initial SOC is fixed at 50%)")
        budget = self.outage_budget(n_time)
        time = range(n_time)
        minimum_soc = (1.0 - battery_depth_of_discharge) * battery_energy_mwh
        initial_soc = 0.5 * battery_energy_mwh

        # Variable blocks: charge, discharge, curtailment, unserved, and SoC.
        charge_start = 0
        discharge_start = n_time
        curtailment_start = 2 * n_time
        unserved_start = 3 * n_time
        soc_start = 4 * n_time
        outage_start = 5 * n_time + 1
        n_variables = outage_start + n_time

        objective = np.zeros(n_variables)
        objective[unserved_start : unserved_start + n_time] = 1.0

        equality = lil_matrix((2 * n_time + 2, n_variables))
        rhs = np.zeros(2 * n_time + 2)
        row = 0
        for t in time:
            equality[row, charge_start + t] = -1.0
            equality[row, discharge_start + t] = 1.0
            equality[row, curtailment_start + t] = -1.0
            equality[row, unserved_start + t] = 1.0
            rhs[row] = self.load_mw - wind_mw[t] - solar_mw[t]
            row += 1

        equality[row, soc_start] = 1.0
        rhs[row] = initial_soc
        row += 1
        for t in time:
            equality[row, soc_start + t + 1] = 1.0
            equality[row, soc_start + t] = -1.0
            equality[row, charge_start + t] = -charge_efficiency
            equality[row, discharge_start + t] = 1.0 / charge_efficiency
            row += 1
        equality[row, soc_start + n_time] = 1.0
        rhs[row] = initial_soc

        bounds = (
            [(0.0, battery_power_mw)] * n_time
            + [(0.0, battery_power_mw)] * n_time
            + [(0.0, float(wind_mw[t] + solar_mw[t])) for t in time]
            + [(0.0, self.load_mw)] * n_time
            + [(minimum_soc, battery_energy_mwh)] * (n_time + 1)
            + [(0.0, 1.0)] * n_time
        )
        matrix = equality.tocsr()
        with cplex.Cplex() as model:
            model.set_log_stream(None)
            model.set_results_stream(None)
            model.parameters.threads.set(1)
            model.parameters.simplex.tolerances.feasibility.set(1e-9)
            model.parameters.mip.tolerances.integrality.set(1e-9)
            model.parameters.mip.tolerances.mipgap.set(0.0)
            model.parameters.mip.tolerances.absmipgap.set(0.0)
            if self.time_limit_s is not None:
                model.parameters.timelimit.set(float(self.time_limit_s))
            model.objective.set_sense(model.objective.sense.minimize)
            model.variables.add(
                obj=objective.tolist(),
                lb=[float(lower) for lower, _ in bounds],
                ub=[float(upper) for _, upper in bounds],
                types="C" * outage_start + "B" * n_time,
            )
            model.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(
                        ind=matrix.indices[matrix.indptr[r] : matrix.indptr[r + 1]].tolist(),
                        val=matrix.data[matrix.indptr[r] : matrix.indptr[r + 1]].tolist(),
                    )
                    for r in range(matrix.shape[0])
                ],
                senses="E" * matrix.shape[0],
                rhs=rhs.tolist(),
            )
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=[unserved_start + t, outage_start + t],
                    val=[1.0, -(self.load_mw - self.epsilon_mw)],
                ) for t in time] + [cplex.SparsePair(
                    ind=list(range(outage_start, n_variables)),
                    val=[1.0] * n_time,
                )],
                senses="L" * (n_time + 1),
                rhs=[self.epsilon_mw] * n_time + [float(budget)],
            )
            # One API call, two internal priority stages. Neither objective
            # minimizes LOLH; the outage budget remains a hard constraint.
            model.multiobj.set_num(2)
            model.multiobj.set_definition(
                0, obj=cplex.SparsePair(
                    ind=list(range(unserved_start, unserved_start + n_time)),
                    val=[1.0] * n_time,
                ), priority=2, abstol=0.0, reltol=0.0, name="EENS_MWh",
            )
            model.multiobj.set_definition(
                1, obj=cplex.SparsePair(
                    ind=list(range(2 * n_time)), val=[1.0] * (2 * n_time),
                ), priority=1, abstol=0.0, reltol=0.0, name="throughput_MWh",
            )
            model.solve()
            code = model.solution.get_status()
            solver_status = model.solution.get_status_string()
            multi = model.solution.multiobj
            stages = {}
            for i in range(multi.get_num_solves()):
                priority = multi.get_info(i, multi.int_info.priority)
                stage_code = multi.get_info(i, multi.int_info.status)
                name = "eens" if priority == 2 else "throughput"
                stages[name] = {
                    "status": model.solution.get_status_string(stage_code),
                    "optimal": stage_code == model.solution.status.MIP_optimal,
                    "runtime_s": multi.get_info(i, multi.float_info.time),
                }
                if stages[name]["optimal"]:
                    stages[name]["objective_mwh"] = multi.get_info(i, multi.float_info.objective)
            stage_summary = {
                "objective_stages": stages,
                "eens_optimal": stages.get("eens", {}).get("optimal", False),
                "throughput_optimal": stages.get("throughput", {}).get("optimal", False),
            }
            if code != model.solution.status.multiobj_optimal:
                # No fallback solve: infeasible candidates have no dispatch metrics.
                # An unfinished throughput stage is distinct from infeasibility,
                # even when the primary EENS stage was already proven optimal.
                return {
                    "status": ("infeasible" if code == model.solution.status.multiobj_infeasible
                               else "inconclusive"),
                    **stage_summary,
                    "solver_status": solver_status,
                    "runtime_s": perf_counter() - started,
                    "outage_budget_h": budget,
                    **{key: np.nan for key in (
                        "reliability", "lolh", "eens_mwh", "maximum_shortfall_mw",
                        "energy_served_mwh", "simultaneous_hours",
                        "max_simultaneous_mw", "throughput_mwh")},
                }
            solution = np.asarray(model.solution.get_values())
        charge_mw = solution[charge_start : charge_start + n_time]
        discharge_mw = solution[discharge_start : discharge_start + n_time]
        curtailment_mw = solution[curtailment_start : curtailment_start + n_time]
        unserved_mw = solution[unserved_start : unserved_start + n_time]
        soc_mwh = solution[soc_start : soc_start + n_time + 1]
        served_mw = np.maximum(self.load_mw - unserved_mw, 0.0)
        eens_mwh = float(unserved_mw.sum())
        # Canonicalize only solver roundoff at the threshold, not physical deficits.
        # Raw dispatch is retained for energy accounting and balance checks.
        shortfall_for_count = unserved_mw.copy()
        shortfall_for_count[np.abs(shortfall_for_count - self.epsilon_mw) <= 1e-7] = self.epsilon_mw
        lolh = int(np.count_nonzero(shortfall_for_count > self.epsilon_mw))
        reliability = 1.0 - lolh / n_time
        indicators = np.rint(solution[outage_start:]).astype(int)
        if (lolh > budget or indicators.sum() > budget
                or np.any((shortfall_for_count > self.epsilon_mw) & (indicators == 0))):
            raise RuntimeError("CPLEX dispatch failed the firm-reliability consistency check")
        overlap = np.minimum(charge_mw, discharge_mw)
        # EENS must not be traded for throughput, even across priority stages.
        primary_eens = stages["eens"]["objective_mwh"]
        if abs(eens_mwh - primary_eens) > 1e-6:
            raise RuntimeError("Throughput optimization changed primary EENS by more than 1e-6 MWh")

        return {
            **stage_summary,
            "status": "feasible",
            "solver_status": solver_status,
            "runtime_s": perf_counter() - started,
            "outage_budget_h": budget,
            "lolh": lolh,
            "maximum_shortfall_mw": float(unserved_mw.max()),
            "simultaneous_hours": int(np.count_nonzero(overlap > 1e-6)),
            "max_simultaneous_mw": float(overlap.max()),
            "throughput_mwh": float((charge_mw + discharge_mw).sum()),
            "wind_mw": wind_mw,
            "solar_mw": solar_mw,
            "charge_mw": charge_mw,
            "discharge_mw": discharge_mw,
            "curtailment_mw": curtailment_mw,
            "unserved_mw": unserved_mw,
            "served_mw": served_mw,
            "soc_mwh": soc_mwh,
            "eens_mwh": eens_mwh,
            "energy_served_mwh": float(served_mw.sum()),
            "reliability": float(reliability),
            "reliability_target": self.reliability_target,
        }
