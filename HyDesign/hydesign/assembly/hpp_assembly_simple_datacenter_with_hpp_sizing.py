"""Thin fixed-DC adapter; generation, dispatch and costs are inherited unchanged."""

import time
from unittest.mock import patch

import numpy as np
from docplex.mp.model import Model

from hydesign.assembly.hpp_assembly_simple_datacenter_with_hpp_cost import (
    hpp_model_simple_datacenter_with_hpp_cost,
)
import hydesign.ems.ems_simple_datacenter as native_ems


SCORE_NAME = "hpp_sizing_score [MEUR/year]"


def normalize_turbines(value):
    value = float(value)
    if not np.isfinite(value) or value < 0 or abs(value - round(value)) > 1e-9:
        raise ValueError("Nwt must be a nonnegative integer (roundoff tolerance 1e-9)")
    return int(round(value))


def sizing_score(cost):
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("Annual HPP cost must be finite and nonnegative")
    return -float(cost) / 1e6


def failure_status(observations, error):
    """Classify structured CPLEX details, never parse exception message strings."""
    has_incumbent = any(item["has_solution"] for item in observations)
    cause = error
    while cause is not None:
        if isinstance(cause, TimeoutError):
            return "TIME_LIMIT_WITH_FEASIBLE_DISPATCH" if has_incumbent else "TIME_LIMIT_UNRESOLVED"
        cause = cause.__cause__
    if observations:
        last = observations[-1]
        code = last["status_code"]
        if code in (3, 103):
            # A later failed solve cannot prove the original physical problem
            # infeasible after an earlier feasible solution (legacy observations).
            if any(item["has_solution"] for item in observations[:-1]):
                return "NUMERICAL_ERROR"
            return "PROVEN_INFEASIBLE"
        if code in (11, 107, 108):
            return "TIME_LIMIT_WITH_FEASIBLE_DISPATCH" if has_incumbent else "TIME_LIMIT_UNRESOLVED"
        if code not in (1, 101, 102):
            return "NUMERICAL_ERROR"
    return "CONFIGURATION_OR_PROGRAMMING_ERROR"


class hpp_model_simple_datacenter_with_hpp_sizing(hpp_model_simple_datacenter_with_hpp_cost):
    """Add status and a negative-cost score without replacing any model equations.

    A scoped Model subclass observes solve status and caps each solve by the
    remaining outer budget. The original EMS Model binding is restored on exit.
    Each process evaluates one candidate at a time. Timed-out incumbents are
    recorded but conservatively NOT accepted: the single dispatch solve and
    unchanged physical validation must finish successfully.
    """

    def __init__(self, sim_pars_fn, hpp_deadline=None, hpp_rejection_score=None,
                 hpp_cost_upper_bound=None, **kwargs):
        super().__init__(sim_pars_fn, **kwargs)
        self.hpp_deadline = hpp_deadline
        self.hpp_rejection_score = hpp_rejection_score
        self.hpp_cost_upper_bound = hpp_cost_upper_bound
        self.list_out_vars = list(self.list_out_vars) + [SCORE_NAME]
        self.candidate_status = None
        self.candidate_error = None
        self.solve_observations = []

    def evaluate(self, *design):
        if self.hpp_rejection_score is None or not np.isfinite(self.hpp_rejection_score):
            raise ValueError("Provide the bounds-derived finite hpp_rejection_score")
        self.candidate_error = None
        self.solve_observations = []
        adapter = self

        class ObservedModel(Model):
            def solve(model, *args, **kwargs):
                if adapter.hpp_deadline is not None:
                    remaining = adapter.hpp_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Outer HPP sizing deadline reached before CPLEX solve")
                    model.parameters.timelimit = min(model.parameters.timelimit.get(), remaining)
                sol = super().solve(*args, **kwargs)
                details = model.solve_details
                adapter.solve_observations.append({
                    "status_code": details.status_code, "status": details.status,
                    "seconds": details.time, "has_solution": sol is not None,
                })
                return sol

        try:
            design = list(design)
            design[3] = normalize_turbines(design[3])
            if self.hpp_deadline is not None and time.monotonic() >= self.hpp_deadline:
                raise TimeoutError("Outer deadline reached before candidate evaluation")
            with patch.object(native_ems, "Model", ObservedModel):
                outputs = super().evaluate(*design)
            cost = self.economics["annualized_cost_hpp"]
            if self.hpp_cost_upper_bound is None or cost > self.hpp_cost_upper_bound + 1e-6:
                raise ValueError("Native HPP cost exceeded configured domain bound; do not continue search")
            if self.metrics["firm_reliability"] < self.sim_pars["reliability_target"] - 1e-12:
                raise ValueError("Validated dispatch failed the configured hourly reliability target")
            self.candidate_status = "FEASIBLE"
            self.outputs = np.append(outputs, sizing_score(cost))
        except Exception as exc:
            self.candidate_status = failure_status(self.solve_observations, exc)
            self.candidate_error = f"{type(exc).__name__}: {exc}"
            self.outputs = np.zeros(len(self.list_out_vars))
            self.outputs[-1] = self.hpp_rejection_score
        return self.outputs
