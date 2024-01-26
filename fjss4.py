from __future__ import annotations

import collections
import math
import os
import random
import itertools as it
from abc import ABC
from collections import defaultdict
from itertools import product, combinations

import gurobipy as gp
import numpy as np
from docplex.mp.model import Model
from gurobipy import GRB
from monty.json import MSONable
from pydantic import BaseModel
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from utils import get_m_value_old, get_m_value_runzhong

# from reaction_network.schema.lv2 import BenchTopLv2, OperationType

# os.environ["GRB_LICENSE_FILE"] = "/home/qai/local/gurobi_lic/gurobi.lic"


def get_gurobi_var_index(
    var_name: str,
    var_header: str,
) -> int | list[int]:
    assert var_name.startswith(var_header)
    assert var_name.endswith("]")
    stripped_index_string = var_name[len(var_header) :].lstrip("[").rstrip("]")
    n_index = var_name.count(",") + 1
    assert n_index >= 1
    if n_index == 1:
        i = int(stripped_index_string)
    else:
        i = stripped_index_string.split(",")
        i = [int(ii) for ii in i]
    return i


class SolvedOperation(BaseModel):
    id: str
    assigned_to: str
    start_time: float
    end_time: float


class FjsOutput(BaseModel):
    solved_operations: list[SolvedOperation]
    makespan: float

    @property
    def machine_timetable(self):
        timetable = defaultdict(list)
        for op in self.solved_operations:
            timetable[op.assigned_to].append((op.start_time, op.end_time))
        return timetable


def get_dummy_time_est(operation_type: OperationType):
    # in seconds
    if operation_type in [
        OperationType.OPERATION_LOADING,
        OperationType.OPERATION_UNLOADING,
        OperationType.OPERATION_RELOADING,
    ]:
        base_cost = 120
    elif operation_type in [OperationType.OPERATION_HEATING]:
        base_cost = 30 * 60
    elif operation_type in [OperationType.OPERATION_ADDITION_LIQUID]:
        base_cost = 2 * 60
    elif operation_type in [OperationType.OPERATION_ADDITION_SOLID]:
        base_cost = 5 * 60
    elif operation_type in [OperationType.OPERATION_PURIFICATION]:
        base_cost = 45 * 60
    else:
        raise NotImplementedError
    return random.uniform(base_cost * 0.8, base_cost * 1.2)


class _FJS(MSONable, ABC):
    def __init__(
        self,
        operations: list[str],
        machines: list[str],
        precedence: dict[str, list[str]],
        time_estimates: dict[str, dict[str, float]] | None,
        model_string: str | None = None,
    ):
        """
        # TODO implement io (loading lp file)

        :param operations: the list of operation ids
        :param machines: the list of machine ids
        :param precedence: p[i] = a list of j where j precedes i
        :param time_estimates: t[i][k] is the time cost of operation i on machine k, inf if i cannot be done by k
        :param model_string: dumped cplex model string (lp file)
        """
        self.model_string = model_string
        self.time_estimates = time_estimates
        self.precedence = precedence
        self.machines = machines
        self.operations = operations
        self._model = None

    def get_machine_ids_can_process(self, operation_id: str) -> list[str]:
        assert operation_id in self.operations
        amids = []
        for machine_id in self.machines:
            if not math.isinf(self.time_estimates[operation_id][machine_id]):
                amids.append(machine_id)
        return amids

    @classmethod
    def from_bench_top(
        cls, bench_top: BenchTopLv2, time_estimates: dict[str, dict[str, float]] = None
    ):
        p = dict()
        for operation_id, operation in bench_top.operation_dict.items():
            p[operation_id] = operation.precedents

        machines = []
        for v in bench_top.operation_type_device_mapping.values():
            machines += v
        machines = sorted(set(machines))
        if time_estimates is None:
            # generate dummy time estimate
            time_est = defaultdict(dict)
            random.seed(42)
            for operation_id, operation in bench_top.operation_dict.items():
                can_be_processed_by = bench_top.operation_type_device_mapping[
                    operation.type
                ]
                for mid in machines:
                    if mid in can_be_processed_by:
                        time_est[operation_id][mid] = get_dummy_time_est(operation.type)
                    else:
                        time_est[operation_id][mid] = math.inf
        else:
            time_est = time_estimates
        return cls(
            operations=sorted(bench_top.operation_dict.keys()),
            machines=machines,
            precedence=p,
            time_estimates=time_est,
        )


# the reimplementation of the FJSS3 model
class FJSS4(_FJS):
    """
    Implementation of the mixed integer programming formulation in:
    Boyer, V., Vallikavungal, J., Rodríguez, X. C., & Salazar-Aguilar, M. A. (2021). The generalized flexible job shop scheduling problem. Computers & Industrial Engineering, 160, 107542.
    """

    def __init__(
        self,
        operations: list[str],
        machines: list[str],
        para_p: np.ndarray,
        para_a: np.ndarray,
        para_w: np.ndarray,
        para_h: np.ndarray,
        para_delta: np.ndarray,
        para_mach_capacity: list[int] | np.ndarray,
        para_lmin: np.ndarray,
        para_lmax: np.ndarray,
        precedence: dict[str, list[str]],
        model_string: str | None = None,
        num_workers: int = 16,
        inf_milp: float = 1.0e7,
        big_m: float | int = None,
        verbose: bool = True,
    ):
        self.num_workers = num_workers
        # self.big_m = get_m_value(
        #     para_p=para_p, para_h=para_h, para_lmin=para_lmin, para_a=para_a
        # )

        para_lmin[para_lmin == -np.inf] = -inf_milp
        para_lmin[para_lmin == np.inf] = inf_milp
        para_lmax[para_lmax == np.inf] = inf_milp
        para_lmax[para_lmax == -np.inf] = -inf_milp
        para_p[para_p == np.inf] = inf_milp
        para_p[para_p == -np.inf] = -inf_milp
        para_w[para_w == np.inf] = inf_milp
        para_w[para_w == -np.inf] = -inf_milp
        para_a[para_a == np.inf] = inf_milp
        para_a[para_a == -np.inf] = -inf_milp

        self.inf_milp = inf_milp

        super().__init__(
            operations=operations,
            machines=machines,
            precedence=precedence,
            time_estimates=None,
            model_string=model_string,
        )

        self.para_p = para_p
        self.para_a = para_a
        self.para_w = para_w
        self.para_delta = para_delta
        self.para_lmin = para_lmin
        self.para_lmax = para_lmax
        self.para_h = para_h
        self.para_mach_capacity = para_mach_capacity

        if big_m is None:
            self.big_m = get_m_value_runzhong(
                para_p=para_p,
                para_h=para_h,
                para_lmin=para_lmin,
                para_a=para_a,
                infinity=inf_milp,
            )
            print(f"the inferred big_m value with Runzhong version is {self.big_m}")
            # self.big_m = 8306

        # # this part works because the leak of infinity to big_m
        # if big_m is None:
        #     self.big_m = get_m_value_old(
        #         para_p=para_p, para_h=para_h, para_lmin=para_lmin, para_a=para_a
        #     )
        else:
            self.big_m = big_m

        self.horizon = self.get_horizon()
        # self.big_m = self.horizon

        print(f"big_m: {self.big_m}")

        self.verbose = verbose
        self.var_c_max = None
        self.var_y = None
        self.var_c = None
        self.var_s = None
        self.var_x = None
        self.var_z = None
        self.model = None

    def build_model_gurobi(self):
        """Build the mixed integer linear programming model with gurobi."""

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()

        model = gp.Model("fjss_mip")
        n_opt, n_mach = self.get_params()

        # create variables
        # if operation i is processed by machine m
        var_y = model.addVars(n_opt, n_mach, vtype=GRB.BINARY, name="var_y")
        # if operation i is processed before operation j
        var_x = model.addVars(n_opt, n_opt, vtype=GRB.BINARY, name="var_x")
        # if operation i is processed before and overlapped operation j in machine m
        var_z = model.addVars(n_opt, n_opt, n_mach, vtype=GRB.BINARY, name="var_z")
        # starting time of operation i
        var_s = model.addVars(n_opt, vtype=GRB.CONTINUOUS, name="var_s")
        # completion time of operation i
        var_c = model.addVars(n_opt, vtype=GRB.CONTINUOUS, name="var_c")

        # objective
        # var_c_max = model.addVar(
        #     lb=1.0e-5, ub=self.horizon + 1, vtype=GRB.CONTINUOUS, name="var_c_max"
        # )
        var_c_max = model.addVar(lb=1.0e-5, name="var_c_max", vtype=GRB.CONTINUOUS)

        # var_c_max = model.addVar(
        #     name="var_c_max", lb=1e-5, ub=float("inf"), vtype=GRB.CONTINUOUS
        # )

        # add constraints
        for i in range(n_opt):
            # eq. (2)
            model.addConstr(var_c_max >= var_c[i], name="eq_2")
            # eq. (3)
            model.addConstr(
                var_c[i]
                >= var_s[i]
                + gp.quicksum(self.para_p[i, m] * var_y[i, m] for m in range(n_mach)),
                name="eq_3",
            )
            # eq. (4)
            model.addConstr(
                var_c[i]
                <= var_s[i]
                + gp.quicksum(
                    (self.para_p[i, m] + self.para_h[i, m]) * var_y[i, m]
                    for m in range(n_mach)
                ),
                name="eq_4",
            )
            # eq. (5)
            model.addConstr(
                gp.quicksum(var_y[i, m] for m in range(n_mach)) == 1, name="eq_5"
            )

        for i, j in it.product(range(n_opt), range(n_opt)):
            if i != j:
                # eq. (6)
                model.addConstr(
                    var_s[j] >= var_c[i] + self.para_lmin[i, j], name="eq_6"
                )
                # eq. (7)
                model.addConstr(
                    var_s[j] <= var_c[i] + self.para_lmax[i, j], name="eq_7"
                )

        for i, j, m in it.product(range(n_opt), range(n_opt), range(n_mach)):
            if i < j:
                expr_0 = self.big_m * (3 - var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (8)
                model.addConstr(
                    var_s[j] >= var_c[i] + self.para_a[i, j, m] - expr_0, name="eq_8"
                )
                # eq. (9)
                expr_1 = self.big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])
                model.addConstr(
                    var_s[i] >= var_c[j] + self.para_a[j, i, m] - expr_1, name="eq_9"
                )
                # eq. (10)
                model.addConstr(
                    var_s[j] >= var_s[i] + self.para_delta[m] - expr_0, name="eq_10"
                )
                # eq. (11)
                model.addConstr(
                    var_s[i] >= var_s[j] + self.para_delta[m] - expr_1, name="eq_11"
                )
                # eq. (12)
                model.addConstr(
                    var_c[j] >= var_c[i] + self.para_delta[m] - expr_0, name="eq_12"
                )
                # eq. (13)
                model.addConstr(
                    var_c[i] >= var_c[j] + self.para_delta[m] - expr_1, name="eq_13"
                )
                # eq. (14)
                expr_2 = self.big_m * (
                    3 + var_z[i, j, m] - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                model.addConstr(var_s[j] >= var_c[i] - expr_2, name="eq_14")
                # eq. (15)
                expr_3 = self.big_m * (
                    2 + var_z[j, i, m] + var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                model.addConstr(var_s[i] >= var_c[j] - expr_3, name="eq_15")

        # eq. (16)
        for i, m in it.product(range(n_opt), range(n_mach)):
            model.addConstr(
                gp.quicksum(
                    self.para_w[j, m] * var_z[i, j, m] for j in range(n_opt) if i != j
                )
                <= (self.para_mach_capacity[m] - self.para_w[i, m]) * var_y[i, m],
                name="eq_16",
            )

        # set the objective
        model.setObjective(var_c_max, GRB.MINIMIZE)

        self.var_c_max = var_c_max
        self.var_y = var_y
        self.var_c = var_c
        self.var_s = var_s
        self.var_x = var_x
        self.var_z = var_z
        self.model = model
        env.close()

    def solve_gurobi(self):
        """Solve the mixed integer linear programming model with gurobi."""
        # creates the solver and solve
        if self.model is None:
            self.build_model_gurobi()

        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            print(f"the solution is : {self.model.objVal}")

            return FjsOutput(
                solved_operations=[],
                makespan=self.model.objVal,
            )
        else:
            print("No solution found.")
            return None

    def get_params(self):
        """Get parameters for the model."""
        n_opt = len(self.operations)
        n_mach = len(self.machines)

        return n_opt, n_mach

    def get_horizon(self):
        """Get the horizon."""
        # the horizon
        para_p_horizon = np.copy(self.para_p)
        para_p_horizon[para_p_horizon == self.inf_milp] = 0

        para_h_horizon = np.copy(self.para_h)
        para_h_horizon[para_h_horizon == self.inf_milp] = 0

        para_lmax_horizon = np.copy(self.para_lmax)
        para_lmax_horizon[para_lmax_horizon == self.inf_milp] = 0
        horizon = (
            np.max(para_p_horizon, axis=1)
            + np.max(para_h_horizon, axis=1)
            + np.max(para_lmax_horizon, axis=1)
        )

        # print("")
        # # print(f"para_p_horizon = {para_p_horizon}")
        # # print(f"para_h_horizon = {para_h_horizon}")
        # print(f"para_lmax_horizon = {para_lmax_horizon}")
        # print(f"para_lmax_horizon max is {np.max(para_lmax_horizon)}")

        horizon = np.sum(horizon) + 1
        # print(f"\nhorizon={horizon}")

        return horizon


# ================================
class FJSS4_v2:
    """
    Implementation of the mixed integer programming formulation in:
    Boyer, V., Vallikavungal, J., Rodríguez, X. C., & Salazar-Aguilar, M. A. (2021). The generalized flexible job shop scheduling problem. Computers & Industrial Engineering, 160, 107542.
    """

    def __init__(
        self,
        operations: list[str],
        machines: list[str],
        para_p: np.ndarray,
        para_a: np.ndarray,
        para_w: np.ndarray,
        para_h: np.ndarray,
        para_delta: np.ndarray,
        para_mach_capacity: list[int] | np.ndarray,
        para_lmin: np.ndarray,
        para_lmax: np.ndarray,
        precedence: dict[str, list[str]],
        model_string: str | None = None,
        num_workers: int = 16,
        inf_milp: float = 1.0e7,
        # big_m: float | int = None,
        big_m=1.0e6,
        verbose: bool = True,
    ):
        self.num_workers = num_workers
        # self.big_m = get_m_value(
        #     para_p=para_p, para_h=para_h, para_lmin=para_lmin, para_a=para_a
        # )

        para_lmin[para_lmin == -np.inf] = -inf_milp
        para_lmin[para_lmin == np.inf] = inf_milp
        para_lmax[para_lmax == np.inf] = inf_milp
        para_lmax[para_lmax == -np.inf] = -inf_milp
        para_p[para_p == np.inf] = inf_milp
        para_p[para_p == -np.inf] = -inf_milp
        para_w[para_w == np.inf] = inf_milp
        para_w[para_w == -np.inf] = -inf_milp
        para_a[para_a == np.inf] = inf_milp
        para_a[para_a == -np.inf] = -inf_milp

        self.inf_milp = inf_milp

        self.operations = operations
        self.machines = machines
        self.precedence = precedence
        # self.time_estimates = None
        self.model_string = model_string

        self.para_p = para_p
        self.para_a = para_a
        self.para_w = para_w
        self.para_delta = para_delta
        self.para_lmin = para_lmin
        self.para_lmax = para_lmax
        self.para_h = para_h
        self.para_mach_capacity = para_mach_capacity

        if big_m is None:
            self.big_m = get_m_value_runzhong(
                para_p=para_p,
                para_h=para_h,
                para_lmin=para_lmin,
                para_a=para_a,
                infinity=inf_milp,
            )
            print(f"the inferred big_m value with Runzhong version is {self.big_m}")
            # self.big_m = 8306

        # # this part works because the leak of infinity to big_m
        # if big_m is None:
        #     self.big_m = get_m_value_old(
        #         para_p=para_p, para_h=para_h, para_lmin=para_lmin, para_a=para_a
        #     )
        else:
            self.big_m = big_m

        # self.horizon = self.get_horizon()
        # self.big_m = self.horizon

        print(f"big_m: {self.big_m}")

        self.verbose = verbose
        self.var_c_max = None
        self.var_y = None
        self.var_c = None
        self.var_s = None
        self.var_x = None
        self.var_z = None
        self.model = None

    def build_model_gurobi(self):
        """Build the mixed integer linear programming model with gurobi."""

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()

        model = gp.Model("fjss_mip")
        n_opt, n_mach = self.get_params()

        # create variables
        # if operation i is processed by machine m
        var_y = model.addVars(n_opt, n_mach, vtype=GRB.BINARY, name="var_y")
        # if operation i is processed before operation j
        var_x = model.addVars(n_opt, n_opt, vtype=GRB.BINARY, name="var_x")
        # if operation i is processed before and overlapped operation j in machine m
        var_z = model.addVars(n_opt, n_opt, n_mach, vtype=GRB.BINARY, name="var_z")
        # starting time of operation i
        var_s = model.addVars(n_opt, vtype=GRB.CONTINUOUS, name="var_s")
        # completion time of operation i
        var_c = model.addVars(n_opt, vtype=GRB.CONTINUOUS, name="var_c")

        # objective
        # var_c_max = model.addVar(
        #     lb=1.0e-5, ub=self.horizon + 1, vtype=GRB.CONTINUOUS, name="var_c_max"
        # )
        var_c_max = model.addVar(lb=1.0e-5, name="var_c_max", vtype=GRB.CONTINUOUS)

        # var_c_max = model.addVar(
        #     name="var_c_max", lb=1e-5, ub=float("inf"), vtype=GRB.CONTINUOUS
        # )

        # add constraints
        for i in range(n_opt):
            # eq. (2)
            model.addConstr(var_c_max >= var_c[i], name="eq_2")
            # eq. (3)
            model.addConstr(
                var_c[i]
                >= var_s[i]
                + gp.quicksum(self.para_p[i, m] * var_y[i, m] for m in range(n_mach)),
                name="eq_3",
            )
            # eq. (4)
            model.addConstr(
                var_c[i]
                <= var_s[i]
                + gp.quicksum(
                    (self.para_p[i, m] + self.para_h[i, m]) * var_y[i, m]
                    for m in range(n_mach)
                ),
                name="eq_4",
            )
            # eq. (5)
            model.addConstr(
                gp.quicksum(var_y[i, m] for m in range(n_mach)) == 1, name="eq_5"
            )

        for i, j in it.product(range(n_opt), range(n_opt)):
            if i != j:
                # eq. (6)
                model.addConstr(
                    var_s[j] >= var_c[i] + self.para_lmin[i, j], name="eq_6"
                )
                # eq. (7)
                model.addConstr(
                    var_s[j] <= var_c[i] + self.para_lmax[i, j], name="eq_7"
                )

        for i, j, m in it.product(range(n_opt), range(n_opt), range(n_mach)):
            if i < j:
                expr_0 = self.big_m * (3 - var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (8)
                model.addConstr(
                    var_s[j] >= var_c[i] + self.para_a[i, j, m] - expr_0, name="eq_8"
                )
                # eq. (9)
                expr_1 = self.big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])
                model.addConstr(
                    var_s[i] >= var_c[j] + self.para_a[j, i, m] - expr_1, name="eq_9"
                )
                # eq. (10)
                model.addConstr(
                    var_s[j] >= var_s[i] + self.para_delta[m] - expr_0, name="eq_10"
                )
                # eq. (11)
                model.addConstr(
                    var_s[i] >= var_s[j] + self.para_delta[m] - expr_1, name="eq_11"
                )
                # eq. (12)
                model.addConstr(
                    var_c[j] >= var_c[i] + self.para_delta[m] - expr_0, name="eq_12"
                )
                # eq. (13)
                model.addConstr(
                    var_c[i] >= var_c[j] + self.para_delta[m] - expr_1, name="eq_13"
                )
                # eq. (14)
                expr_2 = self.big_m * (
                    3 + var_z[i, j, m] - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                model.addConstr(var_s[j] >= var_c[i] - expr_2, name="eq_14")
                # eq. (15)
                expr_3 = self.big_m * (
                    2 + var_z[j, i, m] + var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                model.addConstr(var_s[i] >= var_c[j] - expr_3, name="eq_15")

        # eq. (16)
        for i, m in it.product(range(n_opt), range(n_mach)):
            model.addConstr(
                gp.quicksum(
                    self.para_w[j, m] * var_z[i, j, m] for j in range(n_opt) if i != j
                )
                <= (self.para_mach_capacity[m] - self.para_w[i, m]) * var_y[i, m],
                name="eq_16",
            )

        # set the objective
        model.setObjective(var_c_max, GRB.MINIMIZE)

        self.var_c_max = var_c_max
        self.var_y = var_y
        self.var_c = var_c
        self.var_s = var_s
        self.var_x = var_x
        self.var_z = var_z
        self.model = model
        env.close()

    def solve_gurobi(self):
        """Solve the mixed integer linear programming model with gurobi."""
        # creates the solver and solve
        if self.model is None:
            self.build_model_gurobi()

        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            print(f"the solution is : {self.model.objVal}")

            return FjsOutput(
                solved_operations=[],
                makespan=self.model.objVal,
            )
        else:
            print("No solution found.")
            return None

    def get_params(self):
        """Get parameters for the model."""
        n_opt = len(self.operations)
        n_mach = len(self.machines)

        return n_opt, n_mach

    # def get_horizon(self):
    #     """Get the horizon."""
    #     # the horizon
    #     para_p_horizon = np.copy(self.para_p)
    #     para_p_horizon[para_p_horizon == self.inf_milp] = 0

    #     para_h_horizon = np.copy(self.para_h)
    #     para_h_horizon[para_h_horizon == self.inf_milp] = 0

    #     para_lmax_horizon = np.copy(self.para_lmax)
    #     para_lmax_horizon[para_lmax_horizon == self.inf_milp] = 0
    #     horizon = (
    #         np.max(para_p_horizon, axis=1)
    #         + np.max(para_h_horizon, axis=1)
    #         + np.max(para_lmax_horizon, axis=1)
    #     )

    #     # print("")
    #     # # print(f"para_p_horizon = {para_p_horizon}")
    #     # # print(f"para_h_horizon = {para_h_horizon}")
    #     # print(f"para_lmax_horizon = {para_lmax_horizon}")
    #     # print(f"para_lmax_horizon max is {np.max(para_lmax_horizon)}")

    #     horizon = np.sum(horizon) + 1
    #     # print(f"\nhorizon={horizon}")

    #     return horizon
