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


class FJS1(_FJS):
    """
    following the formulation in:
    Kasapidis 2021, Flexible Job Shop Scheduling Problems with Arbitrary Precedence Graphs, sec 2.2
    """

    @property
    def model(self) -> Model | gp.Model | None:
        return self._model

    def get_params(self):
        i_dict = {i: oid for i, oid in enumerate(self.operations)}
        i_dict_rev = {oid: i for i, oid in enumerate(self.operations)}
        k_dict = {k: mid for k, mid in enumerate(self.machines)}

        p_i_k = defaultdict(dict)
        for i, operation_id in i_dict.items():
            for k, machine_id in k_dict.items():
                te = self.time_estimates[operation_id][machine_id]
                if not math.isinf(te):
                    p_i_k[i][k] = te
        p_i_k: dict[
            int, dict[int, float]
        ]  # processing time of operation i on machine k

        pj = dict()
        for i, operation_id in i_dict.items():
            predecessors = self.precedence[operation_id]
            pj[i] = [i_dict_rev[predecessor] for predecessor in predecessors]
        pj: dict[int, list[int]]

        # estimate curl M
        # "a valid upper bound for M can be calculated as ∑ i∈Ω maxk∈Mi pi,k."
        big_m = 0
        for i in i_dict:
            big_m += max(p_i_k[i].values())

        return i_dict, i_dict_rev, k_dict, p_i_k, pj, big_m

    def build_model_cplex(self):
        from docplex.mp.model import Model
        from docplex.mp.dvar import Var

        i_dict, i_dict_rev, k_dict, p_i_k, pj, big_m = self.get_params()

        model = Model(name="FJS1")
        var_c_max = model.continuous_var(name="C_max", lb=0, ub=model.infinity)

        vars_x_i_j_k = model.binary_var_cube(
            len(i_dict), len(i_dict), len(k_dict), name="X"
        )  # eq 15
        vars_x_i_j_k: dict[tuple[int, int, int], Var]

        vars_y_i_k = model.binary_var_matrix(
            len(i_dict), len(k_dict), name="Y"
        )  # eq 16
        vars_y_i_k: dict[tuple[int, int], Var]

        vars_c_i = model.continuous_var_dict(
            len(i_dict), lb=0, ub=model.infinity, name="C"
        )
        vars_c_i: dict[int, Var]

        # eq 9 and 10
        for i, operation_id in i_dict.items():
            available_ks = sorted(p_i_k[i].keys())
            model.addConstr(
                sum([vars_y_i_k[(i, k)] for k in available_ks]) == 1,
                ctname=f"eq 9: i={i}",
            )
            for j in pj[i]:
                model.addConstr(
                    vars_c_i[i]
                    >= vars_c_i[j]
                    + sum([vars_y_i_k[(i, k)] * p_i_k[i][k] for k in available_ks]),
                    ctname=f"eq 10: i={i} j={j}",
                )

        # eq 11 and eq 12
        for i, operation_id_i in i_dict.items():
            for j, operation_id_j in i_dict.items():
                if i == j:
                    continue
                k_list = sorted(set(p_i_k[i].keys()).intersection(set(p_i_k[j].keys())))
                for k in k_list:
                    model.addConstr(
                        vars_c_i[i]
                        >= vars_c_i[j]
                        + p_i_k[i][k]
                        - big_m
                        * (
                            2
                            + vars_x_i_j_k[(i, j, k)]
                            - vars_y_i_k[(i, k)]
                            - vars_y_i_k[(j, k)]
                        ),
                        ctname=f"eq 11: i={i} j={j} k={k}",
                    )

                    model.addConstr(
                        vars_c_i[j]
                        >= vars_c_i[i]
                        + p_i_k[j][k]
                        - big_m
                        * (
                            3
                            - vars_x_i_j_k[(i, j, k)]
                            - vars_y_i_k[(i, k)]
                            - vars_y_i_k[(j, k)]
                        ),
                        ctname=f"eq 12: i={i} j={j} k={k}",
                    )

        # eq 14
        model.addConstrs([var_c_max >= C_i for C_i in vars_c_i], names="eq 14")
        self.model_string = model.export_to_string()
        self._model = model
        return model

    def build_model_gurobi(self):
        env = gp.Env(
            empty=True
        )  # TODO this arrangement doesn't really suppress stdout...
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()

        i_dict, i_dict_rev, k_dict, p_i_k, pj, big_m = self.get_params()

        model = gp.Model(self.__class__.__name__)

        var_c_max = model.addVar(
            name="C_max", lb=1e-5, ub=float("inf"), vtype=GRB.CONTINUOUS
        )

        vars_x_i_j_k = model.addVars(
            len(i_dict), len(i_dict), len(k_dict), name="X", vtype=GRB.BINARY
        )  # eq 15
        vars_x_i_j_k: gp.tupledict[tuple[int, int, int], gp.Var]

        vars_y_i_k = model.addVars(
            len(i_dict), len(k_dict), name="Y", vtype=GRB.BINARY
        )  # eq 16
        vars_y_i_k: gp.tupledict[tuple[int, int], gp.Var]

        vars_c_i = model.addVars(
            len(i_dict), name="C", lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS
        )
        vars_c_i: gp.tupledict[int, gp.Var]

        # eq 9 and 10
        for i, operation_id in i_dict.items():
            available_ks = sorted(p_i_k[i].keys())
            model.addConstr(
                sum([vars_y_i_k[(i, k)] for k in available_ks]) == 1,
                name=f"eq 9: i={i}",
            )
            for j in pj[i]:
                model.addConstr(
                    vars_c_i[i]
                    >= vars_c_i[j]
                    + sum([vars_y_i_k[(i, k)] * p_i_k[i][k] for k in available_ks]),
                    name=f"eq 10: i={i} j={j}",
                )

        # eq 11 and eq 12
        for i, operation_id_i in i_dict.items():
            for j, operation_id_j in i_dict.items():
                if i == j:
                    continue
                k_list = sorted(set(p_i_k[i].keys()).intersection(set(p_i_k[j].keys())))
                for k in k_list:
                    model.addConstr(
                        vars_c_i[i]
                        >= vars_c_i[j]
                        + p_i_k[i][k]
                        - big_m
                        * (
                            2
                            + vars_x_i_j_k[(i, j, k)]
                            - vars_y_i_k[(i, k)]
                            - vars_y_i_k[(j, k)]
                        ),
                        name=f"eq 11: i={i} j={j} k={k}",
                    )

                    model.addConstr(
                        vars_c_i[j]
                        >= vars_c_i[i]
                        + p_i_k[j][k]
                        - big_m
                        * (
                            3
                            - vars_x_i_j_k[(i, j, k)]
                            - vars_y_i_k[(i, k)]
                            - vars_y_i_k[(j, k)]
                        ),
                        name=f"eq 12: i={i} j={j} k={k}",
                    )

        # eq 14
        for i, c_i in vars_c_i.items():
            model.addConstr(var_c_max >= c_i, name=f"eq 14: i={i}")

        model.setObjective(var_c_max, GRB.MINIMIZE)
        self.model_string = model.__str__()
        self._model = model
        env.close()

    def solve_cplex(self):
        assert self.model is not None
        self.model.solve()
        # TODO export to FjsOutput

    def solve_gurobi(self):
        assert self.model is not None
        self.model.optimize()

        i_dict, i_dict_rev, k_dict, p_i_k, pj, big_m = self.get_params()

        k_dict_rev = {v: k for k, v in k_dict.items()}

        start_times = dict()
        assignments = dict()

        makespan = None

        self.model.solution

        for v in self.model.getVars():
            if v.VarName.startswith("C") and v.VarName.endswith("]"):
                i = get_gurobi_var_index(v.VarName, "C")
                operation_id = i_dict[i]
                start_times[operation_id] = v.X
            elif v.VarName.startswith("Y") and v.VarName.endswith("]"):
                i, k = get_gurobi_var_index(v.VarName, "Y")
                operation_id = i_dict[i]
                machine_id = k_dict[k]
                if v.X == 1:
                    assignments[operation_id] = machine_id
            elif v.VarName == "C_max":
                makespan = v.X
        end_times = dict()
        for operation_id, st in start_times.items():
            machine_id = assignments[operation_id]
            et = st + p_i_k[i_dict_rev[operation_id]][k_dict_rev[machine_id]]
            end_times[operation_id] = et
        assert makespan == self.model.ObjVal
        solved_operations = []
        for operation_id in start_times:
            solved_operation = SolvedOperation(
                id=operation_id,
                assigned_to=assignments[operation_id],
                start_time=start_times[operation_id],
                end_time=end_times[operation_id],
            )
            solved_operations.append(solved_operation)
        return FjsOutput(
            solved_operations=solved_operations,
            makespan=makespan,
        )


class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v}={self.Value(v)}", end=" ")
        print()
        if self.__solution_count >= self.__solution_limit:
            print(f"Stop search after {self.__solution_limit} solutions")
            self.StopSearch()

    def solution_count(self):
        return self.__solution_count


class FJSS2(_FJS):
    """
    Implementation of the constraint programming formulation in:
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
        inf_cp: int = 1.0e10,
        num_workers: int = 16,
        verbose: bool = True,
    ):
        """
        _summary_

        Parameters
        ----------
        operations : list[str]
            _description_
        machines : list[str]
            _description_
        precedence : dict[str, list[str]]
            _description_
        time_estimates : np.ndarray
            _description_
        para_a : np.ndarray
            Setup time of machine m when processing operation i before j and para_a =
            np.full((n_mach, n_opt, n_opt), dtype=object, fill_value=infinity).
        para_w : np.ndarray
            Weight of operation i in machine m and para_w = np.empty((n_opt, n_mach), dtype=object).
        para_mach_capacity : list[int]|np.ndarray
            A list of machine capacity or a numpy array of machine capacity.
        para_lmin : np.ndarray
            Minimum lag between the starting time of operation i and the ending time of operation j.
            Shape=(n_opt, n_opt).
        para_lmax : np.ndarray
            Maximum lag between the starting time of operation i and the ending time of operation j.
            Shape=(n_opt, n_opt).
        job_data : list[list[tuple[int, int]]]
            The list of jobs. Each job is a list of tuples (machine_id, processing_time).
        co_exist_set : list[str], optional
            The list of operation pairs that can overlap, by default None. For example, co_exist_set
            = ["0,1", "3,4"], that means operation 0 can overlap with operation 1 and operation 3
            can overlap with operation 4.
        allowed_overlapping_machine : list[int], optional
            The machine ids for the machines that allow overlapping. Default=None. For example,
            allowed_overlapping_machine = [0, 1, 5] means machine 0, 1, 5 allow overlapping.
        model_string : str | None, optional
            _description_, by default None
        inf_cp : int, optional
            _description_, by default 1.0e10
        verbose : bool, optional
            If print out the solution explicitly, by default True.

        Returns
        -------
        FjsOutput
            The output of the FJS solver.

        Notes
        -----
        Please note that for the constraint programming solver, all the inputs have to be
        integers. If you have float numbers in the input, please multiply a large number to make
        it a large integer. For example, if you have a float number 0.5, you can multiply 1e6 to
        feed into this model. The model will get you solution with 6 decimal points, which is
        from 1e6.

        """
        self.inf_cp = inf_cp
        self.num_workers = num_workers

        para_p[para_p == np.inf] = inf_cp
        para_p[para_p == -np.inf] = -inf_cp
        self.para_p = para_p.astype(int)

        super().__init__(
            operations=operations,
            machines=machines,
            precedence=precedence,
            time_estimates=None,
            model_string=model_string,
        )

        para_a[para_a == np.inf] = inf_cp
        para_a[para_a == -np.inf] = -inf_cp
        para_w[para_w == np.inf] = inf_cp
        para_w[para_w == -np.inf] = -inf_cp
        para_delta[para_delta == np.inf] = inf_cp
        para_delta[para_delta == -np.inf] = -inf_cp
        para_lmin[para_lmin == np.inf] = inf_cp
        para_lmin[para_lmin == -np.inf] = -inf_cp
        para_lmax[para_lmax == np.inf] = inf_cp
        para_lmax[para_lmax == -np.inf] = -inf_cp
        para_h[para_h == np.inf] = inf_cp
        para_h[para_h == -np.inf] = -inf_cp

        self.para_a = para_a.astype(int)
        self.para_w = para_w.astype(int)
        self.para_mach_capacity = para_mach_capacity.astype(int)
        self.para_delta = para_delta.astype(int)
        self.para_lmin = para_lmin.astype(int)
        self.para_lmax = para_lmax.astype(int)
        self.para_h = para_h.astype(int)

        self.horizon = self.get_horizon()

        self._model = None
        self._solver = None
        self.var_s = None
        self.var_c = None
        self.var_y = None
        self.verbose = verbose
        self.var_c_max = inf_cp
        self.var_u = None
        self.yu_list = None
        self.num_t = None

    def get_horizon(self):
        """Get the horizon."""
        # the horizon
        para_p_horizon = np.copy(self.para_p)
        para_p_horizon[para_p_horizon == self.inf_cp] = 0

        para_h_horizon = np.copy(self.para_h)
        para_h_horizon[para_h_horizon == self.inf_cp] = 0

        para_lmax_horizon = np.copy(self.para_lmax)
        para_lmax_horizon[para_lmax_horizon == self.inf_cp] = 0
        horizon = (
            np.sum(para_p_horizon, axis=1)
            + np.sum(para_h_horizon, axis=1)
            + np.sum(para_lmax_horizon, axis=1)
        )
        horizon = int(np.sum(horizon)) + 1

        return horizon

    def build_model_ortools(self):
        """Build the model."""
        n_opt, n_mach = self.get_params()

        model = cp_model.CpModel()

        horizon = self.horizon
        # make span
        var_c_max = model.NewIntVar(0, horizon, "C_max")

        # define the variables
        # if operation i is processed by machine m
        var_y = np.empty((n_opt, n_mach), dtype=object)
        for i, m in product(range(n_opt), range(n_mach)):
            var_y[i, m] = model.NewBoolVar(f"y_{i}_{m}")

        # starting time of operation i
        var_s = np.empty((n_opt), dtype=object)
        for i in range(n_opt):
            var_s[i] = model.NewIntVar(0, horizon, f"s_{i}")

        # completion time of operation i
        var_c = np.empty((n_opt), dtype=object)
        for i in range(n_opt):
            var_c[i] = model.NewIntVar(0, horizon, f"c_{i}")

        # add constraints
        for i in range(n_opt):
            # eq. (2)
            model.Add(var_c_max >= var_c[i])

            # eq. (3)
            expr = [self.para_p[i, m] * var_y[i, m] for m in range(n_mach)]
            model.Add(var_c[i] >= var_s[i] + sum(expr))

            # eq. (4)
            expr = [
                (self.para_p[i, m] + self.para_h[i, m]) * var_y[i, m]
                for m in range(n_mach)
            ]
            model.Add(var_c[i] <= var_s[i] + sum(expr))

        # eq. (5)
        for i in range(n_opt):
            # sum of y_im = 1
            model.Add(sum([var_y[i, m] for m in range(n_mach)]) == 1)

        for i, j in product(range(n_opt), range(n_opt)):
            if i != j:
                # eq. (6)
                # minimum lag between the starting time of operation i and the ending time of operation j
                model.Add(var_s[j] >= var_c[i] + self.para_lmin[i, j])
                # eq. (7)
                # maximum lag between the starting time of operation i and the ending time of operation j
                model.Add(var_s[j] <= var_c[i] + self.para_lmax[i, j])

        # https://developers.google.com/optimization/cp/channeling
        for i, j, m in product(np.arange(n_opt), np.arange(n_opt), np.arange(n_mach)):
            if i != j:
                # https://github.com/d-krupke/cpsat-primer

                # eq. (22)
                # left part of the implication
                bool_b1 = model.NewBoolVar(f"bool_b1_{i}_{j}_{m}")
                model.Add(var_y[i, m] + var_y[j, m] == 2).OnlyEnforceIf(bool_b1)
                model.Add(var_y[i, m] + var_y[j, m] != 2).OnlyEnforceIf(bool_b1.Not())

                # right part of the implication
                bool_b2 = model.NewBoolVar(f"bool_b2_{i}_{j}_{m}")
                bool_a1 = model.NewBoolVar(f"bool_a1_{i}_{j}_{m}")
                bool_a2 = model.NewBoolVar(f"bool_a2_{i}_{j}_{m}")
                model.Add(var_s[j] >= var_c[i] + self.para_a[m, i, j]).OnlyEnforceIf(
                    bool_a1
                )
                model.Add(var_s[j] < var_c[i] + self.para_a[m, i, j]).OnlyEnforceIf(
                    bool_a1.Not()
                )
                model.Add(var_s[i] >= var_c[j] + self.para_a[m, j, i]).OnlyEnforceIf(
                    bool_a2
                )
                model.Add(var_s[i] < var_c[j] + self.para_a[m, j, i]).OnlyEnforceIf(
                    bool_a2.Not()
                )

                model.Add(bool_a1 + bool_a2 >= 1).OnlyEnforceIf(bool_b2)
                model.Add(bool_a1 + bool_a2 < 1).OnlyEnforceIf(bool_b2.Not())

                # the implication
                model.AddImplication(bool_b1, bool_b2)

                # eq. (23)
                # left part of the implication is the same as eq. (22), so we can reuse bool_b1
                # right part of the implication
                bool_b3 = model.NewBoolVar(f"bool_b3_{i}_{j}_{m}")
                max_eq23_first = model.NewIntVar(
                    0, horizon, f"max_eq23_{i}_{j}_{m}_first"
                )
                model.AddMaxEquality(
                    max_eq23_first, [0, var_c[i] - var_s[i] - var_c[j] + var_s[j]]
                )

                max_eq23_second = model.NewIntVar(
                    0, horizon, f"max_eq23_{i}_{j}_{m}_second"
                )
                model.AddMaxEquality(
                    max_eq23_second, [0, var_c[j] - var_s[j] - var_c[i] + var_s[i]]
                )

                # define the intermediate variable for var_s[j] >= var_s[i] + max_eq23_first + para_delta       [m]
                bool_a3 = model.NewBoolVar(f"bool_a3_{i}_{j}_{m}")
                model.Add(
                    var_s[j] >= var_s[i] + max_eq23_first + self.para_delta[m]
                ).OnlyEnforceIf(bool_a3)
                model.Add(
                    var_s[j] < var_s[i] + max_eq23_first + self.para_delta[m]
                ).OnlyEnforceIf(bool_a3.Not())
                # define the intermediate variable for var_s[i] >= var_s[j] + max_eq23_second + para_delta      [m]
                bool_a4 = model.NewBoolVar(f"bool_a4_{i}_{j}_{m}")
                model.Add(
                    var_s[i] >= var_s[j] + max_eq23_second + self.para_delta[m]
                ).OnlyEnforceIf(bool_a4)
                model.Add(
                    var_s[i] < var_s[j] + max_eq23_second + self.para_delta[m]
                ).OnlyEnforceIf(bool_a4.Not())

                model.AddBoolOr(bool_a3, bool_a4).OnlyEnforceIf(bool_b3)
                model.AddBoolAnd(bool_a3.Not(), bool_a4.Not()).OnlyEnforceIf(
                    bool_b3.Not()
                )

                # the implication
                model.AddImplication(bool_b1, bool_b3)

        sum_time = horizon
        num_t = int(sum_time / 1.0e0)
        var_u = np.empty((n_opt, n_mach, num_t), dtype=object)
        for i, m, t in product(range(n_opt), range(n_mach), range(num_t)):
            var_u[i, m, t] = model.NewIntVar(0, np.max(self.para_w), f"u_{i}_{m}_{t}")

        # eq. (25)
        yu_list = []
        for idx_m, m in enumerate(np.arange(n_mach)):
            for idx_t, t in enumerate(np.arange(num_t)):
                constr_25 = 0
                for idx_i, i in enumerate(np.arange(n_opt)):
                    yu = model.NewIntVar(
                        0, np.max(self.para_w), f"yu_{idx_i}_{idx_m}_{idx_t}"
                    )
                    model.AddMultiplicationEquality(
                        yu, [var_y[idx_i, idx_m], var_u[idx_i, idx_m, idx_t]]
                    )
                    constr_25 += yu
                    yu_list.append(yu)
                model.Add(constr_25 <= self.para_mach_capacity[idx_m])

        # eq. (24)
        for idx_m, m in enumerate(np.arange(n_mach)):
            for idx_i, i in enumerate(np.arange(n_opt)):
                for idx_t, t in enumerate(np.arange(num_t)):
                    bool_list = []

                    bool_x7 = model.NewBoolVar(f"bool_x7_{i}_{idx_t}")
                    model.Add(var_s[i] <= t).OnlyEnforceIf(bool_x7)
                    model.Add(var_s[i] > t).OnlyEnforceIf(bool_x7.Not())
                    bool_list.append(bool_x7)

                    bool_x8 = model.NewBoolVar(f"bool_x8_{i}_{idx_t}")
                    model.Add(var_c[i] >= t).OnlyEnforceIf(bool_x8)
                    model.Add(var_c[i] < t).OnlyEnforceIf(bool_x8.Not())
                    bool_list.append(bool_x8)

                    bool_x9_and = model.NewBoolVar(f"bool_x9_{i}_{idx_t}")
                    model.Add(bool_x7 + bool_x8 == 2).OnlyEnforceIf(bool_x9_and)
                    model.Add(bool_x7 + bool_x8 < 2).OnlyEnforceIf(bool_x9_and.Not())

                    # when bool_x7 and bool_x8 are both true, var_u is is w_im
                    model.Add(
                        var_u[idx_i, idx_m, idx_t] == self.para_w[idx_i, idx_m]
                    ).OnlyEnforceIf(bool_x9_and)

                    model.Add(var_u[idx_i, idx_m, idx_t] == 0).OnlyEnforceIf(
                        bool_x9_and.Not()
                    )

        model.Minimize(var_c_max)
        self._model = model
        self.var_c_max = var_c_max
        self.var_s = var_s
        self.var_c = var_c
        self.var_y = var_y
        self.var_u = var_u
        self.yu_list = yu_list
        self.num_t = num_t

        return model

    @property
    def model(self) -> Model | gp.Model | None:
        return self._model

    def get_params(self):
        """Get parameters for the model."""
        n_opt = len(self.operations)
        n_mach = len(self.machines)

        return n_opt, n_mach

    def solve_ortools(self):
        # creates the solver and solve.

        if self._model is None:
            self.build_model_ortools()

        solver = cp_model.CpSolver()

        self._solver = solver

        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.log_search_progress = self.verbose

        # TODO: add call back function to pint out the solution
        status = solver.Solve(self._model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            self.var_c_max = solver.ObjectiveValue()
            # print the solution found.
            print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
            # Statistics.
            # print("\nStatistics")
            # print(f"  - conflicts: {solver.NumConflicts()}")
            # print(f"  - branches : {solver.NumBranches()}")
            # print(f"  - wall time: {solver.WallTime()}s")

            assignments = dict()
            start_times = dict()
            end_times = dict()
            solved_operations = []

            for i, m in product(range(len(self.operations)), range(len(self.machines))):
                if solver.Value(self.var_y[i, m]) == 1:
                    assignments[self.operations[i]] = self.machines[m]
                    start_times[self.operations[i]] = solver.Value(self.var_s[i])
                    end_times[self.operations[i]] = solver.Value(self.var_c[i])
                    solved_operation = SolvedOperation(
                        id=self.operations[i],
                        assigned_to=self.machines[m],
                        start_time=solver.Value(self.var_s[i]),
                        end_time=solver.Value(self.var_c[i]),
                    )
                    solved_operations.append(solved_operation)

            return FjsOutput(
                solved_operations=solved_operations,
                makespan=solver.ObjectiveValue(),
            )

        else:
            print("No solution found.")
            return None


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
        num_workers: int = None,
        inf_milp: float = 1.0e7,
        shift_durations: float|int = None,
        big_m: float | int = None,
        # big_m=1.0e6,
        matrix_variables=True,
        verbose: bool = True,
    ):
        self.num_workers = num_workers
        self.matrix_variables = matrix_variables
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
            # print(f"para_lmin={para_lmin}")
            # print(f"non-negative elements of para_lmin: {para_lmin[para_lmin>=0]}")
            self.big_m = get_m_value_runzhong(
                para_p=para_p,
                para_h=para_h,
                para_lmin=para_lmin,
                para_a=para_a,
                infinity=inf_milp,
            )
            print(f"the inferred big_m value with Runzhong version is {self.big_m}")

        # # this part works because the leak of infinity to big_m
        # if big_m is None:
        #     self.big_m = get_m_value_old(
        #         para_p=para_p, para_h=para_h, para_lmin=para_lmin, para_a=para_a
        #     )
        else:
            self.big_m = big_m

        # self.horizon = self.get_horizon()
        # self.big_m = self.horizon

        # print(f"big_m: {self.big_m}")

        self.shift_durations = shift_durations
        self.verbose = verbose
        self.var_c_max = None
        self.var_y = None
        self.var_c = None
        self.var_s = None
        self.var_x = None
        self.var_z = None
        self.model = None
        self.var_ws = None

    def build_model_gurobi(self):
        """Build the mixed integer linear programming model with gurobi."""

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()

        model = gp.Model("fjss_mip")
        n_opt, n_mach = self.get_params()

        # create variables
        if self.matrix_variables:
            # if operation i is processed by machine m
            var_y = model.addMVar((n_opt, n_mach), vtype=GRB.BINARY, name="var_y")
            # if operation i is processed before operation j
            var_x = model.addMVar((n_opt, n_opt), vtype=GRB.BINARY, name="var_x")
            # if operation i is processed before and overlapped operation j in machine m
            var_z = model.addMVar(
                (n_opt, n_opt, n_mach), vtype=GRB.BINARY, name="var_z"
            )
            # starting time of operation i
            var_s = model.addMVar(n_opt, vtype=GRB.CONTINUOUS, name="var_s")
            # completion time of operation i
            var_c = model.addMVar(n_opt, vtype=GRB.CONTINUOUS, name="var_c")
        else:
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

        # work shifts, formulation by Qianxiang
        # we can add constraints to make sure s_i and c_i falls in one shift:
        # say ws_k represents the start time of kth (let a shift be 8hrs)
        # and the first shift starts at t=0, add a bool var ws_ki for shift assignment (with constraint \Sum_k ws_ki = 1)
        # and ws_ki=1 -> (ws_k < s_i) ^ (ws_k + 8hrs > c_i)
        if self.shift_durations:
            var_ws = model.addVar(n_opt, vtype=GRB.BINARY, name="var_ws")
            # add constraints
            model.addConstr(
                    gp.quicksum(var_ws[i] for i in range(n_opt)) == 1, name="work_shifts_constr_1"
                )
            # add indicator constraints
            for i in range(n_opt):
                # https://support.gurobi.com/hc/en-us/articles/4414392016529-How-do-I-model-conditional-statements-in-Gurobi
                # https://www.gurobi.com/documentation/current/refman/py_model_agc_indicator.html
                model.addConstr((var_ws[i] == 1) >> gp.quicksum(
                    [var_ws[i] <= var_s[i], var_ws[i] + self.shift_durations >= var_c[i]]) == 2,
                                name=f"work_shifts_constr_2_{i}")
            self.var_ws = var_ws

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

        # add the threads
        if self.num_workers is None:
            # https://www.gurobi.com/documentation/current/refman/threads.html
            # The default value of 0 is an automatic setting.
            # It will generally use as many threads as there are virtual processors
            # We've made the pragmatic choice to impose a soft limit of 32 threads for the automatic setting (0)
            self.model.Params.Threads = 0
        # this is not recommended
        if self.num_workers is not None:
            self.model.Params.Threads = self.num_workers

        # set the number of solutions to be found
        # self.model.Params.PoolSolutions = 20

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
