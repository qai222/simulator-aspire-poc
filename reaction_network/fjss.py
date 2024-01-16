from __future__ import annotations

import math
import os
import random
from abc import ABC
from collections import defaultdict

import gurobipy as gp
from docplex.mp.model import Model
from gurobipy import GRB
from monty.json import MSONable
from pydantic import BaseModel

from reaction_network.schema.lv2 import BenchTopLv2, OperationType

os.environ["GRB_LICENSE_FILE"] = "/home/qai/local/gurobi_lic/gurobi.lic"


def get_gurobi_var_index(var_name: str, var_header: str, ) -> int | list[int]:
    assert var_name.startswith(var_header)
    assert var_name.endswith("]")
    stripped_index_string = var_name[len(var_header):].lstrip("[").rstrip("]")
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
    if operation_type in [OperationType.OPERATION_LOADING, OperationType.OPERATION_UNLOADING,
                          OperationType.OPERATION_RELOADING]:
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
    def __init__(self, operations: list[str], machines: list[str], precedence: dict[str, list[str]],
                 time_estimates: dict[str, dict[str, float]], model_string: str | None = None, ):
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
    def from_bench_top(cls, bench_top: BenchTopLv2, time_estimates: dict[str, dict[str, float]] = None):
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
                can_be_processed_by = bench_top.operation_type_device_mapping[operation.type]
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
            time_estimates=time_est
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
        p_i_k: dict[int, dict[int, float]]  # processing time of operation i on machine k

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

        vars_x_i_j_k = model.binary_var_cube(len(i_dict), len(i_dict), len(k_dict), name="X")  # eq 15
        vars_x_i_j_k: dict[tuple[int, int, int], Var]

        vars_y_i_k = model.binary_var_matrix(len(i_dict), len(k_dict), name="Y")  # eq 16
        vars_y_i_k: dict[tuple[int, int], Var]

        vars_c_i = model.continuous_var_dict(len(i_dict), lb=0, ub=model.infinity, name="C")
        vars_c_i: dict[int, Var]

        # eq 9 and 10
        for i, operation_id in i_dict.items():
            available_ks = sorted(p_i_k[i].keys())
            model.add_constraint(sum([vars_y_i_k[(i, k)] for k in available_ks]) == 1, ctname=f"eq 9: i={i}")
            for j in pj[i]:
                model.add_constraint(
                    vars_c_i[i] >= vars_c_i[j] + sum([vars_y_i_k[(i, k)] * p_i_k[i][k] for k in available_ks]),
                    ctname=f"eq 10: i={i} j={j}")

        # eq 11 and eq 12
        for i, operation_id_i in i_dict.items():
            for j, operation_id_j in i_dict.items():
                if i == j:
                    continue
                k_list = sorted(set(p_i_k[i].keys()).intersection(set(p_i_k[j].keys())))
                for k in k_list:
                    model.add_constraint(
                        vars_c_i[i] >= vars_c_i[j] + p_i_k[i][k] - big_m * (
                                2 + vars_x_i_j_k[(i, j, k)] - vars_y_i_k[(i, k)] - vars_y_i_k[(j, k)]),
                        ctname=f"eq 11: i={i} j={j} k={k}"
                    )

                    model.add_constraint(
                        vars_c_i[j] >= vars_c_i[i] + p_i_k[j][k] - big_m * (
                                3 - vars_x_i_j_k[(i, j, k)] - vars_y_i_k[(i, k)] - vars_y_i_k[(j, k)]),
                        ctname=f"eq 12: i={i} j={j} k={k}"
                    )

        # eq 14
        model.add_constraints([var_c_max >= C_i for C_i in vars_c_i], names="eq 14")
        self.model_string = model.export_to_string()
        self._model = model
        return model

    def build_model_gurobi(self):

        env = gp.Env(empty=True)  # TODO this arrangement doesn't really suppress stdout...
        env.setParam('OutputFlag', 0)
        env.setParam('LogToConsole', 0)
        env.start()

        i_dict, i_dict_rev, k_dict, p_i_k, pj, big_m = self.get_params()

        model = gp.Model(self.__class__.__name__)

        var_c_max = model.addVar(name="C_max", lb=1e-5, ub=float('inf'), vtype=GRB.CONTINUOUS)

        vars_x_i_j_k = model.addVars(len(i_dict), len(i_dict), len(k_dict), name="X", vtype=GRB.BINARY)  # eq 15
        vars_x_i_j_k: gp.tupledict[tuple[int, int, int], gp.Var]

        vars_y_i_k = model.addVars(len(i_dict), len(k_dict), name="Y", vtype=GRB.BINARY)  # eq 16
        vars_y_i_k: gp.tupledict[tuple[int, int], gp.Var]

        vars_c_i = model.addVars(len(i_dict), name="C", lb=0, ub=float('inf'), vtype=GRB.CONTINUOUS)
        vars_c_i: gp.tupledict[int, gp.Var]

        # eq 9 and 10
        for i, operation_id in i_dict.items():
            available_ks = sorted(p_i_k[i].keys())
            model.addConstr(sum([vars_y_i_k[(i, k)] for k in available_ks]) == 1, name=f"eq 9: i={i}")
            for j in pj[i]:
                model.addConstr(
                    vars_c_i[i] >= vars_c_i[j] + sum([vars_y_i_k[(i, k)] * p_i_k[i][k] for k in available_ks]),
                    name=f"eq 10: i={i} j={j}")

        # eq 11 and eq 12
        for i, operation_id_i in i_dict.items():
            for j, operation_id_j in i_dict.items():
                if i == j:
                    continue
                k_list = sorted(set(p_i_k[i].keys()).intersection(set(p_i_k[j].keys())))
                for k in k_list:
                    model.addConstr(
                        vars_c_i[i] >= vars_c_i[j] + p_i_k[i][k] - big_m * (
                                2 + vars_x_i_j_k[(i, j, k)] - vars_y_i_k[(i, k)] - vars_y_i_k[(j, k)]),
                        name=f"eq 11: i={i} j={j} k={k}"
                    )

                    model.addConstr(
                        vars_c_i[j] >= vars_c_i[i] + p_i_k[j][k] - big_m * (
                                3 - vars_x_i_j_k[(i, j, k)] - vars_y_i_k[(i, k)] - vars_y_i_k[(j, k)]),
                        name=f"eq 12: i={i} j={j} k={k}"
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
