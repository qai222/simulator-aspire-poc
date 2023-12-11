from __future__ import annotations

import collections
import math
import os
import random
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

# from reaction_network.schema.lv2 import BenchTopLv2, OperationType

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
                 time_estimates: dict[str, dict[str, float]]|None, model_string: str | None =
                 None, ):
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
        para_mach_capacity: list[int] | np.ndarray,
        para_lmin: np.ndarray,
        para_lmax: np.ndarray,
        jobs_data: list[list[tuple[int, int]]],
        precedence: dict[str, list[str]] | None,
        co_exist_set: list[str] = None,
        allowed_overlapping_machine: list[int] = None,
        model_string: str | None = None,
        inf_cp: int = 1.0e10,
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
            np.full((n_opt, n_opt, n_mach), dtype=object, fill_value=infinity).
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

        """
        self.inf_cp = inf_cp

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
        self.para_a = para_a.astype(int)

        para_w[para_w == np.inf] = inf_cp
        para_w[para_w == -np.inf] = -inf_cp
        self.para_w = para_w.astype(int)
        self.para_mach_capacity = para_mach_capacity.astype(int)

        para_lmin[para_lmin == np.inf] = inf_cp
        para_lmin[para_lmin == -np.inf] = -inf_cp
        self.para_lmin = para_lmin.astype(int)

        para_lmax[para_lmax == np.inf] = inf_cp
        para_lmax[para_lmax == -np.inf] = -inf_cp
        self.para_lmax = para_lmax.astype(int)

        self.jobs_data = jobs_data
        self.co_exist_set = co_exist_set
        self.allowed_overlapping_machine = allowed_overlapping_machine
        self._model = None
        self.verbose = verbose

    def build_model_ortools(self):
        """Build the model."""
        n_opt, n_mach = self.get_params()
        # # operation 0 can overlap with operation 1
        # # operation 3 can overlap with operation 4
        # co_exist_set = ["0,1", "3,4"]
        # # machine ids for the machines that allow overlapping
        # allowed_overlapping_machine = [0, 1, 2]

        model = cp_model.CpModel()
        # horizon = sum(task[1] for job in jobs_data for task in job)
        horizon = (
                np.sum(self.para_p[self.para_p != self.inf_cp]).sum()
                + np.sum(self.para_lmax[self.para_lmax != self.inf_cp]).sum()
                + 1
        )
        horizon = int(horizon)

        # if operation i is processed by machine m
        var_y = np.empty((n_opt, n_mach), dtype=object)
        for i, m in product(range(n_opt), range(n_mach)):
            var_y[i, m] = model.NewBoolVar(f"y_{i}_{m}")

        # if operation i is processed before operation j
        var_x = np.empty((n_opt, n_opt), dtype=object)
        for i, j in product(range(n_opt), range(n_opt)):
            var_x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

        # if operation i is processed before operation j on machine m
        var_z = np.empty((n_opt, n_opt, n_mach), dtype=object)
        for i, j, m in product(range(n_opt), range(n_opt), range(n_mach)):
            var_z[i, j, m] = model.NewBoolVar(f"z_{i}_{j}_{m}")

        # starting time of operation i
        var_s = np.empty(n_opt, dtype=object)
        for i in range(n_opt):
            var_s[i] = model.NewIntVar(0, horizon, f"s_{i}")

        # completion time of operation i
        var_c = np.empty(n_opt, dtype=object)
        for i in range(n_opt):
            var_c[i] = model.NewIntVar(0, horizon, f"c_{i}")

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple("task_type", "start end interval")
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple(
            "assigned_task_type", "start job index duration"
        )

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(self.jobs_data):
            for task_id, task in enumerate(job):
                machine, duration = task
                suffix = f"_{job_id}_{task_id}"
                start_var = model.NewIntVar(0, horizon, "start" + suffix)
                end_var = model.NewIntVar(0, horizon, "end" + suffix)
                interval_var = model.NewIntervalVar(
                    start_var, duration, end_var, "interval" + suffix
                )
                all_tasks[job_id, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var
                )
                machine_to_intervals[machine].append(interval_var)

        # ==============================================================================
        # # Create and add disjunctive constraints.
        # # no overlapping

        for machine_i, machine_j in combinations(self.allowed_overlapping_machine, 2):
            for interval_i, interval_j in product(
                    machine_to_intervals[machine_i], machine_to_intervals[machine_j]
            ):
                job_id_i, task_id_i = interval_i.Name().split("_")[-2:]
                job_id_j, task_id_j = interval_j.Name().split("_")[-2:]
                if job_id_i < job_id_j:
                    if f"{job_id_i},{job_id_j}" not in self.co_exist_set:
                        model.AddNoOverlap([interval_i, interval_j])

        # ==============================================================================

        # =============================================================================
        # Precedences inside a job.
        for job_id, job in enumerate(self.jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(
                    all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
                )

        for i, j in product(range(n_opt), range(n_opt)):
            if self.para_lmin[i, j] != -self.inf_cp and self.para_lmin[i, j] != self.inf_cp:
                # eq. (6)
                # minimum lag between the starting time of operation i and the ending time of operation j
                model.Add(var_s[j] >= var_c[i] + self.para_lmin[i, j])
            if self.para_lmax[i, j] != -self.inf_cp and self.para_lmax[i, j] != self.inf_cp:
                # eq. (7)
                # maximum lag between the starting time of operation i and the ending time of operation j
                model.Add(var_s[j] <= var_c[i] + self.para_lmax[i, j])

        # eq. (16)
        for i in range(n_opt):
            for j in range(n_opt):
                expr = []
                for m in range(n_mach):
                    if i != j:
                        # solver.Add(var_w[j, m] * var_z[i, j, m] >= var_w[i, m])
                        expr.append(self.para_w[j, m] * var_z[i, j, m])
                # TODO: Fix this as the index m is not right
                model.Add(
                    cp_model.LinearExpr.Sum(expr)
                    <= (self.para_mach_capacity[m] - self.para_w[i, m]) * var_y[i, m]
                )

        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, "makespan")
        # self.obj_var = obj_var
        model.AddMaxEquality(
            obj_var,
            [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(self.jobs_data)],
        )
        model.Minimize(obj_var)

        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Solution:")
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            solved_operations = []
            for job_id, job in enumerate(self.jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[str(machine)].append(
                        assigned_task_type(
                            start=solver.Value(all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            duration=task[1],
                        )
                    )
                    solved_operation = SolvedOperation(
                        id=f"job_{job_id}_task_{task_id}",
                        assigned_to=str(machine),
                        start_time=solver.Value(all_tasks[job_id, task_id].start),
                        end_time=solver.Value(all_tasks[job_id, task_id].end),
                    )
                    solved_operations.append(solved_operation)

            # Create per machine output lines.
            output = ""
            for machine in self.machines:
                # Sort by starting time.
                assigned_jobs[str(machine)].sort()
                sol_line_tasks = "Machine " + str(machine) + ": "
                sol_line = "           "

                for assigned_task in assigned_jobs[machine]:
                    name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                    # Add spaces to output to align columns.
                    sol_line_tasks += f"{name:15}"

                    start = assigned_task.start
                    duration = assigned_task.duration
                    sol_tmp = f"[{start},{start + duration}]"
                    # Add spaces to output to align columns.
                    sol_line += f"{sol_tmp:15}"

                sol_line += "\n"
                sol_line_tasks += "\n"
                output += sol_line_tasks
                output += sol_line

            # Finally print the solution found.
            print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
            print(output)
        else:
            print("No solution found.")

        # Statistics.
        print("\nStatistics")
        print(f"  - conflicts: {solver.NumConflicts()}")
        print(f"  - branches : {solver.NumBranches()}")
        print(f"  - wall time: {solver.WallTime()}s")

        self._model = model

        return model

    @property
    def model(self) -> Model | gp.Model | None:
        return self._model

    def get_params(self):
        """Get parameters for the model."""
        n_opt = len(self.operations)
        n_mach = len(self.machines)
        # operation_data = defaultdict(dict)

        # sample jobs_data
        # jobs_data = [
        #   # task = (machine_id, processing_time).
        #   [(0, 3), (1, 2), (2, 2)],  # Job0
        #   [(0, 2), (2, 1), (1, 4)],  # Job1
        #   [(1, 4), (2, 3)],  # Job2
        # ]
        # jobs_data = []
        # for i in range(n_opt):
        #     for _, pw_data in enumerate(operation_data[str(i)]["pw"]):
        #         jobs_data.append([(int(pw_data[0]), int(pw_data[1]))])

        # return (
        #     n_opt,
        #     n_mach,
        #     # operation_data,
        #     # machine_data,
        #     # processing time of operation i in machine m
        #     # para_p = np.full((n_opt, n_mach), dtype=object, fill_value=infinity)
        #     # time_estimates is para_p
        #     self.time_estimates,
        #     # para_h,
        #     # weight of operation i in machine m
        #     # para_w = np.empty((n_opt, n_mach), dtype=object)
        #     self.para_w,
        #     # para_delta,
        #     # set up time of machine m when processing operation i before j
        #     # a(i,j,m): setup time of machine m when processing operation i before j (aijm = -inf if
        #     # there is no setups)
        #     self.para_a,
        #     self.para_mach_capacity,
        #     self.jobs_data,
        # )

        return n_opt, n_mach

    def solve_ortools(self):
        assert self.model is not None

        solver = cp_model.CpSolver()
        status = solver.Solve(self._model)
        print(f"Status of solver = {status}.")

        pass
