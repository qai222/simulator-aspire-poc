"""Profiling utilities for the MILP and CP."""

import logging
import time
from collections import OrderedDict, defaultdict
import itertools as it

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from checking_constraints import (
    check_constraints_cp,
    check_constraints_milp,
    infer_var_x,
    infer_var_z,
)
from fjss import FJS2
from utils import *  # get_m_value, parse_data
from utils import parse_data


def load_data(input_fname="gfjsp_10_5_1.txt"):
    n_opt, n_mach, operation_data, machine_data = parse_data(input_fname)
    operation_data["0"]["h"] = 0
    operation_data["1"]["h"] = 0
    operation_data["2"]["h"] = 0
    operation_data["3"]["h"] = 20

    # define the parameters

    # minimum lag between the starting time of operation i and the ending time of operation j
    para_lmin = np.full((n_opt, n_opt), dtype=object, fill_value=-np.inf)
    # maximum lag between the starting time of operation i and the ending time of operation j
    para_lmax = np.full((n_opt, n_opt), dtype=object, fill_value=np.inf)

    # processing time of operation i in machine m
    # para_p = np.full((n_opt, n_mach), dtype=object, fill_value=np.inf)
    para_p = np.full((n_mach, n_opt), dtype=object, fill_value=np.inf)

    # the shape of h in the original file is (n_machine) while the shape of para_h in the
    # paper is (n_opt, n_mach)
    # maximum holding time of operation i in machine m
    para_h = np.empty((n_opt, n_mach), dtype=object)
    # para_h = np.empty(n_mach, dtype=object)

    # mapping of operation i to machine m
    # 20 for the furnaces, 0 for Cutting, Pressing, and Forging
    # 0: Cutting; 1: Pressing; 2: Forging; 3: Furnace
    holding_time_dict = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 20,
    }

    # weight of operation i in machine m
    # para_w = np.empty((n_opt, n_mach), dtype=object)
    # para_w = np.full((n_mach, n_opt), dtype=object, fill_value=np.inf)
    para_w = np.full((n_mach, n_opt), dtype=object, fill_value=0)

    # input/output delay time between two consecutive operations in mahcine m
    # para_delta = np.empty((n_mach), dtype=object)
    para_delta = np.full((n_mach), dtype=object, fill_value=0)

    # setup time of machine m when processing operation i before j
    # para_a = np.full((n_opt, n_opt, n_mach), dtype=object, fill_value=np.inf)
    para_a = np.full((n_mach, n_opt, n_opt), dtype=object, fill_value=-np.inf)

    # capacity of machine
    # para_mach_capacity = np.empty((n_mach), dtype=object)
    para_mach_capacity = np.full((n_mach), dtype=object, fill_value=0)
    for m in range(n_mach):
        # capacity of machine is a set of constant numbers
        para_mach_capacity[m] = machine_data[str(m)]["c"]

        # input/output delay time between two consecutive operations in mahcine m
        # delta(m): loading and unloading time of machine m (=1 for all machines)
        para_delta[m] = 1

        # set up time of machine m when processing operation i before j
        # a(i,j,m): setup time of machine m when processing operation i before j (aijm = -inf if there
        # is no setups)
        for idx_setup, setup_data in enumerate(machine_data[str(m)]["setup_data"][0]):
            para_a[m, int(setup_data[0]), int(setup_data[1])] = setup_data[2]

        # maximum holding time of operation i in machine m
        para_h[:, m] = holding_time_dict[str(machine_data[str(m)]["t"])]

    # lag time
    for i in range(n_opt):
        for idx_lag, lag_data in enumerate(operation_data[str(i)]["lag"]):
            # minimum lag between the starting time of operation i and the ending time of operation j
            para_lmin[i, int(lag_data[0])] = lag_data[1]
            # maximum lag between the starting time of operation i and the ending time of operation j
            para_lmax[i, int(lag_data[0])] = lag_data[2]

        for idx_pw, pw_data in enumerate(operation_data[str(i)]["pw"]):
            # operation_data[str(1)]["pw"]
            # # the shape of para_p in the original file is the transpose of the shape of para_p
            # para_p[i, int(pw_data[0])] = pw_data[1]
            # # the shape of para_w in the original file is the transpose of the shape of para_w
            # para_w[i, int(pw_data[0])] = pw_data[2]

            # the shape of para_p in the original file is the transpose of the shape of para_p
            para_p[int(pw_data[0]), i] = pw_data[1]
            # the shape of para_w in the original file is the transpose of the shape of para_w
            para_w[int(pw_data[0]), i] = pw_data[2]

    # reformat the shape of para_p and para_w
    para_p = para_p.T
    para_w = para_w.T

    # # reshape the shape of para_a
    # para_a = np.einsum("mij->ijm", para_a)

    return (
        n_opt,
        n_mach,
        operation_data,
        machine_data,
        para_lmin,
        para_lmax,
        para_p,
        para_h,
        para_w,
        para_delta,
        para_a,
        para_mach_capacity,
    )


def build_horizon_for_cp(
    para_p,
    para_h,
    para_lmax,
    inf_cp,
):
    """Build the horizon for the CP formulation checking."""

    para_p_horizon = np.copy(para_p)
    para_p_horizon[para_p_horizon == inf_cp] = 0
    para_h_horizon = np.copy(para_h)
    para_h_horizon[para_h_horizon == inf_cp] = 0
    para_lmax_horizon = np.copy(para_lmax)
    para_lmax_horizon[para_lmax_horizon == inf_cp] = 0
    horizon = (
        np.sum(para_p_horizon, axis=1)
        + np.sum(para_h_horizon, axis=1)
        + np.sum(para_lmax_horizon, axis=1)
    )
    horizon = int(np.sum(horizon)) + 1

    return horizon


def prepare_input(method, n_opt_selected, input_fname="gfjsp_10_5_1.txt"):
    """Prepare the input for the MILP and CP formulations."""
    (
        n_opt,
        n_mach,
        operation_data,
        machine_data,
        para_lmin,
        para_lmax,
        para_p,
        para_h,
        para_w,
        para_delta,
        para_a,
        para_mach_capacity,
    ) = load_data(input_fname=input_fname)
    # subselect the data
    n_opt = n_opt_selected
    para_lmin = para_lmin[:n_opt_selected, :n_opt_selected]
    para_lmax = para_lmax[:n_opt_selected, :n_opt_selected]
    para_p = para_p[:n_opt_selected, :]
    para_h = para_h[:n_opt_selected, :]
    para_w = para_w[:n_opt_selected, :]
    # the shape of parsed para_a is (n_mach, n_opt, n_opt)
    para_a = para_a[:, :n_opt_selected, :n_opt_selected]
    # machines
    machines = [str(i) for i in range(n_mach)]
    # operations
    operations = [str(i) for i in range(n_opt_selected)]

    if method.lower() == "milp":
        # ["method", "n_opt", "n_mach", "running_time_seconds", "num_constraints", "makespan"]
        # reshape the shape of para_a
        para_a = np.einsum("mij->ijm", para_a)

    elif method.lower() == "cp":
        pass
    else:
        raise ValueError("method must be either MILP or CP.")

    return (
        n_opt,
        n_mach,
        operations,
        machines,
        operation_data,
        machine_data,
        para_lmin,
        para_lmax,
        para_p,
        para_h,
        para_w,
        para_delta,
        para_a,
        para_mach_capacity,
    )


def check_fix_shape_of_para_a(para_p, para_a, intended_for="milp"):
    """Check and fix the shape of para_a when necessary."""

    # check the shape of para_a
    if intended_for.lower() == "milp":
        if para_a.shape[0] != para_p.shape[0]:
            para_a = np.einsum("mij->ijm", para_a)
    elif intended_for.lower() == "cp":
        if para_a.shape[0] != para_p.shape[1]:
            para_a = np.einsum("ijm->mij", para_a)
    else:
        raise ValueError("intended_for must be either MILP or CP.")

    return para_a


def run_single_milp(
    input_fname="gfjsp_10_5_1.txt",
    infinity=1.0e7,
    n_opt_selected=40,
    num_workers=16,
    verbose=False,
):
    """Solove a single FJSS problem."""
    new_row = OrderedDict()
    new_row["method"] = "MILP"

    print("loading and setting up data")
    (
        n_opt,
        n_mach,
        operations,
        machines,
        operation_data,
        machine_data,
        para_lmin,
        para_lmax,
        para_p,
        para_h,
        para_w,
        para_delta,
        para_a,
        para_mach_capacity,
    ) = prepare_input(
        method="milp", n_opt_selected=n_opt_selected, input_fname=input_fname
    )
    para_a = check_fix_shape_of_para_a(para_p, para_a, intended_for="milp")
    new_row["n_opt"] = n_opt
    new_row["n_mach"] = n_mach

    print("solve the MILP problem with FJSS4_v2")

    # check the running time
    start_time = time.time()
    fjss2 = FJS2(
        operations=operations,
        machines=machines,
        para_p=para_p,
        para_a=para_a,
        para_w=para_w,
        para_h=para_h,
        para_delta=para_delta,
        para_mach_capacity=para_mach_capacity,
        para_lmin=para_lmin,
        para_lmax=para_lmax,
        precedence=None,
        model_string=None,
        inf_milp=infinity,
        num_workers=num_workers,
        verbose=verbose,
        big_m=None,
    )
    fjss2.build_model_gurobi()
    fjss2.solve_gurobi()
    end_time = time.time()
    running_time_seconds = end_time - start_time
    new_row["running_time_seconds"] = running_time_seconds

    print("checking if the solution satisfies the constraints of MILP")
    model = fjss2.model
    # get the number of constraints
    new_row["num_constraints"] = model.NumConstrs
    # get the number of variables
    new_row["num_variables"] = model.NumVars
    # makespan
    makespan = model.objVal
    new_row["makespan"] = makespan

    var_x = fjss2.var_x.X
    var_y = fjss2.var_y.X
    var_z = fjss2.var_z.X
    var_s = fjss2.var_s.X
    var_c = fjss2.var_c.X
    var_c_max = fjss2.var_c_max.X
    big_m = fjss2.big_m

    try:
        para_a = check_fix_shape_of_para_a(
            fjss2.para_p, fjss2.para_a, intended_for="milp"
        )
        check_constraints_milp(
            var_y=var_y,
            var_s=var_s,
            var_c=var_c,
            var_c_max=var_c_max,
            operations=operations,
            machines=machines,
            para_p=fjss2.para_p,
            para_a=para_a,
            para_w=fjss2.para_w,
            para_h=fjss2.para_h,
            para_delta=fjss2.para_delta,
            para_mach_capacity=fjss2.para_mach_capacity,
            para_lmin=fjss2.para_lmin,
            para_lmax=fjss2.para_lmax,
            big_m=big_m,
            var_x=var_x,
            var_z=var_z,
        )
        print("the solution satisfies the constraints of MILP formulation.")
        new_row["feasible_MILP"] = "yes"
    except:  # pylint: disable=bare-except
        print("the solution does not satisfy the constraints of MILP formulation.")
        new_row["feasible_MILP"] = "no"

    print("checking if the solution satisfies the constraints of CP")

    horizon_milp_testing = build_horizon_for_cp(
        para_p=fjss2.para_p,
        para_h=fjss2.para_h,
        para_lmax=fjss2.para_lmax,
        inf_cp=infinity,
    )

    try:
        check_constraints_cp(
            var_y=var_y,
            var_s=var_s,
            var_c=var_c,
            var_c_max=var_c_max,
            operations=operations,
            machines=machines,
            para_p=fjss2.para_p,
            para_a=check_fix_shape_of_para_a(
                fjss2.para_p, fjss2.para_a, intended_for="cp"
            ),
            para_w=fjss2.para_w,
            para_h=fjss2.para_h,
            para_delta=fjss2.para_delta,
            para_mach_capacity=fjss2.para_mach_capacity,
            para_lmin=fjss2.para_lmin,
            para_lmax=fjss2.para_lmax,
            num_t=None,
            var_u=None,
            horizion=horizon_milp_testing,
        )
        print("the solution satisfies the constraints of CP formulation.")
        new_row["feasible_CP"] = "yes"
    except:  # pylint: disable=bare-except
        print("the solution does not satisfy the constraints of CP formulation.")
        new_row["feasible_CP"] = "no"

    if new_row["feasible_MILP"] == "yes" and new_row["feasible_CP"] == "yes":
        print("congragulations! Everything is good now.\n\n")

    return new_row


if __name__ == "__main__":
    new_row = run_single_milp(
        input_fname="gfjsp_10_5_1.txt",
        infinity=1.0e7,
        n_opt_selected=40,
        num_workers=16,
        verbose=True,
    )
