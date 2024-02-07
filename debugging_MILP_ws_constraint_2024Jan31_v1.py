#
# %%
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
from fjss import FJSS2, FJSS4_v2
from utils import *  # get_m_value, parse_data
from utils import parse_data
from profiling_utils import *

# %%
input_fname = "gfjsp_10_5_1.txt"
infinity = 1.0e7
n_opt_selected = 25
num_workers = 6
verbose = False
# %%

new_row = OrderedDict()
new_row["method"] = "CP"

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
) = prepare_input(method="milp", n_opt_selected=n_opt_selected, input_fname=input_fname)
para_a = check_fix_shape_of_para_a(para_p, para_a, intended_for="milp")
new_row["n_opt"] = n_opt
new_row["n_mach"] = n_mach

# print(f"symmstric of para_lmin: {para_lmin == para_lmin.T}")

#
# para_lmin_new = np.triu(np.full_like(para_lmin, 1))
# para_lmin_new[np.diag_indices(para_lmin_new.shape[0])] = 0
# para_lmin_new[para_lmin_new == 0] = -np.inf

print("solve the MILP problem with FJSS4_v2")
# check the running time
start_time = time.time()
fjss4 = FJSS4_v2(
    operations=operations,
    machines=machines,
    para_p=para_p,
    para_a=para_a,
    para_w=para_w,
    para_h=para_h,
    para_delta=para_delta,
    para_mach_capacity=para_mach_capacity,
    # para_lmin=para_lmin_new,
    para_lmin=para_lmin,
    # para_lmax=np.full_like(para_lmax, np.inf),
    para_lmax=para_lmax,
    precedence=None,
    model_string=None,
    inf_milp=infinity,
    # num_workshifts=None, # not used
    shift_durations=None,
    # shift_durations=400, # works
    # shift_durations=1000, # sees the effect of the constraint
    # shift_durations=374, # 324 is the limit
    # shift_durations=375,
    # shift_durations=323,
    # shift_durations=70, # infeasible
    # shift_durations=270, # infeasible
    # shift_durations=2000, # works
    # shift_durations=1650, # works
    operations_subset_indices=None,
    num_workers=num_workers,
    verbose=verbose,
    big_m=None,
    matrix_variables=True,
)
fjss4.build_model_gurobi()
fjss4_output = fjss4.solve_gurobi()
end_time = time.time()
running_time_seconds = end_time - start_time
new_row["running_time_seconds"] = running_time_seconds
print("checking if the solution satisfies the constraints of MILP")
model = fjss4.model
# get the number of constraints
new_row["num_constraints"] = model.NumConstrs
# get the number of variables
new_row["num_variables"] = model.NumVars
# makespan
makespan = model.objVal
new_row["makespan"] = makespan
var_x = fjss4.var_x.X
var_y = fjss4.var_y.X
var_z = fjss4.var_z.X
var_s = fjss4.var_s.X
var_c = fjss4.var_c.X
var_c_max = fjss4.var_c_max.X
big_m = fjss4.big_m
try:
    para_a = check_fix_shape_of_para_a(fjss4.para_p, fjss4.para_a, intended_for="milp")
    check_constraints_milp(
        var_y=var_y,
        var_s=var_s,
        var_c=var_c,
        var_c_max=var_c_max,
        operations=operations,
        machines=machines,
        para_p=fjss4.para_p,
        para_a=para_a,
        para_w=fjss4.para_w,
        para_h=fjss4.para_h,
        para_delta=fjss4.para_delta,
        para_mach_capacity=fjss4.para_mach_capacity,
        para_lmin=fjss4.para_lmin,
        para_lmax=fjss4.para_lmax,
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
    para_p=fjss4.para_p,
    para_h=fjss4.para_h,
    para_lmax=fjss4.para_lmax,
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
        para_p=fjss4.para_p,
        para_a=check_fix_shape_of_para_a(fjss4.para_p, fjss4.para_a, intended_for="cp"),
        para_w=fjss4.para_w,
        para_h=fjss4.para_h,
        para_delta=fjss4.para_delta,
        para_mach_capacity=fjss4.para_mach_capacity,
        para_lmin=fjss4.para_lmin,
        para_lmax=fjss4.para_lmax,
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
