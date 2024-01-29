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
) = prepare_input(method="cp", n_opt_selected=n_opt_selected, input_fname=input_fname)
para_a = check_fix_shape_of_para_a(para_p, para_a, intended_for="cp")
new_row["n_opt"] = n_opt
new_row["n_mach"] = n_mach

print("solving the CP problem with FJSS2")
# checking the running time
start_time = time.time()
fjss2 = FJSS2(
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
    inf_cp=infinity,
    num_workers=num_workers,
    verbose=verbose,
)
fjss2.build_model_ortools()
# print("big_m from fjss3", fjss3.big_m)
fjss2.solve_ortools()
running_time_seconds = time.time() - start_time
new_row["running_time_seconds"] = running_time_seconds

print("checking if the solution satisfies the constraints of MILP")
solver = fjss2._solver
model = fjss2._model
# get the number of constraints
new_row["num_constraints"] = len(model.Proto().constraints)
# get the number of variables
new_row["num_variables"] = len(model.Proto().variables)
# the makespan
new_row["makespan"] = solver.ObjectiveValue()


def get_values(var):
    """Vectorize to get the values of the variables."""
    return solver.Value(var)


# get the number of constraints
v_get_values = np.vectorize(get_values)
var_y = v_get_values(fjss2.var_y)
var_s = v_get_values(fjss2.var_s)
var_c = v_get_values(fjss2.var_c)
var_u = v_get_values(fjss2.var_u)
yu_list = v_get_values(fjss2.yu_list)
var_c_max = fjss2.var_c_max
num_t = v_get_values(fjss2.num_t)
# infer var_x
var_x = infer_var_x(var_s)
# infer var_z
var_z = infer_var_z(var_s=var_s, var_y=var_y, var_c=var_c)
print("checking if the solution satisfies the constraints of MILP")
para_a = check_fix_shape_of_para_a(fjss2.para_p, fjss2.para_a, intended_for="milp")
# para_a = np.einsum("mij->ijm", fjss2.para_a)
big_m = get_m_value_runzhong(
    para_p=fjss2.para_p,
    para_h=fjss2.para_h,
    para_lmin=fjss2.para_lmin,
    para_a=para_a,
    infinity=infinity,
)
try:
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
para_a = check_fix_shape_of_para_a(fjss2.para_p, fjss2.para_a, intended_for="cp")
try:
    check_constraints_cp(
        var_y=var_y,
        var_s=var_s,
        var_c=var_c,
        var_c_max=var_c_max,
        operations=operations,
        machines=machines,
        para_p=fjss2.para_p,
        para_a=fjss2.para_a,
        para_w=fjss2.para_w,
        para_h=fjss2.para_h,
        para_delta=fjss2.para_delta,
        para_mach_capacity=fjss2.para_mach_capacity,
        para_lmin=fjss2.para_lmin,
        para_lmax=fjss2.para_lmax,
        num_t=num_t,
        var_u=var_u,
        horizion=fjss2.horizon,
    )
    new_row["feasible_CP"] = "yes"
    print("the solution satisfies the constraints of CP formulation.")
except:  # pylint: disable=bare-except
    new_row["feasible_CP"] = "no"
    print("the solution does not satisfy the constraints of CP formulation.")
if new_row["feasible_MILP"] == "yes" and new_row["feasible_CP"] == "yes":
    print("congragulations! Everything is good now.\n\n")
