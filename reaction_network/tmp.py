# %% [markdown]
# ## Prototyping the MLIP way for GFJSP
#

# %%
from itertools import product
from collections import OrderedDict

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import numpy as np

from utils import parse_data, get_m_value

# %%
# n_opt, n_mach, operation_data, machine_data = parse_data("gfjsp_10_10_1.txt")
n_opt, n_mach, operation_data, machine_data = parse_data(
    "fjss_cp_data/gfjsp_10_5_1.txt"
)

# %%


# %% [markdown]
# ## Solve the problem with MILP
#

# %%
from ortools.linear_solver import pywraplp

# maximum holding time of operation i in machine m
# =20 for the furnaces, 0 for Cutting, Pressing, and Forging)
# (0: Cutting; 1: Pressing; 2: Forging; 3: Furnace)
operation_data["0"]["h"] = 0
operation_data["1"]["h"] = 0
operation_data["2"]["h"] = 0
operation_data["3"]["h"] = 20


# declare the MIP solver using the Gurobi backend
solver = pywraplp.Solver(
    "SolveIntegerProblem", pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING
)
# solver = pywraplp.Solver.CreateSolver("SCIP")

infinity = 1.0e7

# infinity = np.inf
# infinity = solver.infinity()

# minimum lag between the starting time of operation i and the ending time of operation j
para_lmin = np.full((n_opt, n_opt), dtype=object, fill_value=-infinity)
# maximum lag between the starting time of operation i and the ending time of operation j
para_lmax = np.full((n_opt, n_opt), dtype=object, fill_value=infinity)

# processing time of operation i in machine m
# para_p = np.full((n_opt, n_mach), dtype=object, fill_value=infinity)
para_p = np.full((n_mach, n_opt), dtype=object, fill_value=infinity)

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
para_w = np.full((n_mach, n_opt), dtype=object, fill_value=infinity)

# input/output delay time between two consecutive operations in mahcine m
para_delta = np.empty((n_mach), dtype=object)

# setup time of machine m when processing operation i before j
# para_a = np.full((n_opt, n_opt, n_mach), dtype=object, fill_value=infinity)
para_a = np.full((n_mach, n_opt, n_opt), dtype=object, fill_value=infinity)

# capacity of machine
para_mach_capacity = np.empty((n_mach), dtype=object)
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
# reshape the shape of para_a
para_a = np.einsum("mij->ijm", para_a)

# the big M value
big_m = get_m_value(para_p=para_p, para_h=para_h, para_lmin=para_lmin, para_a=para_a)
# big_m = 1.0e4

# replace infinities with infinity from the solver
para_lmin[para_lmin == -np.inf] = -infinity
para_lmin[para_lmin == np.inf] = infinity
para_lmax[para_lmax == np.inf] = infinity
para_lmax[para_lmax == -np.inf] = -infinity

para_p[para_p == np.inf] = infinity
para_p[para_p == -np.inf] = -infinity
para_w[para_w == np.inf] = infinity
para_w[para_w == -np.inf] = -infinity

para_a[para_a == np.inf] = infinity
para_a[para_a == -np.inf] = -infinity


# =================================================
# only take part of the data for debugging purpose
n_opt = 8
para_a = para_a[:n_opt, :n_opt, :]
para_lmin = para_lmin[:n_opt, :n_opt]
para_lmax = para_lmax[:n_opt, :n_opt]
para_p = para_p[:n_opt, :]
para_w = para_w[:n_opt, :]
para_h = para_h[:n_opt, :]
# =================================================

# %%
# n_opt_selected = 8
# n_opt = n_opt_selected

# para_lmin = para_lmin[:n_opt_selected, :n_opt_selected]
# para_lmax = para_lmax[:n_opt_selected, :n_opt_selected]

# para_p = para_p[:n_opt_selected, :]
# para_h = para_h[:n_opt_selected, :]
# para_w = para_w[:n_opt_selected, :]

# para_a = para_a[:n_opt_selected, :n_opt_selected, :]

# %%


# %%
# create variables
ort_infinity = solver.infinity()

# if operation i is processed by machine m
var_y = np.empty((n_opt, n_mach), dtype=object)
for i, m in product(range(n_opt), range(n_mach)):
    var_y[i, m] = solver.BoolVar(f"y_{i}_{m}")

# if operation i is processed before operation j
var_x = np.empty((n_opt, n_opt), dtype=object)
for i, j in product(range(n_opt), range(n_opt)):
    var_x[i, j] = solver.BoolVar(f"x_{i}_{j}")

# if operation i is processed before operation j on machine m
var_z = np.empty((n_opt, n_opt, n_mach), dtype=object)
for i, j, m in product(range(n_opt), range(n_opt), range(n_mach)):
    var_z[i, j, m] = solver.BoolVar(f"z_{i}_{j}_{m}")

# starting time of operation i
var_s = np.empty((n_opt), dtype=object)
for i in range(n_opt):
    var_s[i] = solver.NumVar(0, ort_infinity, f"s_{i}")

# completion time of operation i
var_c = np.empty((n_opt), dtype=object)
for i in range(n_opt):
    var_c[i] = solver.NumVar(0, ort_infinity, f"c_{i}")

# make span
var_c_max = solver.NumVar(0, ort_infinity, "C_max")

# %%
# create objective
solver.Minimize(var_c_max)

for i in range(n_opt):
    # eq. (2)
    solver.Add(var_c_max >= var_c[i])

    # eq. (3)
    # !!!!!!!!! expr = [para_p[i, m] * var_y[i, m] for i in range(n_mach)]
    expr = [para_p[i, m] * var_y[i, m] for m in range(n_mach)]
    solver.Add(var_c[i] >= var_s[i] + sum(expr))

    # eq. (4)
    # !!!!!!!!!! expr = [(para_p[i, m] + para_h[i, m]) * var_y[i, m] for i in range(n_mach)]
    expr = [(para_p[i, m] + para_h[i, m]) * var_y[i, m] for m in range(n_mach)]
    solver.Add(var_c[i] <= var_s[i] + sum(expr))

# eq. (5)
# !!!!!!!!!!!!!!
# for m in range(n_mach):
#     # sum of y_im = 1
#     solver.Add(sum([var_y[i, m] for i in range(n_opt)]) == 1)
for i in range(n_opt):
    # sum of y_im = 1
    solver.Add(sum([var_y[i, m] for m in range(n_mach)]) == 1)

for i, j in product(range(n_opt), range(n_opt)):
    # eq. (6)
    # minimum lag between the starting time of operation i and the ending time of operation j
    solver.Add(var_s[j] >= var_c[i] + para_lmin[i, j])
    # eq. (7)
    # maximum lag between the starting time of operation i and the ending time of operation j
    solver.Add(var_s[j] <= var_c[i] + para_lmax[i, j])

# ==================================================================
# comment out the following constraints for debugging purpose

for i, j in product(range(n_opt), range(n_opt)):
    # eq. (6)
    # minimum lag between the starting time of operation i and the ending time of operation j
    solver.Add(var_s[j] >= var_c[i] + para_lmin[i, j])
    # eq. (7)
    # maximum lag between the starting time of operation i and the ending time of operation j
    solver.Add(var_s[j] <= var_c[i] + para_lmax[i, j])

for i, j, m in product(range(n_opt), range(n_opt), range(n_mach)):
    if i < j:
        # eq. (8)
        # setup time of machine m when processing operation i before j
        solver.Add(
            var_s[j]
            >= var_c[i]
            + para_a[i, j, m]
            - big_m * (3 - var_x[i, j] - var_y[i, m] - var_y[j, m])
        )
        # eq. (9)
        # setup time of machine m when processing operation i before j
        solver.Add(
            var_s[i]
            >= var_c[i]
            + para_a[i, j, m]
            - big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])
        )

# eq. (10) and (11)
for i, m in product(range(n_opt), range(n_mach)):
    if i < j:
        # eq. (10)
        solver.Add(
            var_s[j]
            >= var_s[i]
            + para_delta[m]
            - big_m * (3 - var_x[i, j] - var_y[i, m] - var_y[j, m])
        )
        # eq. (11)
        solver.Add(
            var_s[i]
            >= var_s[j]
            + para_delta[m]
            - big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])
        )

# eq. (12) and (13)
for i, m in product(range(n_opt), range(n_mach)):
    if i < j:
        # eq. (12)
        solver.Add(
            var_c[j]
            >= var_c[i]
            + para_delta[m]
            - big_m * (3 - var_x[i, j] - var_y[i, m] - var_y[j, m])
        )
        # eq. (13)
        solver.Add(
            var_c[i]
            >= var_c[j]
            + para_delta[m]
            - big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])
        )

# eq. (14) and (15)
for i, j, m in product(range(n_opt), range(n_opt), range(n_mach)):
    if i < j:
        # eq. (14)
        solver.Add(
            var_s[j]
            >= var_c[i]
            - big_m * (3 + var_z[i, j, m] - var_x[i, j] - var_y[i, m] - var_y[j, m])
        )
        # eq. (15)
        solver.Add(
            var_s[i]
            >= var_c[j]
            - big_m * (2 + var_z[i, j, m] + var_x[i, j] - var_y[i, m] - var_y[j, m])
        )

# eq. (16)
for i in range(n_opt):
    for j in range(n_opt):
        expr = []
        for m in range(n_mach):
            if i != j:
                # solver.Add(var_w[j, m] * var_z[i, j, m] >= var_w[i, m])
                expr.append(para_w[j, m] * var_z[i, j, m])
        solver.Add(
            solver.Sum(expr) <= (para_mach_capacity[m] - para_w[i, m]) * var_y[i, m]
        )

# ==================================================================

# %%


# %%
# solve the problem
# solver.set_time_limit(600 * 1000)
solver.EnableOutput()
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print(f"opt_obj = {solver.Objective().Value():.4f}")

# %%
print(f"status: {status}")

# %%

# %%
solver.Objective().Value()

# %%
var_c_max.solution_value()

# %%
