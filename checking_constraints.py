# check constrints

import numpy as np
import itertools as it

def check_constraints_milp(
    var_y: np.ndarray,
    var_s: np.ndarray,
    var_c: np.ndarray,
    var_c_max: float|int,
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
    big_m: float | int,
    var_x: np.array = None,
    var_z: np.array = None,
):
    """Check the constraints of the MILP problem."""
    n_opt = len(operations)
    n_mach = len(machines)

    # eq. (2)
    for i in np.arange(n_opt):
        assert var_c_max >= var_c[i]

    for i in np.arange(n_opt):
        # eq. (3)
        assert var_c[i] >= var_s[i] + sum([para_p[i, m] * var_y[i, m] for m in np.arange(n_mach)])
        # eq. (4)
        assert var_c[i] <= var_s[i] + sum([(para_p[i, m] + para_h[i, m]) * var_y[i, m] for m in np.arange(n_mach)])
        # eq. (5)
        assert sum([var_y[i, m] for m in np.arange(n_mach)]) == 1

    for i, j in it.product(np.arange(n_opt), np.arange(n_opt)):
        if i != j:
            # eq. (6)
            assert var_s[j] >= var_c[i] + para_lmin[i, j]
            # eq. (7)
            assert var_s[j] <= var_c[i] + para_lmax[i, j]

    if var_x is not None:
        for i, j, m in it.product(np.arange(n_opt), np.arange(n_opt), np.arange(n_mach)):
            if i < j:
                # eq. (8)
                assert var_s[j] >= var_c[i] + para_a[i, j, m] - big_m * (3 - var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (9)
                assert var_s[i] >= var_c[j] + para_a[j, i, m] - big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (10)
                assert var_s[j] >= var_s[i] + para_delta[m] - big_m * (3 - var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (11)
                assert var_s[i] >= var_s[j] + para_delta[m] - big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (12)
                assert var_c[j] >= var_c[i] + para_delta[m] - big_m * (3 - var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (13)
                assert var_c[i] >= var_c[j] + para_delta[m] - big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (14)
                assert var_s[j] >= var_c[i] - big_m * (3 + var_z[i, j, m] - var_x[i, j] - var_y[i, m] - var_y[j, m])
                # eq. (15)
                assert var_s[i] >= var_c[j] - big_m * (2 + var_z[i, j, m] + var_x[i, j] - var_y[i, m] - var_y[j, m])

    # eq. (16)
    if var_z is not None:
        for i, m in it.product(np.arange(n_opt), np.arange(n_mach)):
            for j in np.arange(n_opt):
                expr_constraint = []
                if i != j:
                    expr_constraint.append(para_w[j, m] * var_z[i, j, m])
            assert sum(expr_constraint) <= (para_mach_capacity[m] - para_w[i, m]) * var_y[i, m]


def check_constraints_cp(
    var_y: np.ndarray,
    var_s: np.ndarray,
    var_c: np.ndarray,
    var_c_max: float|int,
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
    num_t: int=None,
    var_u: np.array= None,
):
    """Check the constraints of the CP problem."""
    n_opt = len(operations)
    n_mach = len(machines)

    for i, m in it.product(np.arange(n_opt), np.arange(n_mach)):
        # eq. (3)
        assert var_c[i] >= var_s[i] + sum([para_p[i, m] * var_y[i, m] for m in np.arange(n_mach)])
        # eq. (4)
        assert var_c[i] <= var_s[i] + sum([(para_p[i, m] + para_h[i, m]) * var_y[i, m] for m in np.arange(n_mach)])
        # eq. (5)
        assert sum([var_y[i, m] for m in np.arange(n_mach)]) == 1

    for i, j in it.product(np.arange(n_opt), np.arange(n_opt)):
        if i != j:
            # eq. (6)
            assert var_s[j] >= var_c[i] + para_lmin[i, j]
            # eq. (7)
            assert var_s[j] <= var_c[i] + para_lmax[i, j]

    # check eq. (22)
    for i, j, m in it.product(range(n_opt), range(n_opt), range(n_mach)):
        if i != j:
            if var_y[i, m] == 1 and var_y[j, m] == 1:
                bool_left = (var_s[j] >= var_c[i] + para_a[m, i, j])
                bool_right = (var_s[i] >= var_c[j] + para_a[m, j, i])
                assert bool_left or bool_right

    # check eq. (23)
    for i, j, m in it.product(range(n_opt), range(n_opt), range(n_mach)):
        if i != j:
            if var_y[i, m] == 1 and var_y[j, m] == 1:
                bool_left = (
                    var_s[j] >= var_s[i] + max([0, var_c[i] - var_s[i] - var_c[j] + var_s[j]]) + para_delta[m]
                )
                bool_right = (
                    var_s[i] >= var_s[j] + max([0, var_c[j] - var_s[j] - var_c[i] + var_s[i]]) + para_delta[m]
                )
                assert bool_left or bool_right


    num_t_list = np.arange(num_t)
    for i, m in it.product(range(n_opt), range(n_mach)):
        for t in num_t_list:
            # check eq. (24)
            if var_s[i] <= t and t <= var_c[i]:
                assert var_u[i, m, t] == para_w[i, m]
            else:
                assert var_u[i, m, t] == 0

    # eq. (25)
    for m in np.arange(n_mach):
        yu_list = []
        for t in num_t_list:
            for i in np.arange(n_opt):
                yu_list.append(var_y[i,m] * var_u[i, m, t])
        yu_max = max(yu_list)
        assert yu_max <= para_mach_capacity[m]
