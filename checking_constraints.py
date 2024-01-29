# check constrints

import numpy as np
import itertools as it

eps = 1e-6


def check_constraints_milp(
    var_y: np.ndarray,
    var_s: np.ndarray,
    var_c: np.ndarray,
    var_c_max: float | int,
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

    print(f"shape of para_a = {para_a.shape}")

    # eq. (2)
    for i in range(n_opt):
        assert var_c_max >= var_c[i]

    for i in range(n_opt):
        # eq. (3)
        assert var_c[i] + eps >= var_s[i] + sum(
            [para_p[i, m] * var_y[i, m] for m in range(n_mach)]
        )  # , f"var_c[i]={var_c[i]}, var_s[i]={var_s[i]}, i={i}, sum={sum([para_p[i, m] * var_y[i, m] for m in range(n_mach)])}"
        # eq. (4)
        assert var_c[i] - eps <= var_s[i] + sum(
            [(para_p[i, m] + para_h[i, m]) * var_y[i, m] for m in range(n_mach)]
        )
        # eq. (5)
        assert sum([var_y[i, m] for m in range(n_mach)]) == 1

    for i, j in it.product(range(n_opt), range(n_opt)):
        if i != j:
            # eq. (6)
            assert var_s[j] + eps >= var_c[i] + para_lmin[i, j]
            # eq. (7)
            assert var_s[j] - eps <= var_c[i] + para_lmax[i, j]

    if var_x is not None:
        for i, j, m in it.product(range(n_opt), range(n_opt), range(n_mach)):
            if i < j:
                # eq. (8)
                assert var_s[j] + eps >= var_c[i] + para_a[i, j, m] - big_m * (
                    3 - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )  # , f"var_s[j]={var_s[j]}, var_c[i]={var_c[i]}, para_a[i, j, m]={para_a[i, j, m]}, big_m={big_m}, var_x[i, j]={var_x[i, j]}, var_y[i, m]={var_y[i, m]}, var_y[j, m]={var_y[j, m]}"
                # eq. (9)
                assert var_s[i] + eps >= var_c[j] + para_a[j, i, m] - big_m * (
                    2 + var_x[i, j] - var_y[i, m] - var_y[j, m]
                ), f"var_s[i]={var_s[i]} \n var_c[j]={var_c[j]}\n para[j, i, m]={para_a[j, i, m]} \n big_m={big_m} \n var_x[i, j]={var_x[i, j]} \n var_y[i, m]={var_y[i, m]} \n var_y[j, m]={var_y[j, m]} \n difference={var_s[i] - var_c[j] - para_a[j, i, m] + big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])}"
                # eq. (10)
                assert var_s[j] + eps >= var_s[i] + para_delta[m] - big_m * (
                    3 - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                # eq. (11)
                assert var_s[i] + eps >= var_s[j] + para_delta[m] - big_m * (
                    2 + var_x[i, j] - var_y[i, m] - var_y[j, m]
                ), f"var_s[i]={var_s[i]}, var_s[j]={var_s[j]}, para_delta[m]={para_delta[m]}, big_m={big_m}, var_x[i, j]={var_x[i, j]}, var_y[i, m]={var_y[i, m]}, var_y[j, m]={var_y[j, m]}, difference={var_s[i] - var_s[j] - para_delta[m] + big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])}"
                # assert var_s[i] + eps >= var_s[j] + para_delta[m] - big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m]), f"difference = {var_s[i] - var_s[j] - para_delta[m] + big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])}"
                # eq. (12)
                assert var_c[j] + eps >= var_c[i] + para_delta[m] - big_m * (
                    3 - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                # eq. (13)
                assert var_c[i] + eps >= var_c[j] + para_delta[m] - big_m * (
                    2 + var_x[i, j] - var_y[i, m] - var_y[j, m]
                ), f"var_c[i]={var_c[i]}, var_c[j]={var_c[j]}, para_delta[m]={para_delta[m]}, big_m={big_m}, var_x[i, j]={var_x[i, j]}, var_y[i, m]={var_y[i, m]}, var_y[j, m]={var_y[j, m]}, difference={var_c[i] - var_c[j] - para_delta[m] + big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])}"
                # eq. (14)
                assert var_s[j] + eps >= var_c[i] - big_m * (
                    3 + var_z[i, j, m] - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                # eq. (15)
                assert var_s[i] + eps >= var_c[j] - big_m * (
                    2 + var_z[j, i, m] + var_x[i, j] - var_y[i, m] - var_y[j, m]
                ), # f"var_s[i]={var_s[i]}, var_c[j]={var_c[j]}, big_m={big_m}, var_z[i, j, m]={var_z[j,i, m]}, var_x[i, j]={var_x[i, j]}, var_y[i, m]={var_y[i, m]}, var_y[j, m]={var_y[j, m]}, difference={var_s[i] - var_c[j] + big_m * (2 + var_z[j,i,m] + var_x[i, j] - var_y[i, m] - var_y[j, m])}"

    # eq. (16)
    if var_z is not None:
        # for i, m in it.product(range(n_opt), range(n_mach)):
        #     expr_constraint = []
        #     for j in range(n_opt):
        #         if i != j:
        #             expr_constraint.append(para_w[j, m] * var_z[i, j, m])
        #     assert (
        #         sum(expr_constraint)
        #         <= (para_mach_capacity[m] - para_w[i, m]) * var_y[i, m]
        #     ), f"sum(expr_constraint)={sum(expr_constraint)},
        #     para_mach_capacity[m]={para_mach_capacity[m]}, para_w[i, m]={para_w[i, m]}, var_y[i,
        #     m]={var_y[i, m]}, difference={sum(expr_constraint) - (para_mach_capacity[m] -
        #     para_w[i, m]) * var_y[i, m]}"

        for i, m in it.product(range(n_opt), range(n_mach)):
            expr_constraint_left = [
                para_w[j, m] * var_z[i, j, m] for j in range(n_opt) if i != j
            ]
            expr_constraint_right = (para_mach_capacity[m] - para_w[i, m]) * var_y[i, m]
            assert (
                sum(expr_constraint_left) <= expr_constraint_right
            ), f"sum(expr_constraint_left)={sum(expr_constraint_left)}, expr_constraint_right={expr_constraint_right}, difference={sum(expr_constraint_left) - expr_constraint_right}, C_m = {para_mach_capacity[m]}, w_i = {para_w[i, m]}, y_i = {var_y[i, m]}"


def check_constraints_cp(
    var_y: np.ndarray,
    var_s: np.ndarray,
    var_c: np.ndarray,
    var_c_max: float | int,
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
    horizion: int | float,
    num_t: int = None,
    var_u: np.array = None,
):
    """Check the constraints of the CP problem."""
    n_opt = len(operations)
    n_mach = len(machines)

    for i, m in it.product(range(n_opt), range(n_mach)):
        # eq. (3)
        assert var_c[i] >= var_s[i] + sum(
            [para_p[i, m] * var_y[i, m] for m in range(n_mach)]
        )
        # eq. (4)
        assert var_c[i] <= var_s[i] + sum(
            [(para_p[i, m] + para_h[i, m]) * var_y[i, m] for m in range(n_mach)]
        )
        # eq. (5)
        assert sum([var_y[i, m] for m in range(n_mach)]) == 1

    for i, j in it.product(range(n_opt), range(n_opt)):
        if i != j:
            # eq. (6)
            assert var_s[j] >= var_c[i] + para_lmin[i, j]
            # eq. (7)
            assert var_s[j] <= var_c[i] + para_lmax[i, j]

    # check eq. (22)
    for i, j, m in it.product(range(n_opt), range(n_opt), range(n_mach)):
        if i != j:
            if var_y[i, m] == 1 and var_y[j, m] == 1:
                bool_left = var_s[j] >= var_c[i] + para_a[m, i, j]
                bool_right = var_s[i] >= var_c[j] + para_a[m, j, i]
                assert bool_left or bool_right

    # check eq. (23)
    for i, j, m in it.product(range(n_opt), range(n_opt), range(n_mach)):
        if i != j:
            if var_y[i, m] == 1 and var_y[j, m] == 1:
                bool_left = (
                    var_s[j]
                    >= var_s[i]
                    + max([0, var_c[i] - var_s[i] - var_c[j] + var_s[j]])
                    + para_delta[m]
                )
                bool_right = (
                    var_s[i]
                    >= var_s[j]
                    + max([0, var_c[j] - var_s[j] - var_c[i] + var_s[i]])
                    + para_delta[m]
                )
                assert bool_left or bool_right

    if num_t is None:
        num_t = int(horizion / 1.0e0)

    if var_u is None and num_t is not None:
        print("inferring var_u with eq. (24)")
        num_t_list = range(num_t)
        var_u = np.zeros((n_opt, n_mach, num_t))
        for i, m, t in it.product(range(n_opt), range(n_mach), num_t_list):
            # use eq. (24) to assign value to var_u
            if var_s[i] <= t and t <= var_c[i]:
                var_u[i, m, t] = para_w[i, m]
            # else:
            #     var_u[i, m, t] = 0

    if num_t is not None and var_u is not None:
        print("checking eq. (24)")
        num_t_list = range(num_t)
        for i, m in it.product(range(n_opt), range(n_mach)):
            for t in num_t_list:
                # check eq. (24)
                if var_s[i] <= t and t <= var_c[i]:
                    assert var_u[i, m, t] == para_w[i, m]
                else:
                    assert var_u[i, m, t] == 0

    # eq. (25)
    for m in range(n_mach):
        yu_list = []
        for t in range(num_t):
            for i in range(n_opt):
                yu_list.append(var_y[i, m] * var_u[i, m, t])
        yu_max = max(yu_list)
        assert yu_max <= para_mach_capacity[m]


def infer_var_x(var_s: np.ndarray):
    """Infer variable x based on results from CP.

    Parameters
    ----------
    var_s : np.ndarray
        Start time of each operation.

    Returns
    -------
    var_x : np.ndarray
        Variable x where x[i, j] = 1 if operation i is before operation j.

    """
    n_opt = var_s.shape[0]
    var_x = np.zeros((n_opt, n_opt))

    for i, j in it.product(range(n_opt), range(n_opt)):
        if i != j:
            if var_s[i] < var_s[j]:
                var_x[i, j] = 1
            else:
                var_x[i, j] = 0

    return var_x


def infer_var_z(var_s: np.ndarray, var_y: np.ndarray, var_c: np.ndarray):
    """Infer variable z based on results from CP.

    Parameters
    ----------
    var_s : np.ndarray
        Start time of each operation. Shape is (n_opt,).
    var_y : np.ndarray
        Variable y where y[i, m] = 1 if operation i is assigned to machine m.
    var_c : np.ndarray
        Completion time of each operation. Shape is (n_opt,).

    Returns
    -------
    var_z : np.ndarray
        Variable z where z[i, j, m] = 1 if operation i is before and overlapped operation j on machine m.

    """
    n_opt = var_y.shape[0]
    n_mach = var_y.shape[1]

    var_z = np.zeros((n_opt, n_opt, n_mach))

    for i, j, m in it.product(range(n_opt), range(n_opt), range(n_mach)):
        if i != j:
            if var_y[i, m] == 1 and var_y[j, m] == 1:
                if var_s[j] < var_c[i]:
                    var_z[i, j, m] = 1
                else:
                    var_z[i, j, m] = 0

    return var_z
