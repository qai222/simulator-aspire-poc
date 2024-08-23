"""Test the FJSS module."""

import itertools as it

import numpy as np
import pytest
from numpy.testing import assert_allclose

from reaction_network.fjss import FJS2
from reaction_network.sample_data import build_sample_example_data
from reaction_network.utils import check_fix_shape_of_para_a

eps = 1e-6
infinity = 1e7


def checking_fjss_milp(
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
    var_v: np.array = None, # work shift assginment variable
    ws_s: np.array = None, # start time of work shift
    ws_c: np.array = None, # completion time of work shift
    tol: float = 1.0e-6,
):
    """Check the constraints of fjss with MILP fomulation, fjss2.

    Parameters
    ----------
    var_y : np.ndarray
        Binary variable indicating the assignment of operation i to machine j.
    var_s : np.ndarray
        Starting time of operation i.
    var_c : np.ndarray
        Completion time of operation i.
    var_c_max : float | int
        Makespan of the problem.
    operations : list[str]
        List of operation names.
    machines : list[str]
        List of machine names.
    para_p : np.ndarray
        Processing time of operation i on machine j.
    para_a : np.ndarray
        Setup time of operation i before operation j on machine m.
    para_w : np.ndarray
        Weight of operation i on machine m.
    para_h : np.ndarray
        Holding cost of operation i on machine m.
    para_delta : np.ndarray
        input/output delay time between two consecutive operations in machine m.
    para_mach_capacity : list[int] | np.ndarray
        Capacity of machine m.
    para_lmin : np.ndarray
        Minimum time between operation i and j. This indicates the precedence relationship.
    para_lmax : np.ndarray
        Maximum time between operation i and j.
    big_m : float | int
        Big M value.
    var_x : np.array, optional
        Binary variable indicating if operation is is processed before operation j.
    var_z : np.array, optional
        Binary variable indicating if operation i is processed before operation j on machine m.
    var_v : np.array, optional
        Binary variable indicating the work shift assignment.
    ws_s : np.array, optional
        Start time of work shift.
    ws_c : np.array, optional
        Completion time of work shift.
    tol : float, optional
        Tolerance value. This is required due to the numerical precision of the solver.

    """
    n_opt = len(operations)
    n_mach = len(machines)

    print(f"shape of para_a = {para_a.shape}")

    # eq. (2)
    for i in range(n_opt):
        assert var_c_max >= var_c[i] - tol

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
                ) # , f"var_s[i]={var_s[i]} \n var_c[j]={var_c[j]}\n para[j, i, m]={para_a[j, i, m]} \n big_m={big_m} \n var_x[i, j]={var_x[i, j]} \n var_y[i, m]={var_y[i, m]} \n var_y[j, m]={var_y[j, m]} \n difference={var_s[i] - var_c[j] - para_a[j, i, m] + big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])}"
                # eq. (10)
                assert var_s[j] + eps >= var_s[i] + para_delta[m] - big_m * (
                    3 - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                # eq. (11)
                assert var_s[i] + eps >= var_s[j] + para_delta[m] - big_m * (
                    2 + var_x[i, j] - var_y[i, m] - var_y[j, m]
                ) # , f"var_s[i]={var_s[i]}, var_s[j]={var_s[j]}, para_delta[m]={para_delta[m]}, big_m={big_m}, var_x[i, j]={var_x[i, j]}, var_y[i, m]={var_y[i, m]}, var_y[j, m]={var_y[j, m]}, difference={var_s[i] - var_s[j] - para_delta[m] + big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])}"
                # assert var_s[i] + eps >= var_s[j] + para_delta[m] - big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m]), f"difference = {var_s[i] - var_s[j] - para_delta[m] + big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])}"
                # eq. (12)
                assert var_c[j] + eps >= var_c[i] + para_delta[m] - big_m * (
                    3 - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                # eq. (13)
                assert var_c[i] + eps >= var_c[j] + para_delta[m] - big_m * (
                    2 + var_x[i, j] - var_y[i, m] - var_y[j, m]
                ) # , f"var_c[i]={var_c[i]}, var_c[j]={var_c[j]}, para_delta[m]={para_delta[m]}, big_m={big_m}, var_x[i, j]={var_x[i, j]}, var_y[i, m]={var_y[i, m]}, var_y[j, m]={var_y[j, m]}, difference={var_c[i] - var_c[j] - para_delta[m] + big_m * (2 + var_x[i, j] - var_y[i, m] - var_y[j, m])}"
                # eq. (14)
                assert var_s[j] + eps >= var_c[i] - big_m * (
                    3 + var_z[i, j, m] - var_x[i, j] - var_y[i, m] - var_y[j, m]
                )
                # eq. (15)
                assert var_s[i] + eps >= var_c[j] - big_m * (
                    2 + var_z[j, i, m] + var_x[i, j] - var_y[i, m] - var_y[j, m]
                ) # f"var_s[i]={var_s[i]}, var_c[j]={var_c[j]}, big_m={big_m}, var_z[i, j, m]={var_z[j,i, m]}, var_x[i, j]={var_x[i, j]}, var_y[i, m]={var_y[i, m]}, var_y[j, m]={var_y[j, m]}, difference={var_s[i] - var_c[j] + big_m * (2 + var_z[j,i,m] + var_x[i, j] - var_y[i, m] - var_y[j, m])}"

    # eq. (16)
    if var_z is not None:
        for i, m in it.product(range(n_opt), range(n_mach)):
            expr_constraint_left = [
                para_w[j, m] * var_z[i, j, m] for j in range(n_opt) if i != j
            ]
            expr_constraint_right = (para_mach_capacity[m] - para_w[i, m]) * var_y[i, m]
            assert (
                sum(expr_constraint_left) <= expr_constraint_right
            ) # , f"sum(expr_constraint_left)={sum(expr_constraint_left)}, expr_constraint_right={expr_constraint_right}, difference={sum(expr_constraint_left) - expr_constraint_right}, C_m = {para_mach_capacity[m]}, w_i = {para_w[i, m]}, y_i = {var_y[i, m]}"

    # checking the work shift constraints
    if var_v:
        # sum of work shift assignment columns should be equal to 1
        for i in range(n_opt):
            assert sum(var_v[i, :]) == 1

        # loop over var_v with their indices to
        # check the starting and finishing time of each operation for each work shift
        for i, j in it.product(range(n_opt), range(n_opt)):
            if var_v[i, j] == 1:
                assert (ws_s[j] <= var_s[i]) and (ws_c[j] >= var_c[i]) == 1


def test_fjss_milp_no_ws():
    """Test the fjss_milp function with toy example without work shift constraints."""

    # load the toy example data
    operations, machines, para_lmin, para_lmax, para_p, para_h, para_w, para_a, para_mach_capacity, para_delta = build_sample_example_data()

    # solve the problem with FJSS2
    print("solving the toy problem with FJS2")

    fjss2 = FJS2(
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
        # precedence=None,
        model_string=None,
        inf_milp=infinity,
        workshifts=None,
        operations_subset_indices=None,
        num_workers=None,
        verbose=True,
        big_m=None,
    )

    print(f"para_w: {para_w}")
    fjss2.build_model_gurobi()
    fjss2_output = fjss2.solve_gurobi()

    # check the constraints
    # makespan
    assert_allclose(24, fjss2.model.objVal, rtol=eps)

    # other variables
    var_x = fjss2.var_x.X
    var_y = fjss2.var_y.X
    var_z = fjss2.var_z.X
    var_s = fjss2.var_s.X
    var_c = fjss2.var_c.X
    var_c_max = fjss2.var_c_max.X
    big_m = fjss2.big_m

    # checking the constraints
    para_a = check_fix_shape_of_para_a(fjss2.para_p, fjss2.para_a, intended_for="milp")
    checking_fjss_milp(
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
        var_v= None,
        ws_s= None,
        ws_c= None,
        tol=eps,
    )


def test_fjss_milp_with_ws_01():
    """Test the fjss_milp function with toy example with work shift constraints."""

    # load the toy example data
    operations, machines, para_lmin, para_lmax, para_p, para_h, para_w, para_a, para_mach_capacity, para_delta = build_sample_example_data()

    # solve the problem with FJSS2
    print("solving the toy problem with FJS2")

    fjss2 = FJS2(
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
        # precedence=None,
        model_string=None,
        inf_milp=infinity,
        workshifts=[(30, 10)],
        operations_subset_indices=None,
        num_workers=None,
        verbose=True,
        big_m=None,
    )

    print(f"para_w: {para_w}")
    fjss2.build_model_gurobi()
    fjss2_output = fjss2.solve_gurobi()

    # check the constraints
    # makespan
    assert_allclose(24, fjss2.model.objVal, rtol=eps)

    # other variables
    var_x = fjss2.var_x.X
    var_y = fjss2.var_y.X
    var_z = fjss2.var_z.X
    var_s = fjss2.var_s.X
    var_c = fjss2.var_c.X
    var_c_max = fjss2.var_c_max.X
    big_m = fjss2.big_m

    # checking the constraints
    para_a = check_fix_shape_of_para_a(fjss2.para_p, fjss2.para_a, intended_for="milp")
    checking_fjss_milp(
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
        var_v= None,
        ws_s= None,
        ws_c= None,
        tol=eps,
    )


def test_fjss_milp_with_ws_02():
    """Test the fjss_milp function with toy example with infeasible work shift constraints."""

    # load the toy example data
    operations, machines, para_lmin, para_lmax, para_p, para_h, para_w, para_a, para_mach_capacity, para_delta = build_sample_example_data()

    # solve the problem with FJSS2
    print("solving the toy problem with FJS2")

    fjss2 = FJS2(
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
        # precedence=None,
        model_string=None,
        inf_milp=infinity,
        workshifts=[(20, 10)],
        operations_subset_indices=None,
        num_workers=None,
        verbose=True,
        big_m=None,
    )

    fjss2.build_model_gurobi()

    # this case will not have a feasible solution
    # capture the warning message
    with pytest.warns(UserWarning) as record:
        fjss2_output = fjss2.solve_gurobi()

    assert len(record) == 1
    assert str(record[0].message) == "No solution found."
    assert fjss2_output is None
