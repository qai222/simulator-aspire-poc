"""Build a toy example for testing the MILP model."""

from collections import OrderedDict
import numpy as np
from reaction_network.utils import check_fix_shape_of_para_a


def build_sample_example_data():
    """Build a toy example for testing the MILP model."""

    num_operations = 5
    num_machines = 3

    operations = {str(i): str(i) for i in range(num_operations)}
    operations = OrderedDict(sorted(operations.items(), key=lambda x: int(x[0])))

    machines = {str(i): str(i) for i in range(num_machines)}
    machines = OrderedDict(sorted(machines.items(), key=lambda x: int(x[0])))


    # minimum lag time
    para_lmin = np.full((5, 5), fill_value=-np.inf)
    para_lmin[0, 1] = 2
    para_lmin[2, 3] = 2
    para_lmin[3, 4] = 2
    # para_lmin

    # maximum lag time
    para_lmax = np.full((5, 5), fill_value=np.inf)
    para_lmax[0, 1] = 5
    para_lmax[2, 3] = 5
    para_lmax[3, 4] = 5

    # para_p
    para_p = np.full((5, 3), fill_value=np.inf)
    for i in [idx - 1 for idx in [1, 3, 5]]:
        for m in [idx - 1 for idx in [1, 3]]:
            para_p[i, m] = 5

    for i in [idx - 1 for idx in [2, 4]]:
        for m in [1]:
            para_p[i, m] = 10
    # para_p

    # para_h
    para_h = np.full((5, 3), fill_value=2)

    # para_w
    para_w = np.full((5, 3), fill_value=1)

    # para_a, a_mij
    para_a = np.full((3, 5, 5), fill_value=-np.inf)
    para_a[0, 0, 4] = 5
    para_a[0, 4, 0] = 5

    # para_mach_capacity
    para_mach_capacity = np.array([2, 2, 2])

    # para_delta
    para_delta = np.array([1, 1, 1])

    para_a = check_fix_shape_of_para_a(para_p, para_a, intended_for="milp")

    return operations, machines, para_lmin, para_lmax, para_p, para_h, para_w, para_a, para_mach_capacity, para_delta
