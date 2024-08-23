"""Build input data for FJSS."""

import os
import json
import math
from collections import OrderedDict

import numpy as np
import pandas as pd


def reassign_heater():
    """Resign heater for heating operations."""
    pass


def setup_inputs_random(
        json_fpath,
        random_seed=42,
        lmax_value=30,
):
    """Build input for FJSS model with randomly assigned time.

    Parameters
    ----------
    json_fpath : str
        Path to the json file that contains the operation information.
    random_seed : int
        Random seed for setting up the random time for each operation.
    lmax_value : int
        The maximum lag between the starting time of operation j and the ending time of operation i.
        We use a fixed value for now, but it should be updated in the future.

    Returns
    -------
    dict
        A dictionary that contains the following
        - operations: an ordered dictionary where each index is the string of integer index and the value is
        the operation type
        - machines: an ordered dictionary where each index is the integer index and the value is the machine
        name
        - machines_name_id_dict: an odered dictionary where the key is the machine name and the value is the
        integer index
        - machines_id_name_dict: an ordered dictionary where the key is the integer index and the
        value is the machine name
        - para_a: the setup time of machine m when processing operation i before j
        - para_w: weight of operation i in machine m
        - para_h: extra time of operation i in machine m after being processed
        - para_delta: input/output delay time between two consecutive operations in machine m
        - para_mach_capacity: the capacity of machine m
        - para_p: processing time of operation i in machine m
        - para_lmin: the minimum lag between the starting time of operation j and the ending time of
        operation i
        - para_lmax: the maximum lag between the starting time of operation j and the ending time of
        operation i
        - ws_operations_subset_indices: the indices of operations that need worker or accessory machines

    """
    with open(json_fpath, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # mapping of operation type to device/machine
    operation_machine_dict = json_data["operation_type_device_mapping"]
    # TODO: not dealing with the devices field for now
    # operation dict
    operation_dict = json_data["operation_dict"]
    # operation_dict = OrderedDict(sorted(operation_dict.items()))

    operation_dict = OrderedDict(operation_dict)
    operation_df = pd.DataFrame(operation_dict).T
    # get the unique machine list, which is the can_be_realized_by field in the operation_dict
    machine_list = []
    for machine in operation_df["can_be_realized_by"]:
        machine_list.extend(machine)
    machine_list = list(set(machine_list))
    # number of machines
    num_machines = len(machine_list)
    # number of operations
    num_operations = len(operation_dict)

    # simplified operations as a dictionary
    # where each index is the string of integer index and the value is the operation type
    operations = {
        str(i): operation["type"]
        for i, operation in zip(range(num_operations), operation_dict.values())
    }
    # TODO: this messed up the order of operations
    # operations = OrderedDict(sorted(operations.items()))

    # simplified machines as a dictionary
    # where each index is the integer index and the value is the machine name
    machines = {
        str(i): machine for i, machine in zip(range(len(machine_list)), machine_list)
    }
    # machines = OrderedDict(sorted(machines.items()))

    machines = OrderedDict(machines)
    machines_name_id_dict = {machine: i for i, machine in machines.items()}
    machines_name_id_dict = OrderedDict(sorted(machines_name_id_dict.items()))
    machines_id_name_dict = {v: k for k, v in machines_name_id_dict.items()}
    machines_id_name_dict = OrderedDict(sorted(machines_id_name_dict.items()))

    rng = np.random.default_rng(random_seed)

    # para_a: the setup time of machine m when processing operation i before j
    # para_a_ijm = -inf if there is no setups between operations i and j
    para_a = np.full((num_operations, num_operations, num_machines), -np.inf)

    for m, machine in enumerate(machines.values()):
        # operation i at temperature tempeture_i is processed before operation j with   temmperature_j
        for i, operation_i in enumerate(operation_dict.values()):
            for j, operation_j in enumerate(operation_dict.values()):
                if (
                    operation_i["type"] == "OPERATION_HEATING"
                    and operation_j["type"] == "OPERATION_HEATING"
                ):
                    temperature_i = operation_i["temperature"]
                    temperature_j = operation_j["temperature"]

                    # TODO: this is a temporary solution and needs to be updated once the values for
                    # temperature segmentations are changed
                    if temperature_i <= 300:
                        temperature_i = 300
                    elif temperature_i > 300 and temperature_i < 400:
                        temperature_i = 350
                    else:
                        temperature_i = 400

                    if temperature_j <= 300:
                        temperature_j = 300
                    elif temperature_j > 300 and temperature_j < 400:
                        temperature_j = 350
                    else:
                        temperature_j = 400

                    # no setup time if the temperature is the same
                    if temperature_i == temperature_j:
                        para_a[i, j, m] = 0
                    # setup time when temperature increases, that's heating
                    elif temperature_i < temperature_j:
                        para_a[i, j, m] = int(
                            math.ceil(abs(temperature_j - temperature_i) / 100 * 10)
                        )
                    # setup time when temperature decreases, that's cooling
                    else:
                        para_a[i, j, m] = int(
                            math.ceil(abs(temperature_i - temperature_j) / 100 * 25)
                        )

    # para_w: weight of operation i in machine m, with +inf as default value
    para_w = np.ones((num_operations, num_machines))

    # para_h: extra time of operation i in machine m after being processed, default value is 0 for now
    # TODO: need to change to random value in the future
    para_h = np.zeros((num_operations, num_machines))

    # para_delta: input/output delay time between two consecutive operations in machine m
    # this is related to loading and unloading time
    # TODO: for now, we assume there is no delay time, but needs to be updated in the future
    para_delta = np.zeros(num_machines)

    # para_mach_capacity: the capacity of machine m
    # TODO: for now, we assume the capacity of each machine is 1, and the heaters' capacity is 3
    # para_mach_capacity = np.full(num_machines, 6)
    para_mach_capacity = np.full(num_machines, 1)

    # assign all the heaters' capacity to 3
    # for machine in machines.values():
    #     if "HEATER" in machine:
    #         para_mach_capacity[int(machines_name_id_dict[machine])-1] = 3

    operations_subset_indices = []
    # para_p
    # processing time of operation i in machine m,
    # p_im = +inf if operation i cannot be realized by machine m
    # shape = (num_operations, num_machines)
    para_p = np.full((num_operations, num_machines), np.inf)
    for i, operation in enumerate(operation_dict.values()):
        needs_ws = [
            mach
            for mach in operation["can_be_realized_by"]
            if "ACC" in mach or "WORKER" in mach
        ]
        if len(needs_ws) > 0:
            operations_subset_indices.append(i)
        for machine in operation["can_be_realized_by"]:
            m = machines_name_id_dict.get(machine)
            # set random operation time based on the operation type
            # OPERATION_UNLOADING: random integer between 1 and 5 min
            if operation["type"] == "OPERATION_UNLOADING":
                para_p[int(i), int(m)] = rng.integers(1, 6)
            # "OPERATION_LOADING": random integer between 1 and 10 min
            elif operation["type"] == "OPERATION_LOADING":
                para_p[int(i), int(m)] = rng.integers(1, 11)
            # "OPERATION_ADDITION_LIQUID": random integer between 5 and 20 min
            elif operation["type"] == "OPERATION_ADDITION_LIQUID":
                para_p[int(i), int(m)] = rng.integers(5, 21)
            # "OPERATION_ADDITION_SOLID": random integer between 15 and 30 min
            elif operation["type"] == "OPERATION_ADDITION_SOLID":
                para_p[int(i), int(m)] = rng.integers(15, 31)
            # "OPERATION_HEATING": random integer between 180 and 260 min
            elif operation["type"] == "OPERATION_HEATING":
                para_p[int(i), int(m)] = rng.integers(180, 241)
            # "OPERATION_PURIFICATION": random integer between 120 and 180 min
            elif operation["type"] == "OPERATION_PURIFICATION":
                para_p[int(i), int(m)] = rng.integers(120, 180)
            # "OPERATION_RELOADING": random integer between 5 and 10
            elif operation["type"] == "OPERATION_RELOADING":
                para_p[int(i), int(m)] = rng.integers(5, 11)
            else:
                raise ValueError(f"Unknown operation type: {operation['type']}")

    # para_lmin: the minimum lag between the starting time of operation j and the ending time   ofoperation i
    # l_ij= −inf if there is no precedence relationship between operations i and j
    # TODO: we set the para_lmin to -np.inf for now, but needs to be updated in the future
    para_lmin = np.full((num_operations, num_operations), -np.inf)
    df_lmin = pd.DataFrame(para_lmin)
    df_lmin.columns = list(operation_dict.keys())
    df_lmin.index = list(operation_dict.keys())

    # para_lmax: the maximum lag between the starting time of operation j and the ending time of
    # operation i
    # l_ij= +inf if there is no precedence relationship between operations i and j
    para_lmax = np.full((num_operations, num_operations), np.inf)
    df_lmax = pd.DataFrame(para_lmax)
    df_lmax.columns = list(operation_dict.keys())
    df_lmax.index = list(operation_dict.keys())
    for i, operation in enumerate(operation_dict.values()):
        operation_id = operation["id"]
        precedents = operation["precedents"]
        if len(precedents) > 0:
            for precedent in precedents:
                df_lmin.at[precedent, operation_id] = 0

                # TODO: only set the lmax for OPERATION_ADDITION_SOLID for now
                # assuming that the newly prepared solution should be consumed within a certain time
                if operation["type"] == "OPERATION_ADDITION_SOLID":
                    df_lmax.at[precedent, operation_id] = lmax_value
                else:
                    df_lmax.at[precedent, operation_id] = np.inf

    para_lmin = df_lmin.to_numpy()
    para_lmax = df_lmax.to_numpy()

    return {
        "operations": operations,
        "machines": machines,
        "machines_name_id_dict": machines_name_id_dict,
        "machines_id_name_dict": machines_id_name_dict,
        "para_a": para_a,
        "para_w": para_w,
        "para_h": para_h,
        "para_delta": para_delta,
        "para_mach_capacity": para_mach_capacity,
        "para_p": para_p,
        "para_lmin": para_lmin,
        "para_lmax": para_lmax,
        "ws_operations_subset_indices": operations_subset_indices,
    }


def setup_input_deterministic():
    """Build input for FJSS model."""
    pass
