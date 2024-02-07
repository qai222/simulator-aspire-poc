import json
from collections import OrderedDict

import requests
import numpy as np
from pandas._typing import FilePath

ASKCOS_URL = "http://72.70.38.130"


def query_askcos_condition_rec(reaction_smiles: str) -> dict:
    response = requests.post(
        headers={"Content-type": "application/json"},
        url=f"{ASKCOS_URL}:9918/api/context-recommender/v2/predict/FP/call-sync",
        data=f'{{"smiles": "{reaction_smiles}", "n_conditions": 3}}',
    )
    response = response.content.decode("utf8")
    response = json.loads(response)
    assert response["status_code"] == 200
    return response


def drawing_url(smiles: str, size=80) -> str:
    smiles = smiles.replace("+", "%2b")
    q = f"/api/draw/?smiles={smiles}&transparent=true&annotate=false&size={size}"
    q = ASKCOS_URL + q
    return q


def json_dump(o, fn: FilePath):
    with open(fn, "w") as f:
        json.dump(o, f, indent=2)


def json_load(fn: FilePath):
    with open(fn, "r") as f:
        o = json.load(f)
    return o


def parse_data(input_fname):
    section_breaker = False
    operation_data = OrderedDict()
    machine_data = OrderedDict()

    machine_counter = 0
    with open(input_fname, "r") as f:
        # loop over the lines in the file
        for idx, line in enumerate(f):
            # # due date of operations
            d = []
            # starting time of time window of operation i
            a = []
            # ending time of time window of operation i
            b = []
            # number of machines that can process operation i
            nm = []
            # processing time of operations and weight of operation i
            pw_data = []
            # no(i): number of operation that must be process after operation i
            num_successor = []
            lag_data = []

            # machine data related
            # capacity of machine m
            c = []
            # type of machine m (0: Cutting; 1: Pressing; 2: Forging; 3: Furnace)
            t = []
            # nsu(m): number of setups (different than +inf) on machine m
            nsu = []
            # [op_i op_j SetUp(0,op_i, op_j)]xnsu(0)
            # where there are nsu(0) number of setups on machine 0
            setup_data = []

            line = line.strip()
            # check if line is empty
            if not line:
                section_breaker = True
                continue

            if section_breaker:
                line_data = np.array(line.split())
                line_data[line_data == "inf"] = np.inf
                line_data[line_data == "-inf"] = -np.inf
                line_data = line_data.astype(float).tolist()
                # parse machine data
                # C(m): capacity of machine m
                c.append(int(line_data[0]))
                # t(m): type of machine m (0: Cutting; 1: Pressing; 2: Forging; 3: Furnace)
                t.append(int(line_data[1]))
                # nsu(m): number of setups (different than +inf) on machine m
                nsu.append(int(line_data[2]))
                #  nsu(0)
                # [op_i op_j SetUp(0,op_i, op_j)]xnsu(0)
                tmp = line_data[3:]
                # reformat the list into a set of tuples where each tuple contains 3 values
                setup_data.append(tuple(tmp[i : i + 3] for i in range(0, len(tmp), 3)))

                machine_data[str(machine_counter)] = {
                    "c": int(line_data[0]),
                    "t": int(line_data[1]),
                    "nsu": int(line_data[2]),
                    "setup_data": setup_data,
                }
                machine_counter += 1

            else:
                # parse operation data
                if idx == 0:
                    # n_opt: number of operations
                    # n_mach: number of machines
                    n_opt, n_mach = map(int, line.split())
                else:
                    # parse operation data
                    line_data = np.array(line.split())
                    line_data[line_data == "inf"] = np.inf
                    line_data[line_data == "-inf"] = -np.inf
                    line_data = line_data.astype(float).tolist()
                    # d(i): due date of operation i where i is the operation index
                    d.append(line_data[0])

                    # a(i): starting time of time window of operation i
                    a.append(line_data[1])
                    # b(i): ending time of time window of operation i
                    b.append(line_data[2])
                    # nm(i): number of machines that can process operation i
                    nm_i = int(line_data[3])
                    nm.append(nm_i)
                    # p(i,m): processing time of operation i in machine m
                    # w(i): weight of operation i
                    for idx_pw in range(nm_i):
                        pw_data.append(
                            tuple(line_data[4 + idx_pw * 3 : 4 + idx_pw * 3 + 3])
                        )
                    # pw_end_idx = 4 + n_mach * 3
                    # pw_data.append(line_data[4:pw_end_idx])

                    # no(i): number of operation that must be process after operation i
                    num_successor_i = int(line_data[nm_i * 3 + 4])
                    num_successor.append(num_successor_i)

                    # [op_id lmin(0,op_id) lmax(0,op_id)]xno(0)
                    # where there are no(0) number of operations that must be process after     operation 0
                    # there are 3 values in one tuple
                    for idx_oper in range(num_successor_i):
                        lag_data.append(
                            tuple(
                                line_data[
                                    nm_i * 3
                                    + 5
                                    + idx_oper * 3 : nm_i * 3
                                    + 5
                                    + idx_oper * 3
                                    + 3
                                ]
                            )
                        )

                    operation_data[str(idx - 1)] = {
                        "a": line_data[1],
                        "b": line_data[2],
                        "nm": int(line_data[3]),
                        # "pw": line_data[4 : 4 + nm_i * 3],
                        "pw": pw_data,
                        # "no": line_data[4 + nm_i * 3],
                        "no": num_successor,
                        # "lag": line_data[4 + nm_i * 3 + 1 : 4 + nm_i * 3 + 4],
                        "lag": lag_data,
                    }

    return n_opt, n_mach, operation_data, machine_data  # pylint: disable=E0601


def get_big_m_value(para_p, para_h, para_lmin, para_a, infinity):
    """Implementation after discussion with Runzhong."""
    n_opt, n_mach = para_h.shape

    # eq. (17)
    p_i = []
    for i in range(n_opt):
        p_i_max = []
        for m in range(n_mach):
            if para_p[i, m] < infinity:
                p_i_max.append(para_p[i, m] + para_h[i, m])
        p_i.append(max(p_i_max))

    # eq. (18)
    l_i = []
    for i in range(n_opt):
        l_i_max = []
        for j in range(n_opt):
            if para_lmin[i, j] >= 0:
                l_i_max.append(para_lmin[i, j])
        # print(f"l_i_max={l_i_max}")
        if len(l_i_max) == 0:
            l_i_max.append(0)
        else:
            l_i.append(max(l_i_max))

    # eq. (19)
    a_i = []
    for i in range(n_opt):
        a_i_max = []
        for j in range(n_opt):
            for m in range(n_mach):
                if para_p[i, m] < infinity:
                    a_i_max.append(para_a[i, j, m])
        a_i.append(max(a_i_max))

    # eq. (20)
    l_a_max = [max(l_element, a_element) for l_element, a_element in zip(l_i, a_i)]
    big_m = sum(p_i + l_a_max)
    return big_m
