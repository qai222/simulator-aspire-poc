from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import requests
from pandas._typing import FilePath

ASKCOS_URL = "http://72.70.38.130"


def query_askcos_condition_rec(reaction_smiles: str, return_query: bool = False) -> dict | tuple[dict, dict]:
    q = dict(
        headers={"Content-type": "application/json"},
        url=f"{ASKCOS_URL}:9918/api/context-recommender/v2/predict/FP/call-sync",
        data=f'{{"smiles": "{reaction_smiles}", "n_conditions": 3}}',
    )
    response = requests.post(**q)
    response = response.content.decode('utf8')
    response = json.loads(response)
    assert response['status_code'] == 200
    if return_query:
        return response, q
    else:
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
                setup_data.append(tuple(tmp[i: i + 3] for i in range(0, len(tmp), 3)))

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
                            tuple(line_data[4 + idx_pw * 3: 4 + idx_pw * 3 + 3])
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
                                + idx_oper * 3: nm_i * 3
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


# from https://stackoverflow.com/questions/16699180
MM_of_Elements = {'H': 1.00794, 'D': 2.014, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107,
                  'N': 14.0067,
                  'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
                  'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
                  'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
                  'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64,
                  'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
                  'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42,
                  'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6,
                  'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
                  'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
                  'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
                  'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
                  'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
                  'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278,
                  'Th': 232.03806, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614,
                  'Cm': 247.0703, 'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951,
                  'No': 259.1009, 'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
                  'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294,
                  '': 0}
