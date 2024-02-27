from __future__ import annotations


from reaction_network.schema import NetworkLv1, Operation
from reaction_network.schema.lv2 import BENCH_TOP_LV2
from reaction_network.fjss import FJS1
from reaction_network.utils import json_dump, json_load

if __name__ == '__main__':
    network_lv1 = json_load("step02_network_lv1.json")
    Operation.build_transforms_from_network(NetworkLv1(**network_lv1))
    fjs = FJS1.from_bench_top(BENCH_TOP_LV2, None)
    fjs.build_model_gurobi()
    fjs_output = fjs.solve_gurobi()
    # import pprint
    # pprint.pp(fjs_output.model_dump())
    print(fjs_output.machine_timetable)
    # fjs.solve_cplex()

