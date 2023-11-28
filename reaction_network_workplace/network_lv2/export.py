from __future__ import annotations

from reaction_network.schema import NetworkLv1, MaterialTransform
from reaction_network.schema.lv2 import BENCH_TOP_LV2
from reaction_network.utils import json_dump, json_load

if __name__ == '__main__':
    network_lv1 = json_load("../network_lv1/network_lv1.json")
    MaterialTransform.build_transforms_from_network(NetworkLv1(**network_lv1))
    json_dump(BENCH_TOP_LV2.model_dump(), "bench_top_lv2.json")
