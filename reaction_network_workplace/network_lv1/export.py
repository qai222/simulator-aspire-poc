from __future__ import annotations

import json
import random

from reaction_network.schema import NetworkLv0, NetworkLv1
from reaction_network.utils import json_dump, json_load

"""
define a reaction network

LV0. given the routes selection seed, scraper_input, and scraper_output
LV1. given the target product amounts, and expected yields
"""

if __name__ == '__main__':
    network_lv0 = json_load("../network_lv0/network_lv0.json")
    network_lv0 = NetworkLv0(**network_lv0)

    random.seed(42)
    network_lv1 = NetworkLv1.from_target_masses_and_expected_yields(
        target_masses={smi: 0.5 for smi in network_lv0.product_smis},
        expected_yields={smi: random.uniform(0.5, 1) for smi in network_lv0.reaction_dict},
        network_lv0=network_lv0
    )
    json_dump(network_lv1.model_dump(), "network_lv1.json")
