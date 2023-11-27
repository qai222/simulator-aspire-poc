from __future__ import annotations

import random

from loguru import logger

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
    target_masses = {smi: random.uniform(0.3, 0.5) for smi in network_lv0.product_smis}
    expected_yields = {smi: random.uniform(0.7, 1) for smi in network_lv0.reaction_dict}

    network_lv1 = NetworkLv1.from_target_masses_and_expected_yields(
        target_masses=target_masses,
        expected_yields=expected_yields,
        network_lv0=network_lv0
    )
    for r in network_lv1.reactions:
        logger.info(f"REACTION: {r.reaction_lv0.reaction_smiles}")
        logger.info(f"volume sum of the reaction mixture: {r.volume_sum}")
        logger.info(f"the solvent is identified as: {r.solvent_smi}")
        logger.info(f"this is a reaction that makes a target: {r.product.compound_lv0.smiles in target_masses}\n")
    json_dump(network_lv1.model_dump(), "network_lv1.json")
