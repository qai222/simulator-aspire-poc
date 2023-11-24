import json
import random

from loguru import logger
from pandas._typing import FilePath

from reaction_network.schema import ReactionLv0, NetworkLv0

"""
collect more info based on Jenna's data
this produces the input for ChemScraper
"""


def load_routes(routes_file: FilePath = "data/routes.json", n_target=5, sample_seed=42) -> dict:
    with open(routes_file, "r") as f:
        routes = json.load(f)
    routes = {k: routes[k] for k in routes if len(routes[k]['Reactions']) > 1}  # exclude orphans
    random.seed(sample_seed)
    routes = {k: routes[k] for k in random.sample(sorted(routes.keys()), k=n_target)}
    logger.info(f"loaded routes for: {sorted(routes.keys())}")
    return routes


def collect_reactions(routes_file: FilePath = "data/routes.json", n_target=5, sample_seed=42):
    routes = load_routes(routes_file, n_target, sample_seed)
    reaction_smis = []
    for target, data in routes.items():
        for r in data['Reactions']:
            r_smi = r['smiles']
            if r_smi.startswith(">>"):
                continue
            reaction_smis.append(r_smi)
    reaction_smis = sorted(set(reaction_smis))
    lv0_reactions = [ReactionLv0.from_reaction_smiles(smi) for smi in reaction_smis]
    network = NetworkLv0(
        reactions=lv0_reactions,
        n_targets=n_target,
        source_file=routes_file,
        seed=sample_seed,
        compounds=[],
    )
    return network


if __name__ == '__main__':
    NETWORK_LV0 = collect_reactions(routes_file="../data/routes.json", n_target=5, sample_seed=2)
    with open("scraper_input.json", "w") as f:
        s = NETWORK_LV0.model_dump_json(indent=2)
        f.write(s)
