from __future__ import annotations

from reaction_network.schema import NetworkLv0
from reaction_network.utils import json_dump

"""
collect more info based on Jenna's data
this produces the input for ChemScraper
"""

if __name__ == '__main__':
    nw = NetworkLv0.from_routes("../data/routes.json", seed=2, n_target=None, scraper_output=None)
    json_dump(nw.unique_molecular_smis, "scraper_input.json")
