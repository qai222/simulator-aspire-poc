from __future__ import annotations

from reaction_network.schema import NetworkLv0
from reaction_network.utils import json_dump

"""
LV0. given the routes selection seed, scraper_input, and scraper_output

note: this script needs `scraper_output.json` generated by `ChemScraper` using `scraper_input.json`
"""

if __name__ == '__main__':
    network = NetworkLv0.from_files(scraper_input="scraper_input.json",
                                    scraper_output="scraper_output.json")
    json_dump(network.model_dump(), "network_lv0.json")