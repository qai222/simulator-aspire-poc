from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from typing import Any

import networkx as nx
from loguru import logger
from pandas._typing import FilePath
from pydantic import BaseModel
from pysmiles import read_smiles

from reaction_network.schema.provenance import get_provenance_model
from reaction_network.utils import query_askcos_condition_rec, drawing_url, json_load, MM_of_Elements
from reaction_network.visualization import CytoNodeData, CytoEdge, CytoNode, CytoEdgeData


class ReactionLv0Base(BaseModel):
    """
    an unquantified (i.e. consumption of chemicals are unknown) reaction,
    usually generated from predictive chemistry models
    """

    reaction_smiles: str
    """ the reaction smiles that contains `>>`, i.e. reagents are excluded """

    reactant_stoichiometry: dict[str, float]
    """ a mapping from `molecular smiles` to `stoichiometric coefficient` for reactants """

    reagent_stoichiometry: dict[str, float]
    """ a mapping from `molecular smiles` to `stoichiometric coefficient` for reagents """

    temperature: float
    """ in C """

    @property
    def stoichiometry(self) -> dict[str, float]:
        """
        a mapping from `molecular smiles` to `stoichiometric coefficient` for all species,
        note product always has a coefficient of `-1`
        """
        assert not set(self.reagent_stoichiometry.keys()).intersection(self.reactant_stoichiometry.keys())
        d = dict()
        d.update(self.reactant_stoichiometry)
        d.update(self.reagent_stoichiometry)
        d.update({self.product_smi: - 1.0})
        return d

    @property
    def product_smi(self):
        """ molecular smiles of the product """
        s = self.reaction_smiles.split(">>")[-1]
        assert "." not in s
        return s

    @property
    def unique_molecular_smis(self):
        """ a sorted set of molecular smiles for all species """
        return sorted(self.stoichiometry.keys())


class ReactionLv0(get_provenance_model(ReactionLv0Base, "ReactionLv0_")):

    @classmethod
    def from_reaction_smiles(cls: type[ReactionLv0Base], reaction_smiles: str):
        response, query = query_askcos_condition_rec(reaction_smiles, return_query=True)
        result = response['result'][0]

        return cls(
            reaction_smiles=reaction_smiles,
            reactant_stoichiometry=result['reactants'],
            reagent_stoichiometry=result['reagents'],
            temperature=result['temperature'],

            reactant_stoichiometry__provenance=query,
            reagent_stoichiometry__provenance=query,
            temperature__provenance=query,
        )


ReactionLv0: type[ReactionLv0Base]


class CompoundLv0Base(BaseModel):
    smiles: str

    state_of_matter: str

    molecular_weight: float  # g/mol

    density: float  # g/mL


class CompoundLv0(get_provenance_model(CompoundLv0Base, "CompoundLv0_")):

    @staticmethod
    def parse_scraper_output(scraper_output: FilePath) -> dict[str, tuple[str | None, float | None]]:
        with open(scraper_output, "r") as f:
            output = json.load(f)
        output_data = dict()
        for smi, data in output.items():
            try:
                form_string = data['from']
                if "liquid" in form_string.lower():
                    form = "LIQUID"
                else:
                    form = "SOLID"
            except KeyError:
                form = None
            try:
                density_string = data['density']
                try:
                    density_string = re.findall("\d+\.\d+\s*g\/mL", density_string)[0]
                    density_string = density_string.replace("g/mL", "").strip()
                except IndexError:
                    density_string = re.findall("\d+\.\d+\s*g\/cm", density_string)[0]
                    density_string = density_string.replace("g/cm", "").strip()
                density = float(density_string)
            except KeyError:
                density = None
            output_data[smi] = (form, density)
        return output_data

    @staticmethod
    def make_up_compound_info(smiles: str):
        logger.warning(f"making up properties for: {smiles}")
        mol = read_smiles(smiles, explicit_hydrogen=True)
        mw = 0
        for _, element in mol.nodes(data='element'):
            mw += MM_of_Elements[element]
        assert mw > 1
        if mw < 200 and all(m not in smiles for m in ['Li', 'Na', 'K', 'Mg', 'Ca', 'Pd']):
            form = "LIQUID"
            density = 1.0
        else:
            form = "SOLID"
            density = 1.4
        return mw, form, density

    @staticmethod
    def get_default_compound_lv0(smiles: str):
        mw, form, density = CompoundLv0.make_up_compound_info(smiles)
        mw_p = "calculated using pysmiles"
        form_p = "made up"
        density_p = "made up"
        return CompoundLv0(
            smiles=smiles, state_of_matter=form, molecular_weight=mw, density=density,
            state_of_matter__provenance=form_p,
            molecular_weight__provenance=mw_p,
            density__provenance=density_p,
        )

    @classmethod
    def from_smiles(cls, smiles, scraper_output: FilePath = None):
        if scraper_output is None:
            return CompoundLv0.get_default_compound_lv0(smiles)
        else:
            output_data = CompoundLv0.parse_scraper_output(scraper_output)
            if smiles not in output_data:
                return CompoundLv0.get_default_compound_lv0(smiles)
            else:
                form, density = output_data[smiles]
                mw, form_mu, density_mu = CompoundLv0.make_up_compound_info(smiles)
                form_p = f"from scraper output: {scraper_output}"
                density_p = f"from scraper output: {scraper_output}"
                mw_p = "calculated using pysmiles"
                if form is None:
                    form = form_mu
                    form_p = "made up"
                if density is None:
                    density = density_mu
                    density_p = "made up"
            return CompoundLv0(
                smiles=smiles, state_of_matter=form, molecular_weight=mw, density=density,
                state_of_matter__provenance=form_p,
                molecular_weight__provenance=mw_p,
                density__provenance=density_p,
            )


CompoundLv0: type[CompoundLv0Base]


class NetworkLv0(BaseModel):
    seed: int
    compounds: list[CompoundLv0]
    reactions: list[ReactionLv0]
    provenance: Any = None

    @property
    def summary(self):
        g = self.to_nx()
        n_reactions = 0
        n_materials = 0
        for n in g.nodes:
            if ">>" in n:
                n_reactions += 1
            else:
                n_materials += 1
        return {
            "number of unique reactions": n_reactions,
            "number of unique materials": n_materials,
            "number of unique starting materials": len(self.starting_smis),
            "number of unique target materials": len(self.target_smis),
        }

    @classmethod
    def from_routes(cls, routes_file: FilePath, seed: int = 42, n_target: int | None = 5,
                    scraper_output: FilePath = None, ):
        routes = json_load(routes_file)
        routes = {k: routes[k] for k in routes if len(routes[k]['Reactions']) > 1}  # exclude orphans
        random.seed(seed)
        if n_target:
            routes = {k: routes[k] for k in random.sample(sorted(routes.keys()), k=n_target)}
        else:
            routes = {k: routes[k] for k in sorted(routes.keys())}

        reaction_smis = []
        for target, data in routes.items():
            for r in data['Reactions']:
                r_smi = r['smiles']
                if r_smi.startswith(">>"):
                    continue
                reaction_smis.append(r_smi)
        reaction_smis = sorted(set(reaction_smis))
        lv0_reactions = [ReactionLv0.from_reaction_smiles(smi) for smi in reaction_smis]
        network = cls(
            reactions=lv0_reactions,
            seed=seed,
            compounds=[],
            provenance=routes_file
        )
        network.populate_compounds(scraper_output=scraper_output)
        return network

    @property
    def unique_molecular_smis(self) -> list[str]:
        smis = []
        for r in self.reactions:
            smis += r.unique_molecular_smis
        return sorted(set(smis))

    def populate_compounds(self, scraper_output: FilePath | None):
        if scraper_output is None:
            logger.warning(f"no scraper output given, "
                           f"populating {len(self.unique_molecular_smis)} compounds using made up properties!")
        for smi in self.unique_molecular_smis:
            c = CompoundLv0.from_smiles(smi, scraper_output=scraper_output)
            self.compounds.append(c)

    @property
    def target_smis(self) -> list[str]:
        g = self.to_nx()
        return [n for n in g.nodes if g.out_degree(n) == 0 and g.in_degree(n) > 0]

    @property
    def compound_dict(self) -> dict[str, CompoundLv0]:
        return {r.smiles: r for r in self.compounds}

    @property
    def reaction_dict(self) -> dict[str, ReactionLv0]:
        return {r.reaction_smiles: r for r in self.reactions}

    def to_nx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for reaction in self.reactions:
            for smi in reaction.reactant_stoichiometry:
                g.add_edge(smi, reaction.reaction_smiles)
            for smi in reaction.reagent_stoichiometry:
                g.add_edge(smi, reaction.reaction_smiles)
            g.add_edge(reaction.reaction_smiles, reaction.product_smi)
        return g

    def to_cyto_elements(self):
        g = self.to_nx()
        cyto_nodes = []
        cyto_edges = []
        for n in g.nodes:
            if ">>" in n:
                data = self.reaction_dict[n].model_dump()
                classes = "reaction"
            else:
                try:
                    data = self.compound_dict[n].model_dump()
                except KeyError:
                    data = {}
                classes = "compound"
            cnd = CytoNodeData(id=n, label=n, url=drawing_url(n), data=data)
            cn = CytoNode(data=cnd, classes=classes, group="nodes")
            cyto_nodes.append(cn)
        for u, v in g.edges:
            ce = CytoEdge(data=CytoEdgeData(id=f"{u} {v}", source=u, target=v, ), classes="", group="edges")
            cyto_edges.append(ce)
        return cyto_nodes + cyto_edges

    @property
    def starting_smis(self) -> list[str]:
        g = self.to_nx()
        return [n for n in g.nodes if g.in_degree(n) == 0 and ">>" not in n]

    @property
    def intermediate_product_smis(self) -> list[str]:
        g = self.to_nx()
        return [n for n in g.nodes if g.in_degree(n) > 0 and g.out_degree(n) > 0 and ">>" not in n]

    @property
    def intermediate_reaction_smis(self) -> list[str]:
        # an intermediate reaction is one whose product will be used in another reaction
        # this means an additional loading_storage transform
        g = self.to_nx()
        reaction_nodes = [n for n in g.nodes if ">>" in n]

        intermediate_reaction_smis = []
        for reaction_smi in reaction_nodes:
            product_smi = self.reaction_dict[reaction_smi].product_smi
            if g.out_degree(product_smi) > 0:
                intermediate_reaction_smis.append(reaction_smi)
        return intermediate_reaction_smis

    def get_reaction_precedence(self) -> dict[str, list[str]]:
        precedence_dict = defaultdict(list)
        g = self.to_nx()
        reaction_nodes = [n for n in g.nodes if ">>" in n]
        for i in reaction_nodes:
            for j in reaction_nodes:
                # global precedence
                # TODO this seems an overkill, dfs may suffice
                if nx.has_path(g, j, i) and i != j:
                    precedence_dict[i].append(j)
        return precedence_dict
