from __future__ import annotations

import json
import re

import networkx as nx
from pandas._typing import FilePath
from pydantic import BaseModel
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles

from reaction_network.utils import query_askcos_condition_rec, drawing_url
from reaction_network.visualization import CytoNodeData, CytoEdge, CytoNode, CytoEdgeData


class ReactionLv0(BaseModel):
    reaction_smiles: str

    reactant_smis_and_ratios: dict[str, float]

    reagent_smis_and_ratios: dict[str, float]

    temperature: float

    @property
    def smis_and_ratios(self) -> dict[str, float]:
        assert not set(self.reagent_smis_and_ratios.keys()).intersection(self.reactant_smis_and_ratios.keys())
        d = dict()
        d.update(self.reactant_smis_and_ratios)
        d.update(self.reagent_smis_and_ratios)
        d.update({self.product_smi: - 1.0})
        return d

    @property
    def product_smi(self):
        s = self.reaction_smiles.split(">>")[-1]
        assert "." not in s
        return s

    @property
    def unique_molecular_smis(self):
        smis = [self.product_smi]
        for smi in self.reactant_smis_and_ratios:
            smis.append(smi)
        for smi in self.reagent_smis_and_ratios:
            smis.append(smi)
        return sorted(set(smis))

    @classmethod
    def from_reaction_smiles(cls, reaction_smiles: str):
        response = query_askcos_condition_rec(reaction_smiles)
        result = response['result'][0]
        return cls(
            reaction_smiles=reaction_smiles,
            reactant_smis_and_ratios=result['reactants'],
            reagent_smis_and_ratios=result['reagents'],
            temperature=result['temperature'],
        )


class CompoundLv0(BaseModel):
    smiles: str

    state_of_matter: str

    molecular_weight: float  # g/mol

    density: float  # g/mL

    @classmethod
    def from_smiles(cls, smiles, scraper_output: FilePath = None):
        if scraper_output is None:
            return CompoundLv0.get_default_compound_lv0(smiles)
        else:
            output_data = CompoundLv0.parse_scraper_output(scraper_output)
            if smiles not in output_data:
                return CompoundLv0.get_default_compound_lv0(smiles)
            form, mw, density = output_data[smiles]
            return CompoundLv0(smiles=smiles, state_of_matter=form, molecular_weight=mw, density=density)

    @staticmethod
    def get_default_compound_lv0(smiles: str):
        mw = Descriptors.MolWt(MolFromSmiles(smiles))
        if mw < 200 and all(m not in smiles for m in ['Li', 'Na', 'K', 'Mg', 'Ca', 'Pd']):
            form = "LIQUID"
            density = 1.0
        else:
            form = "SOLID"
            density = 1.4
        return CompoundLv0(smiles=smiles, state_of_matter=form, molecular_weight=mw, density=density)

    @staticmethod
    def parse_scraper_output(scraper_output: FilePath) -> dict:
        with open(scraper_output, "r") as f:
            output = json.load(f)
        output_data = dict()
        for smi, data in output.items():
            mw = Descriptors.MolWt(MolFromSmiles(smi))
            try:
                form_string = data['from']
            except KeyError:
                # TODO has metalloid then solid
                if mw < 200 and all(m not in smi for m in ['Li', 'Na', 'K', 'Mg', 'Ca', 'Pd']):
                    form_string = "liquid"
                else:
                    form_string = "solid"
            if "liquid" in form_string.lower():
                form = "LIQUID"
            else:
                form = "SOLID"
            try:
                density_string = data['density']
            except KeyError:
                if form == "LIQUID":
                    density_string = "1.0 g/mL"
                else:
                    density_string = "1.4 g/mL"
            density_string = re.findall("\d+\.\d+\s*g\/mL", density_string)[0]
            density_string = density_string.replace("g/mL", "").strip()
            density = float(density_string)
            output_data[smi] = (form, mw, density)
        return output_data


class NetworkLv0(BaseModel):
    reactions: list[ReactionLv0]
    n_targets: int
    source_file: str
    seed: int
    compounds: list[CompoundLv0]

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
            "number of unique starting materials": len([n for n in g.nodes if g.in_degree(n) == 0]),
            "number of unique target materials": len([n for n in g.nodes if g.out_degree(n) == 0]),
        }

    @classmethod
    def from_files(cls, scraper_input: FilePath, scraper_output: FilePath = None):
        with open(scraper_input, "r") as f:
            input_data = json.load(f)
            network = cls(**input_data)
        if scraper_output is not None:
            network.populate_compounds(scraper_output)
        return network

    @property
    def unique_molecular_smis(self) -> list[str]:
        smis = []
        for r in self.reactions:
            smis += r.unique_molecular_smis
        return sorted(set(smis))

    def populate_compounds(self, scraper_output: FilePath):
        for smi in self.unique_molecular_smis:
            c = CompoundLv0.from_smiles(smi, scraper_output=scraper_output)
            self.compounds.append(c)

    @property
    def product_smis(self) -> list[str]:
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
            for smi in reaction.reactant_smis_and_ratios:
                g.add_edge(smi, reaction.reaction_smiles)
            for smi in reaction.reagent_smis_and_ratios:
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
