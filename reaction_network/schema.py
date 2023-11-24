from __future__ import annotations

import json
import re

import networkx as nx
from loguru import logger
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


class CompoundLv1(BaseModel):
    amount: float

    amount_unit: str

    moles: float

    compound_lv0: CompoundLv0

    @classmethod
    def from_moles(cls, moles: float, compound_lv0: CompoundLv0):
        mass = compound_lv0.molecular_weight * moles
        if compound_lv0.state_of_matter == 'SOLID':
            return cls(amount=mass, amount_unit='GRAM', moles=moles, compound_lv0=compound_lv0)
        else:
            return cls(amount=mass / compound_lv0.density, amount_unit='MILLILITER', moles=moles,
                       compound_lv0=compound_lv0)

    def __add__(self, other: CompoundLv1):
        assert self.compound_lv0.smiles == other.compound_lv0.smiles
        return CompoundLv1.from_moles(self.moles + other.moles, self.compound_lv0)


class ReactionLv1(BaseModel):
    batch_size: float

    expected_yield: float

    reaction_lv0: ReactionLv0

    reactants: list[CompoundLv1]

    reagents: list[CompoundLv1]

    product: CompoundLv1

    @property
    def volume_dict(self) -> dict[str, float]:
        vd = dict()
        for r in self.reagents + self.reactants:
            if r.compound_lv0.state_of_matter == "SOLID":
                v = r.amount / r.compound_lv0.density
            else:
                v = r.amount
            vd[r.compound_lv0.smiles] = v
        return vd

    @property
    def solvent_smi(self):
        return sorted(self.volume_dict.keys(), key=lambda x: self.volume_dict[x])[-1]

    @property
    def volume_sum(self) -> float:
        return sum(self.volume_dict.values())

    @staticmethod
    def calculate_batch_size(product_moles: float, expected_yield: float):
        """
        batch size is the "moles" of this reaction assuming the stoichiometry coefficient of the product is 1
        batch size * reactant stoichiometry coefficient = required reactant moles
        """
        return product_moles / expected_yield

    @staticmethod
    def calculate_required_moles(batch_size: float, reaction_lv0: ReactionLv0) -> dict[str, float]:
        # TODO dead volume
        data = dict()
        for smis_and_ratios in (
                reaction_lv0.reactant_smis_and_ratios,
                reaction_lv0.reagent_smis_and_ratios,
        ):
            for smi, ratio in smis_and_ratios.items():
                moles = batch_size * ratio
                data[smi] = moles
        return data


class NetworkLv1(BaseModel):
    reactions: list[ReactionLv1]
    compounds: list[CompoundLv1]
    network_lv0: NetworkLv0

    @property
    def compound_dict(self) -> dict[str, CompoundLv1]:
        return {r.compound_lv0.smiles: r for r in self.compounds}

    @property
    def reaction_dict(self) -> dict[str, ReactionLv1]:
        return {r.reaction_lv0.reaction_smiles: r for r in self.reactions}

    @property
    def summary(self):
        # TODO more lv1 info
        return self.network_lv0.summary

    @staticmethod
    def calculate_required_moles_and_batches(network_lv0: NetworkLv0, target_masses: dict[str, float],
                                             expected_yields: dict[str, float]):

        for r in network_lv0.reactions:
            if r.reaction_smiles not in expected_yields:
                logger.warning(f"yield not specified, auto specify 100% yield to reaction: {r.reaction_smiles}")
                expected_yields[r.reaction_smiles] = 1.0

        required_moles = {k: v / network_lv0.compound_dict[k].molecular_weight for k, v in target_masses.items()}
        required_batches = dict()
        reactions_lv1 = []

        # the following method assumes a product can only be made from one reaction, i.e. no materials merge
        g = network_lv0.to_nx()
        product_nodes = [n for n in g.nodes if g.in_degree(n) > 0 and ">>" not in n]
        target_nodes = [n for n in g.nodes if g.in_degree(n) > 0 and g.out_degree(n) == 0 and ">>" not in n]
        assert all(g.in_degree(n) == 1 for n in product_nodes)
        for n in target_nodes:
            if n not in target_masses:
                raise RuntimeError(f"product mass not specified for: {n}")
        assert set(target_nodes) == set(target_masses.keys())

        # assign moles using the reversed graph with a dummy root (so we can do bfs)
        # TODO IMPORTANT strictly, the aforementioned graph is not a tree, need more testing
        g = g.reverse(copy=True)

        dummy_root = "DUMMY_ROOT"
        for target_smi in target_masses:
            g.add_edge(dummy_root, target_smi)
        bfs_edges = nx.bfs_edges(g, dummy_root)
        bfs_nodes = [dummy_root] + [v for u, v in bfs_edges]
        for n in bfs_nodes:
            if n == dummy_root:
                continue
            if ">>" not in n:
                continue
            reaction_smi = n
            reaction_lv0 = network_lv0.reaction_dict[reaction_smi]
            assert reaction_lv0.product_smi in required_moles
            batch_size = ReactionLv1.calculate_batch_size(required_moles[reaction_lv0.product_smi],
                                                          expected_yields[reaction_smi])
            required_batches[reaction_smi] = batch_size
            required_moles_for_this_reaction = ReactionLv1.calculate_required_moles(batch_size, reaction_lv0)

            reagents_lv1 = []
            reactants_lv1 = []
            for compound_smi in reaction_lv0.reactant_smis_and_ratios:
                c_lv1 = CompoundLv1.from_moles(required_moles_for_this_reaction[compound_smi],
                                               network_lv0.compound_dict[compound_smi])
                reactants_lv1.append(c_lv1)
            for compound_smi in reaction_lv0.reagent_smis_and_ratios:
                c_lv1 = CompoundLv1.from_moles(required_moles_for_this_reaction[compound_smi],
                                               network_lv0.compound_dict[compound_smi])
                reagents_lv1.append(c_lv1)
            product_lv1 = CompoundLv1.from_moles(required_moles[reaction_lv0.product_smi],
                                                 network_lv0.compound_dict[reaction_lv0.product_smi])
            reaction_lv1 = ReactionLv1(
                batch_size=batch_size,
                expected_yield=expected_yields[reaction_smi],
                reaction_lv0=reaction_lv0,
                reactants=reactants_lv1,
                reagents=reagents_lv1,
                product=product_lv1
            )
            reactions_lv1.append(reaction_lv1)
            for reactant_smi, reactant_smi_moles in required_moles_for_this_reaction.items():
                try:
                    required_moles[reactant_smi] += reactant_smi_moles
                except KeyError:
                    required_moles[reactant_smi] = reactant_smi_moles
        return required_batches, required_moles, reactions_lv1

    @classmethod
    def from_target_masses_and_expected_yields(cls, target_masses: dict[str, float], expected_yields: dict[str, float],
                                               network_lv0: NetworkLv0):
        required_batches, required_moles, reactions_lv1 = NetworkLv1.calculate_required_moles_and_batches(
            network_lv0, target_masses, expected_yields
        )

        cs_lv1 = []
        for compound in network_lv0.compounds:
            c_lv1 = CompoundLv1.from_moles(required_moles[compound.smiles], compound)
            cs_lv1.append(c_lv1)
        return cls(reactions=reactions_lv1, compounds=cs_lv1, network_lv0=network_lv0)

    def to_nx(self) -> nx.DiGraph:
        # TODO move this part to pydantic validation
        assert len(self.reactions) == len(self.network_lv0.reactions)
        assert len(self.compounds) == len(self.network_lv0.compounds)
        # TODO also check if `compounds_by_reaction` has more or equal items than `self.compounds`

        return self.network_lv0.to_nx()

    def to_cyto_elements(self):
        # TODO both this and `to_nx` can be realized using lv0 methods with duck-typing
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
