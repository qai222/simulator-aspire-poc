from __future__ import annotations

import networkx as nx
from loguru import logger
from pydantic import BaseModel, Field

from enum import Enum
from hardware_pydantic.utils import str_uuid
from reaction_network.schema.lv1 import CompoundLv1, ReactionLv1, NetworkLv1
from reaction_network.visualization import CytoNodeData, CytoEdge, CytoNode, CytoEdgeData

"""
lv2 specifications, given a lv1 reaction
- unit operations
- abstract bench top
- contained materials
"""


class ContainerLv2(BaseModel):
    # TODO if there are multiple batches it may be better to define an abstract container
    #  that can be mapped to multiple physical vessels (that share the same purpose)
    id: str = Field(default_factory=str_uuid)
    made_of: str = "GLASS"
    type: str = "VIAL"
    volume_capacity: float = 50.0  # in mL

    # starting_material_smi: str | None = None

    def model_post_init(self, *args) -> None:
        assert self.id not in BENCH_TOP_LV2.container_dict
        BENCH_TOP_LV2.container_dict[self.id] = self

    def __eq__(self, other: MaterialLv2):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class MaterialLv2(BaseModel):
    id: str = Field(default_factory=str_uuid)
    compounds: list[CompoundLv1]
    contained_by: str
    impure: bool = False

    def model_post_init(self, *args) -> None:
        assert self.contained_by in BENCH_TOP_LV2.container_dict
        assert self.id not in BENCH_TOP_LV2.material_dict
        BENCH_TOP_LV2.material_dict[self.id] = self

    @property
    def volume(self):
        return sum([c.volume for c in self.compounds])

    @property
    def is_solid(self):
        return all(c.compound_lv0.state_of_matter == 'SOLID' for c in self.compounds)

    def duplicate(self):
        return MaterialLv2(**{k: v for k, v in self.model_dump().items() if k != "id"})

    def __eq__(self, other: MaterialLv2):
        # TODO we may want to check identity in a "chemical" way
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class OperationType(str, Enum):
    OPERATION_ADDITION_SOLID = "OPERATION_ADDITION_SOLID"
    OPERATION_ADDITION_LIQUID = "OPERATION_ADDITION_LIQUID"

    OPERATION_HEATING = "OPERATION_HEATING"

    OPERATION_PURIFICATION = "OPERATION_PURIFICATION"

    OPERATION_LOADING = "OPERATION_LOADING"
    OPERATION_RELOADING = "OPERATION_RELOADING"
    OPERATION_UNLOADING = "OPERATION_UNLOADING"

    OPERATION_UNDEFINED = "OPERATION_UNDEFINED"


class Operation(BaseModel):
    id: str = Field(default_factory=str_uuid)
    description: str = ""

    consumes: str | None = None
    produces: str | None = None

    precedents: list[str] = []

    type: OperationType = OperationType.OPERATION_UNDEFINED

    can_be_realized_by: list[str] = []

    def cyto_dump(self):
        # what I would call a semantic dump...
        d = self.model_dump()
        if self.produces is not None:
            d['produces'] = BENCH_TOP_LV2.material_dict[self.produces].model_dump()
        if self.consumes is not None:
            d['consumes'] = BENCH_TOP_LV2.material_dict[self.consumes].model_dump()
        return d

    @staticmethod
    def find_root_operations(operation_list: list[Operation]) -> list[Operation]:
        operation_dict = {t.id: t for t in operation_list}
        g = Operation.get_operation_graph(operation_list)
        return [operation_dict[n] for n in g.nodes if g.in_degree(n) == 0 and n in operation_dict]

    @staticmethod
    def get_operation_graph(operation_list: list[Operation]):
        # TODO contraction based on mapped device so we have fewer vars/consts in mip
        g = nx.DiGraph()
        for t in operation_list:
            for precedent in t.precedents:
                g.add_edge(precedent, t.id)
        return g

    def model_post_init(self, *args) -> None:
        assert self.id not in BENCH_TOP_LV2.operation_dict
        BENCH_TOP_LV2.operation_dict[self.id] = self

    @property
    def associated_container_ids(self) -> list[str]:
        ids = []
        for c in [self.consumes, self.produces]:
            if c is not None:
                i = BENCH_TOP_LV2.material_dict[c].contained_by
                ids.append(i)
        return ids

    @staticmethod
    def chain_operations(operations: list[Operation]):
        if len(operations) < 2:
            return
        for i in range(1, len(operations)):
            p = operations[i - 1]
            s = operations[i]
            s.precedents.append(p.id)

    @staticmethod
    def build_addition_operation(compounds: list[CompoundLv1], from_container: ContainerLv2,
                                 to_container: ContainerLv2):
        mat_from = MaterialLv2(compounds=compounds, contained_by=from_container.id)
        mat_to = MaterialLv2(compounds=compounds, contained_by=to_container.id)
        if all(c.compound_lv0.state_of_matter == "SOLID" for c in compounds):
            t = OperationType.OPERATION_ADDITION_SOLID
        else:
            t = OperationType.OPERATION_ADDITION_LIQUID
        return Operation(consumes=mat_from.id, produces=mat_to.id, type=t)

    @staticmethod
    def build_transforms_from_reaction(reaction_lv1: ReactionLv1,
                                       loading_starting_materials_transform_dict: dict[str, Operation],
                                       is_intermediate: bool) -> list[Operation]:
        # TODO we made the following assumptions:
        #  1. materials transfer happens at 100% efficiency (no lost, no dead volume)
        #  2. all vials are 50 mL mrv (so we don't need to switch plates and balance nests)
        #  3. assuming one reactor is enough, otherwise we need to determine sub batches here
        #  4. assuming we can safely mix solids with infinite delays, otherwise we need to first prepare solutions
        #  5. assuming the transforms constitute a path graph from loading storage to purification

        reactor = ContainerLv2(id=f"reactor for the reaction of: {reaction_lv1.reaction_lv0.reaction_smiles}")

        solid_additions = []
        for solid_compound in reaction_lv1.solids:
            loading_storage = loading_starting_materials_transform_dict[solid_compound.compound_lv0.smiles]
            stored_material = BENCH_TOP_LV2.material_dict[loading_storage.produces]
            storage_container_id = stored_material.contained_by
            storage_container = BENCH_TOP_LV2.container_dict[storage_container_id]
            addition = Operation.build_addition_operation([solid_compound], from_container=storage_container,
                                                          to_container=reactor)
            addition.precedents.append(loading_storage.id)
            solid_additions.append(addition)

        liquid_additions = []
        # TODO DRY this
        for liquid_compound in reaction_lv1.liquids:
            loading_storage = loading_starting_materials_transform_dict[liquid_compound.compound_lv0.smiles]
            stored_material = BENCH_TOP_LV2.material_dict[loading_storage.produces]
            storage_container_id = stored_material.contained_by
            storage_container = BENCH_TOP_LV2.container_dict[storage_container_id]
            addition = Operation.build_addition_operation([liquid_compound], from_container=storage_container,
                                                          to_container=reactor)
            addition.precedents.append(loading_storage.id)
            liquid_additions.append(addition)
        Operation.chain_operations(solid_additions + liquid_additions)

        # TODO there should be a vessel transport operation between these

        # actual reaction
        reaction_mixture = MaterialLv2(compounds=reaction_lv1.reactants + reaction_lv1.reagents,
                                       contained_by=reactor.id)
        raw_product = MaterialLv2(compounds=[reaction_lv1.product, ], contained_by=reactor.id, impure=True)
        if len(liquid_additions):
            precedent_of_this = liquid_additions[-1].id
        else:
            assert len(solid_additions)
            precedent_of_this = solid_additions[-1].id
        operation_reaction = Operation(
            consumes=reaction_mixture.id,
            produces=raw_product.id,
            precedents=[precedent_of_this],
            type=OperationType.OPERATION_HEATING,
        )

        operation_unloading = Operation(
            precedents=[operation_reaction.id],
            type=OperationType.OPERATION_UNLOADING,
        )

        # purification
        purification_container = ContainerLv2()
        pure_product = raw_product.duplicate()
        pure_product.impure = False
        pure_product.contained_by = purification_container.id
        operation_purification = Operation(
            consumes=raw_product.duplicate().id,
            produces=pure_product.id,
            precedents=[operation_unloading.id],
            type=OperationType.OPERATION_PURIFICATION,
        )

        # if this product will be used for another reaction, select a new container
        if is_intermediate:
            storage_container = ContainerLv2()
            pure_product_loaded = pure_product.duplicate()
            pure_product_loaded.contained_by = storage_container.id
            operation_reloading = Operation(
                consumes=pure_product.duplicate().id,
                produces=pure_product_loaded.id,
                precedents=[operation_purification.id],
                type=OperationType.OPERATION_RELOADING
            )
            loading_starting_materials_transform_dict[
                pure_product.compounds[0].compound_lv0.smiles] = operation_reloading
            return solid_additions + liquid_additions + [operation_reaction, operation_purification,
                                                         operation_reloading]
        else:
            return solid_additions + liquid_additions + [operation_reaction, operation_purification]

    @staticmethod
    def build_transforms_from_network(network_lv1: NetworkLv1):

        # define on-platform storage
        storage_dict = dict()
        for compound_smi, compound in network_lv1.compound_dict.items():
            storage = ContainerLv2(volume_capacity=50.0, type="on_platform_storage")
            storage_dict[compound_smi] = storage

        # define loading starting materials
        loading_starting_materials_transform_dict = dict()
        for compound_smi in network_lv1.network_lv0.starting_smis:
            compound = network_lv1.compound_dict[compound_smi]
            loading_transform = Operation(
                consumes=None,
                produces=MaterialLv2(compounds=[compound, ], contained_by=storage_dict[compound_smi].id).id,
                type=OperationType.OPERATION_LOADING
            )
            loading_starting_materials_transform_dict[compound_smi] = loading_transform
            # logger.info(f"added transform: {loading_transform}")

        # define reaction transforms
        reaction_precedence = network_lv1.network_lv0.get_reaction_precedence()
        transform_list_by_reaction = dict()
        for this_reaction in sorted(network_lv1.reactions,
                                    key=lambda x: len(reaction_precedence[x.reaction_lv0.reaction_smiles])):
            logger.info(
                f"create transforms for: {this_reaction.reaction_lv0.reaction_smiles}\n it has n_precedence: {len(reaction_precedence[this_reaction.reaction_lv0.reaction_smiles])}")

            transforms_of_this_reaction = Operation.build_transforms_from_reaction(
                this_reaction, loading_starting_materials_transform_dict,
                is_intermediate=this_reaction.reaction_lv0.product_smi in network_lv1.network_lv0.intermediate_product_smis
            )
            transform_list_by_reaction[this_reaction.reaction_lv0.reaction_smiles] = transforms_of_this_reaction

        for reaction_smi, preceding_reaction_smis in reaction_precedence.items():
            transforms_of_this_reaction = transform_list_by_reaction[reaction_smi]
            root_transforms = Operation.find_root_operations(transforms_of_this_reaction)
            for preceding_smi in preceding_reaction_smis:
                transforms_of_preceding_reaction = transform_list_by_reaction[preceding_smi]

                # TODO this relies on the assumption that reaction transforms are a path graph
                end_transform = transforms_of_preceding_reaction[-1]

                for t in root_transforms:
                    t.precedents.append(end_transform.id)
        return BENCH_TOP_LV2


class BenchTopLv2(BaseModel):
    container_dict: dict[str, ContainerLv2] = dict()
    material_dict: dict[str, MaterialLv2] = dict()
    operation_dict: dict[str, Operation] = dict()

    devices: list[str] = ["SOLID_DISPENSER", "LIQUID_DISPENSER", "HEATING_BLOCK_1", "HEATING_BLOCK_2",
                          "HEATING_BLOCK_3", "ROTAVAP_1", "ROTAVAP_2"]

    operation_type_device_mapping: dict[OperationType, list[str]] = {
        OperationType.OPERATION_HEATING: ["HEATER-1", "HEATER-2", "HEATER-3"],
        OperationType.OPERATION_LOADING: ["WORKER-1", "WORKER-2", "WORKER-3"],
        OperationType.OPERATION_UNLOADING: ["WORKER-1", "WORKER-2"],
        OperationType.OPERATION_RELOADING: ["WORKER-1", "WORKER-2", "WORKER-3"],
        OperationType.OPERATION_PURIFICATION: ["ACC-1", "ACC-2"],
        OperationType.OPERATION_ADDITION_SOLID: ["PREPARATION PLATFORM",],
        OperationType.OPERATION_ADDITION_LIQUID: ["PREPARATION PLATFORM",],
    }

    def add_operation_can_be_realized_by(self):
        for operation_id, operation in self.operation_dict.items():
            try:
                machines = self.operation_type_device_mapping[operation.type]
            except KeyError:
                machines = []
            operation.can_be_realized_by = machines

    @property
    def summary(self):
        return {
            "unique operations": len(self.operation_dict),
            "unique materials": len(self.material_dict),
        }

    def to_cyto_elements(self) -> list[CytoNode | CytoEdge]:

        g = Operation.get_operation_graph(list(self.operation_dict.values()))
        cyto_nodes = []
        cyto_edges = []
        for n in g.nodes:
            operation = self.operation_dict[n]
            classes = operation.type.value
            data = operation.cyto_dump()
            cnd = CytoNodeData(id=n, label=n, url="", data=data)
            cn = CytoNode(data=cnd, classes=classes, group="nodes")
            cyto_nodes.append(cn)
        for u, v in g.edges:
            ce = CytoEdge(data=CytoEdgeData(id=f"{u} {v}", source=u, target=v, ), classes="", group="edges")
            cyto_edges.append(ce)
        return cyto_nodes + cyto_edges


BENCH_TOP_LV2 = BenchTopLv2()


def clear_benchtop_lv2():
    BENCH_TOP_LV2 = BenchTopLv2()
    return BENCH_TOP_LV2
