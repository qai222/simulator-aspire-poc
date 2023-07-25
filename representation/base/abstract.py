from __future__ import annotations

import pprint
from typing import Any, Type

import networkx as nx
from deepdiff import DeepDiff
from loguru import logger
from pydantic import BaseModel, Field

from hardware_pydantic.utils import str_uuid

# TODO about graph field see https://github.com/pydantic/pydantic/issues/1763
# TODO should we include a field describing how confident are we for an Individual/ObjectProperty?
# TODO make world global a good idea?

WORLD = nx.MultiDiGraph()


class Individual(BaseModel):
    # unique identifier for this individual
    identifier: str = Field(default_factory=str_uuid)

    # you need pydantic 2 for this
    def model_post_init(self, *args) -> None:
        if self.identifier in WORLD.nodes:
            existing_individual = WORLD.nodes[self.identifier]['individual']

            diff = DeepDiff(self.model_dump(), existing_individual.model_dump())
            diff_dict = diff.to_dict()
            msg = f"Found existing individual: class=={self.__class__.__name__} identifier=={self.identifier}"
            if len(diff_dict) == 0:
                msg += "\n\t they are identical, no further actions"
            else:
                msg += f"\n\t they are different, update existing nodes:\n{pprint.pformat(diff_dict)}"
                nx.set_node_attributes(WORLD, {self.identifier: self}, "individual")
            logger.warning(msg)
        else:
            WORLD.add_node(self.identifier, individual=self)

        for k in self.model_fields_set:
            if k in ("identifier", ):
                continue
            v = getattr(self, k)
            if isinstance(v, Individual):
                ObjectProperty(predicate=k, subject_identifier=self.identifier, object_identifier=v.identifier)

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other: Individual):
        return self.identifier == other.identifier

    class Config:
        arbitrary_types_allowed = True


class ObjectProperty(BaseModel):
    identifier: str = Field(default_factory=str_uuid)

    predicate: str

    # subject of the triple
    subject_identifier: str

    # object of the triple
    object_identifier: str

    def as_tuple(self):
        return self.subject_identifier, self.predicate, self.object_identifier

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other: ObjectProperty):
        return self.__hash__() == other.__hash__()

    # you need pydantic 2 for this
    def model_post_init(self, *args) -> None:
        add_edge = False
        if (self.subject_identifier, self.object_identifier) not in WORLD.edges:
            add_edge = True
        else:
            existing_object_properties = []
            for u, v, k, d in WORLD.edges(nbunch=self.subject_identifier, keys=True, data=True):
                if v != self.object_identifier:
                    continue
                existing_object_property = d['object_property']
                existing_object_properties.append(existing_object_property)
            if self not in existing_object_properties:
                add_edge = True
        if add_edge:
            WORLD.add_edge(
                self.subject_identifier, self.object_identifier,
                object_property=self,
                # subject=self.subject_individual, object=self.object_individual, predicate=self.__class__.__name__
            )


def find_individuals(individual_class: Type = None, data_properties: dict[str, Any] = None):
    individuals = []
    for n, d in WORLD.nodes(data=True):
        individual = d['individual']
        individual: Individual
        accept_class = False
        accept_data_properties = False
        if individual_class is None or individual.__class__ == individual_class:
            accept_class = True
        if data_properties is None:
            accept_data_properties = True
        elif set(data_properties.keys()).issubset(individual.model_fields_set) and all(
                individual.model_dump()[k] == v for k, v in data_properties.items()):
            accept_data_properties = True
        if accept_class and accept_data_properties:
            individuals.append(individual)
    return individuals


def find_object_properties(
        predicate: str = None,
        subject_id: str = None,
        object_id: str = None,
) -> list[ObjectProperty]:
    ops = []
    for u, v, k, d in WORLD.edges(data=True, keys=True):
        op = d['object_property']
        op: ObjectProperty
        p_match = False
        s_match = False
        o_match = False
        if predicate is None or op.__class__.__name__ == predicate:
            p_match = True
        if subject_id is None or subject_id == op.subject_identifier:
            s_match = True
        if object_id is None or object_id == op.object_identifier:
            o_match = True
        if p_match and s_match and o_match:
            ops.append(op)
    return ops
