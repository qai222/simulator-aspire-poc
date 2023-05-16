from __future__ import annotations

from uuid import UUID, uuid4

import networkx as nx
from pydantic import BaseModel, Field

MASTER_GRAPH = nx.DiGraph()


class Individual(BaseModel):
    identifier: UUID = Field(default_factory=uuid4)

    # you need pydantic 2 for this
    def model_post_init(self, *args) -> None:
        MASTER_GRAPH.add_node(self)

    def __hash__(self):
        return hash(self.identifier)


class ObjectProperty(BaseModel):
    subject_individual: Individual

    object_individual: Individual

    def as_tuple(self):
        return self.subject_individual, self.__class__.__name__, self.object_individual

    def __hash__(self):
        return hash(self.as_tuple())

    # you need pydantic 2 for this
    def model_post_init(self, *args) -> None:
        MASTER_GRAPH.add_edge(self.subject_individual, self.object_individual, predicate=self.__class__.__name__)

# # not very useful rn
# class DataProperty(BaseModel):
#     value: str | float | None = None
#
#     unit: str | None = None
