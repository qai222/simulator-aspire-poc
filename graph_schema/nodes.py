from __future__ import annotations

from typing import Type

from .abstract import Individual, ObjectProperty, MASTER_GRAPH


class Quality(Individual):
    name: str
    value: str | float | int | bool | None = None
    related_to: Individual | None = None


class Device(Individual):
    name: str
    description: str = ""


class Matter(Individual):
    amount: float
    unit: str


class IsRelatedTo(ObjectProperty):
    """ a quality can be related to an artifact """
    pass


class HasQuality(ObjectProperty):
    """ an artifact can have a quality """
    pass


class ContainerCanHold(ObjectProperty):
    """ a container can hold a  """
    pass


def get_neighbors(n: Individual, op: Type[ObjectProperty] | None = None) -> tuple[list[Individual], list[Individual]]:
    successors = []
    for _, v, d in MASTER_GRAPH.out_edges(n, data=True):
        if op is None or d['predicate'] == op.__name__:
            successors.append(v)
    predecessors = []
    for _, v, d in MASTER_GRAPH.in_edges(n, data=True):
        if op is None or d['predicate'] == op.__name__:
            predecessors.append(v)
    return predecessors, successors
