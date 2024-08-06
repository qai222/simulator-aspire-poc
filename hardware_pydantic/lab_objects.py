from __future__ import annotations

from typing import Type, Optional

from hardware_pydantic.base import Lab, LabObject, JuniorOntology, Individual, Material
from twa.data_model.base_ontology import BaseClass, ObjectProperty, DatatypeProperty


class Volume_capacity(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology

class Chemical_name(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology

class Amount(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology

class ChemicalContent(Individual):
    rdfs_isDefinedBy = JuniorOntology
    chemical_name: Chemical_name[str]
    amount: Amount[float]

    def __init__(self, **data):
        # NOTE this is a workaround to make sure the value is casted to float
        if 'amount' in data:
            data['amount'] = float(data['amount'])
        super().__init__(**data)

class Has_chemical_content(ObjectProperty):
    rdfs_isDefinedBy = JuniorOntology

class ChemicalContainer(LabObject):
    """
    a container that is designed to be in direct contact with (reaction-participating) chemicals

    subclass of [container](http://purl.allotrope.org/ontologies/equipment#AFE_0000407)
    """

    volume_capacity: Volume_capacity[float] = 40.0

    material: Material[str] = "GLASS"

    has_chemical_content: Optional[Has_chemical_content[ChemicalContent]] = set()
    # chemical_content: dict[str, float] = dict()
    """ what is inside now? """

    def model_post_init(self, __context) -> None:
        # NOTE adding this as it seems to be necessary for other actually overwritten methods to work when multi-inheritance is used
        # i.e. JuniorLabObject and JuniorInstruction
        return super().model_post_init(__context)

    @property
    def chemical_content(self) -> dict[str, float]:
        return {list(c.chemical_name)[0]: list(c.amount)[0] for c in self.has_chemical_content}

    @chemical_content.setter
    def chemical_content(self, content: dict[str, float]):
        self.add_content(content)

    @property
    def content_sum(self) -> float:
        if len(self.chemical_content) == 0:
            return 0
        return sum(self.chemical_content.values())

    def add_content(self, content: dict[str, float]):
        for k, v in content.items():
            if k not in self.chemical_content:
                self.has_chemical_content.add(ChemicalContent(chemical_name=k, amount=v))
            else:
                self.get_content(k).amount = {list(self.get_content(k).amount)[0] + content[k]}

    def get_content(self, chemical_name: str) -> ChemicalContent | None:
        for c in self.has_chemical_content:
            if list(c.chemical_name)[0] == chemical_name:
                return c
        return None

    def remove_content(self, amount: float) -> dict[str, float]:
        # by default the content is homogeneous liquid
        pct = amount / self.content_sum
        removed = dict()
        for k in self.chemical_content:
            removed[k] = self.chemical_content[k] * pct
            self.get_content(k).amount = {list(self.get_content(k).amount)[0] - removed[k]}
        return removed


class Can_contain(ObjectProperty):
    rdfs_isDefinedBy = JuniorOntology


class Capacity(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology


class Has_slot_content(ObjectProperty):
    rdfs_isDefinedBy = JuniorOntology

class LabContainer(LabObject):
    """
    a container designed to hold other LabObject instances, such as a vial plate

    it should have a finite, fixed number of slots

    subclass of [container](http://purl.allotrope.org/ontologies/equipment#AFE_0000407)
    """

    def model_post_init(self, __context) -> None:
        # NOTE adding this as it seems to be necessary for other actually overwritten methods to work when multi-inheritance is used
        # i.e. JuniorLabObject and JuniorInstruction
        return super().model_post_init(__context)

    can_contain: Can_contain[LabObject]
    """ the class names of the thing it can hold """
    # TODO validation

    capacity: Optional[Capacity[int]] = None
    has_slot_content: Optional[Has_slot_content[LabObject]] = []

    @property
    def slot_content(self):
        """ dict[<slot identifier>, <object identifier>] """
        if len(self.has_slot_content) == 0:
            return {'SLOT': None}
        else:
            return {k.contained_in_slot: k.identifier for k in self.has_slot_content}

    @property
    def empty_slot_keys(self):
        lst = [str(k+1) for k in range(self.slot_capacity)]
        for k in self.has_slot_content:
            if k.contained_in_slot in lst:
                lst.remove(k.contained_in_slot)
            else:
                lst = lst[:-1]
        return lst

    @property
    def slot_capacity(self):
        return list(self.capacity)[0]

    @classmethod
    def from_capacity(cls, can_contain: list[str], capacity: int = 16, container_id: str = None, **kwargs) -> LabContainer:
        # content = {str(i + 1): None for i in range(capacity)}
        if container_id is None:
            return cls(capacity=capacity, can_contain=can_contain, **kwargs)
        else:
            return cls(capacity=capacity, identifier=container_id, can_contain=can_contain, **kwargs)

    @staticmethod
    def get_all_containees(container: LabContainer, lab: Lab) -> list[str]:
        """
        if we are requesting a container as a resource, we should always also request its containees
        ex. if an arm is trying to move a plate while another arm is aspirating liquid from a vial on this plate,
        the former arm should wait for the vial until it is released by the aspiration action
        """
        containees = []
        for containee in container.slot_content.values():
            if containee is None:
                continue
            containees.append(containee)
            if isinstance(lab[containee], LabContainer):
                containees += LabContainer.get_all_containees(lab[containee], lab)
        return containees


class Is_contained_by(ObjectProperty):
    rdfs_isDefinedBy = JuniorOntology
    owl_maxQualifiedCardinality = 1

class Is_contained_in_slot(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology
    owl_maxQualifiedCardinality = 1
class LabContainee(LabObject):
    """
    lab objects that can be held by another lab container

    related to [containing](http://purl.allotrope.org/ontologies/process#AFP_0003623)
    """
    def model_post_init(self, __context) -> None:
        # NOTE adding this as it seems to be necessary for other actually overwritten methods to work when multi-inheritance is used
        # i.e. JuniorLabObject and JuniorInstruction
        return super().model_post_init(__context)

    is_contained_by: Is_contained_by[LabObject] = set()
    is_contained_in_slot: Optional[Is_contained_in_slot[str]] = "SLOT"

    def __init__(self, **data):
        # NOTE this is a hack to avoid massive changes in the codebase
        if 'contained_by' in data:
            data['is_contained_by'] = {data.pop('contained_by')}
        if 'contained_in_slot' in data:
            data['is_contained_in_slot'] = {data.pop('contained_in_slot')}
        super().__init__(**data)

    @property
    def contained_by(self):
        if len(self.is_contained_by) == 0:
            return None
        elif len(self.is_contained_by) == 1:
            return list(self.is_contained_by)[0]
        else:
            raise Exception(f'Only one container is allowed for containee {self.instance_iri}, but multiple containers are found: {self.is_contained_by}')

    @contained_by.setter
    def contained_by(self, container_id: str):
        if container_id is not None:
            self.is_contained_by = {container_id}
        else:
            self.is_contained_by = set()

    @property
    def contained_in_slot(self):
        return list(self.is_contained_in_slot)[0]

    @contained_in_slot.setter
    def contained_in_slot(self, slot_id: str):
        if slot_id is not None:
            self.is_contained_in_slot = {slot_id}
        else:
            self.is_contained_in_slot = None

    @staticmethod
    def move(containee: LabContainee, dest_container: LabContainer, lab: Lab, dest_slot: str = "SLOT"):
        if containee.contained_by is not None:
            source_container = lab[containee.contained_by]
            source_container: LabContainer
            assert source_container.slot_content[containee.contained_in_slot] == containee.identifier
            source_container.has_slot_content.remove(containee)
        assert dest_container.slot_content[dest_slot] is None
        dest_container.has_slot_content.add(containee)
        containee.contained_by = dest_container.identifier
        containee.contained_in_slot = dest_slot

    @staticmethod
    def get_container(containee: LabContainee, lab: Lab, upto: Type = None) -> LabContainer | None:
        current_container_id = containee.contained_by
        if current_container_id is None:
            return containee
        else:
            current_container = lab[current_container_id]
            if upto is not None and isinstance(current_container, upto):
                return current_container
            elif isinstance(current_container, LabContainee):
                return LabContainee.get_container(current_container, lab)
            else:
                return current_container
