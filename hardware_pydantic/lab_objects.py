from __future__ import annotations

from typing import Type

from hardware_pydantic.base import Lab, LabObject


class ChemicalContainer(LabObject):
    """
    a container that is designed to be in direct contact with (reaction-participating) chemicals

    subclass of [container](http://purl.allotrope.org/ontologies/equipment#AFE_0000407)
    """

    volume_capacity: float = 40

    material: str = "GLASS"

    chemical_content: dict[str, float] = dict()
    """ what is inside now? """

    @property
    def content_sum(self) -> float:
        if len(self.chemical_content) == 0:
            return 0
        return sum(self.chemical_content.values())

    def add_content(self, content: dict[str, float]):
        for k, v in content.items():
            if k not in self.chemical_content:
                self.chemical_content[k] = content[k]
            else:
                self.chemical_content[k] += content[k]

    def remove_content(self, amount: float) -> dict[str, float]:
        # by default the content is homogeneous liquid
        pct = amount / self.content_sum
        removed = dict()
        for k in self.chemical_content:
            removed[k] = self.chemical_content[k] * pct
            self.chemical_content[k] -= removed[k]
        return removed


class LabContainer(LabObject):
    """
    a container designed to hold other LabObject instances, such as a vial plate

    it should have a finite, fixed number of slots

    subclass of [container](http://purl.allotrope.org/ontologies/equipment#AFE_0000407)
    """

    can_contain: list[str]
    """ the class names of the thing it can hold """
    # TODO validation

    slot_content: dict[str, str | None] = dict(SLOT=None)
    """ dict[<slot identifier>, <object identifier>] """

    @property
    def slot_capacity(self):
        return len(self.slot_content)

    @classmethod
    def from_capacity(cls, can_contain: list[str], capacity: int = 16, container_id: str = None, ) -> LabContainer:
        content = {str(i + 1): None for i in range(capacity)}
        if container_id is None:
            return cls(slot_content=content, can_contain=can_contain)
        else:
            return cls(slot_content=content, identifier=container_id, can_contain=can_contain)

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


class LabContainee(LabObject):
    """
    lab objects that can be held by another lab container

    related to [containing](http://purl.allotrope.org/ontologies/process#AFP_0003623)
    """

    contained_by: str | None = None

    contained_in_slot: str | None = "SLOT"

    @staticmethod
    def move(containee: LabContainee, dest_container: LabContainer, lab: Lab, dest_slot: str = "SLOT"):
        if containee.contained_by is not None:
            source_container = lab[containee.contained_by]
            source_container: LabContainer
            assert source_container.slot_content[containee.contained_in_slot] == containee.identifier
            source_container.slot_content[containee.contained_in_slot] = None
        assert dest_container.slot_content[dest_slot] is None
        dest_container.slot_content[dest_slot] = containee.identifier
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
