from __future__ import annotations

from typing import Type

from hardware_pydantic.base import Lab, LabObject


"""The lab objects that cannot perform actions themselves, but can be acted upon by other lab 
objects."""


class ChemicalContainer(LabObject):
    """
    A container that is designed to be in direct contact with (reaction-participating) chemicals.

    Att
    ----------
    volume_capacity : float
        The maximum volume of the container in mL. Default is 40.
    material : str
        The material of the container. One of "GLASS", "PLASTIC", "METAL", "OTHER". Default is
        "GLASS".
    chemical_content : dict[str, float]
        The chemical content of the container. A dictionary of chemical identifiers and their
        amounts. Default is an empty dictionary.

    Notes
    -----
    This is a subclass of [container](http://purl.allotrope.org/ontologies/equipment#AFE_0000407).

    """

    volume_capacity: float = 40

    material: str = "GLASS"

    # what is inside now
    chemical_content: dict[str, float] = dict()

    @property
    def content_sum(self) -> float:
        """The sum of the amounts of all chemicals in the container.

        Returns
        -------
        float
            The sum of the amounts of all chemicals in the container.
        """
        if len(self.chemical_content) == 0:
            return 0
        return sum(self.chemical_content.values())

    def add_content(self, content: dict[str, float]):
        """Add chemicals to the container.

        Parameters
        ----------
        content : dict[str, float]
            A dictionary of chemical identifiers and their amounts.

        """
        for k, v in content.items():
            if k not in self.chemical_content:
                self.chemical_content[k] = content[k]
            else:
                self.chemical_content[k] += content[k]

    def remove_content(self, amount: float) -> dict[str, float]:
        """Remove chemicals from the container.

        Parameters
        ----------
        amount : float
            The amount of chemicals to remove from the container.

        Returns
        -------
        dict[str, float]
            A dictionary of chemical identifiers and their amounts that were removed.
        """
        # by default the content is homogeneous liquid
        pct = amount / self.content_sum
        removed = dict()
        for k in self.chemical_content:
            removed[k] = self.chemical_content[k] * pct
            self.chemical_content[k] -= removed[k]
        return removed


class LabContainer(LabObject):
    """A container designed to hold other LabObject instances, such as a vial plate.

    Attributes
    ----------
    can_contain : list[str]
        The class names of the thing it can hold.
    slot_content : dict[str, str | None]
        A dictionary of slot identifiers and the object identifiers of the objects in the slots.

    Notes
    -----
    It should have a finite, fixed number of slots. It is a subclass of
    [container](http://purl.allotrope.org/ontologies/equipment#AFE_0000407).

    """

    can_contain: list[str]
    # TODO validation
    slot_content: dict[str, str | None] = dict(SLOT=None)

    @property
    def slot_capacity(self):
        """The number of slots in the container.

        Returns
        -------
        int
            The number of slots in the container.
        """
        return len(self.slot_content)

    @classmethod
    def from_capacity(cls,
                      can_contain: list[str],
                      capacity: int = 16,
                      container_id: str = None,
                      **kwargs,
                      ) -> LabContainer:
        """
        Create a `LabContainer` instance with given capacity, container_id, and can_contain.

        Parameters
        ----------
        can_contain : list[str]
            The class names of the thing it can hold.
        capacity : int
            The number of slots in the container. Default is 16.
        container_id : str
            The identifier of the container. Default is None.

        Returns
        -------
        LabContainer
            A `LabContainer` instance with given capacity, container_id, and can_contain.
        """
        content = {str(i + 1): None for i in range(capacity)}
        if container_id is None:
            return cls(slot_content=content,
                       can_contain=can_contain,
                       **kwargs)
        else:
            return cls(slot_content=content,
                       identifier=container_id,
                       can_contain=can_contain,
                       **kwargs
                       )

    @staticmethod
    def get_all_containees(container: LabContainer, lab: Lab) -> list[str]:
        """Get all the containees of a container.

        Parameters
        ----------
        container : LabContainer
            The container to get the containees from.
        lab : Lab
            The lab that contains the container.

        Returns
        -------
        list[str]
            A list of identifiers of the containees of the container.

        Notes
        -----
        If we are requesting a container as a resource, we should always also request its containees
        ex. if an arm is trying to move a plate while another arm is aspirating liquid from a vial
        on this plate,
        the former arm should wait for the vial until it is released by the aspiration action.
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
    Lab objects that can be held by another lab container

    Attributes
    ----------
    contained_by : str | None
        The identifier of the container that contains this object. Default is None.
    contained_in_slot : str | None
        The identifier of the slot in the container that contains this object. Default is 'SLOT'.

    Notes
    -----
    This is related to [containing](http://purl.allotrope.org/ontologies/process#AFP_0003623)
    """

    contained_by: str | None = None

    contained_in_slot: str | None = "SLOT"

    @staticmethod
    def move(containee: LabContainee,
             dest_container: LabContainer,
             lab: Lab,
             dest_slot: str = "SLOT"):
        """Move a containee to a new container.

        Parameters
        ----------
        containee : LabContainee
            The containee to move.
        dest_container : LabContainer
            The container to move the containee to.
        lab : Lab
            The lab that contains the containee and the destination container.
        dest_slot : str
            The slot in the destination container to move the containee to. Default is 'SLOT'.
        """
        if containee.contained_by is not None:
            source_container = lab[containee.contained_by]
            source_container: LabContainer
            assert source_container.slot_content[containee.contained_in_slot] == \
                   containee.identifier
            source_container.slot_content[containee.contained_in_slot] = None
        assert dest_container.slot_content[dest_slot] is None
        dest_container.slot_content[dest_slot] = containee.identifier
        containee.contained_by = dest_container.identifier
        containee.contained_in_slot = dest_slot

    @staticmethod
    def get_container(containee: LabContainee, lab: Lab, upto: Type = None) -> LabContainer | None:
        """
        Get the container that contains the containee.

        Parameters
        ----------
        containee : LabContainee
            The containee to get the container from.
        lab : Lab
            The lab that contains the containee.
        upto : Type
            The type of the container to get. Default is None.

        Returns
        -------
        LabContainer | None
            The container that contains the containee.

        """
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
