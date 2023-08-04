from __future__ import annotations

from hardware_pydantic.lab_objects import ChemicalContainer, LabContainee, LabContainer
from hardware_pydantic.tecan.settings import TecanLayout, TecanLabObject


class TecanPlateWell(ChemicalContainer, LabContainee, TecanLabObject):
    """A well in a Tecan plate."""
    pass


class TecanPlate(LabContainer, LabContainee, TecanLabObject):
    """A Tecan plate."""

    @staticmethod
    def create_plate_with_empty_wells(
            n_wells: int = 2,
            plate_id: str = "TecanPlate1", plate_id_inherit: bool = True
    ) -> tuple[TecanPlate, list[TecanPlateWell]]:
        """Create a Tecan plate with empty wells.

        Parameters
        ----------
        n_wells : int, optional
            Number of wells. Default is 2.
        plate_id : str, optional
            Identifier of the plate. Default is "TecanPlate1".

        Returns
        -------
        tuple[TecanPlate, list[TecanPlateWell]]
            A Tecan plate and a list of wells.

        """

        plate = TecanPlate.from_capacity(
            can_contain=[TecanPlateWell.__name__, ],
            capacity=n_wells,
            container_id=plate_id
        )

        assert n_wells <= plate.slot_capacity, f"{n_wells} {plate.slot_capacity}"

        wells = []
        n_created = 0
        for k in plate.slot_content:
            if plate_id_inherit:
                w = TecanPlateWell(
                    identifier=f"well-{k} " + plate_id, contained_by=plate.identifier,
                    contained_in_slot=k,
                )
            else:
                w = TecanPlateWell(
                    contained_by=plate.identifier,
                    contained_in_slot=k,
                )
            plate.slot_content[k] = w.identifier
            wells.append(w)
            n_created += 1
            if n_created == n_wells:
                break
        return plate, wells


class TecanWashBay(TecanLabObject):
    """The Tecan wash bay for the washing needles.

    Attributes
    ----------
    layout : TecanLayout
        The layout of the wash bay.

    """
    layout: TecanLayout | None = None


class TecanLiquidTank(ChemicalContainer, TecanLabObject):
    """Tecan liquid tank.

    Attributes
    ----------
    layout : TecanLayout
        The layout of the liquid tank.

    """
    layout: TecanLayout | None = None


class TecanArm1Needle(ChemicalContainer, LabContainee, TecanLabObject):
    """Tecan arm 1 needle."""
    pass


class TecanHotel(LabContainer, TecanLabObject):
    """Tecan hotel.

    Attributes
    ----------
    can_contain : list[str]
        The list of objects that can be contained in the hotel.
    layout : TecanLayout
        The layout of the hotel.

    """
    can_contain: list[str] = TecanPlate.__name__
    layout: TecanLayout | None = None
