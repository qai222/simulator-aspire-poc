from __future__ import annotations

from typing import Optional

from hardware_pydantic.junior.settings import JUNIOR_LAB, JuniorLabObject, JUNIOR_VIAL_TYPE, JuniorLayout, Layout
from hardware_pydantic.lab_objects import ChemicalContainer, LabContainee, LabContainer, Can_contain
from hardware_pydantic.base import JuniorOntology, Material, LabObject

from twa.data_model.base_ontology import DatatypeProperty

""""Lab objects for the JUNIOR platform from NCATS."""

class Is_spinning(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology

class JuniorStirBar(LabContainee, JuniorLabObject):
    """A stir bar that can be placed in a vial.

    Parameters
    ----------
    material : str
        The material the stir bar is made of. Default is "TEFLON".
    is_spinning : bool
        Whether the stir bar is spinning. Default is False.

    """
    material: Material[str] = "TEFLON"

    is_spinning: Is_spinning[bool] = False


class JuniorVial(ChemicalContainer, LabContainee, LabContainer, JuniorLabObject):
    """A vial on the JUNIOR platform, usually placed in a rack, it can contain a stir bar.

    Parameters
    ----------
    vial_type : str
        The type of vial. Default is "HRV".
    can_contain : list[str]
        The list of objects that can be placed in the vial. Default is ["JuniorStirBar"].

    """

    can_contain: Can_contain[LabObject] = [JuniorStirBar.rdf_type, ]

    vial_type: JUNIOR_VIAL_TYPE = "HRV"


class JuniorPdpTip(ChemicalContainer, LabContainee, JuniorLabObject):
    """Positive displacement pipette tip.

    Parameters
    ----------
    material : str
        The material the tip is made of. Default is "PLASTIC".

    """
    material: Material[str] = "PLASTIC"


class JuniorRack(LabContainer, LabContainee, JuniorLabObject):
    """A rack used to hold vials or positive displacement pipette tips."""

    @staticmethod
    def create_rack_with_empty_tips(
            n_tips: int = 2,
            rack_capacity: int = 4,
            rack_id: str = "PdpTipRack1",
            tip_id_inherit: bool = True
    ) -> tuple[JuniorRack, list[JuniorPdpTip]]:
        """Create a rack with empty tips.

        Parameters
        ----------
        n_tips : int
            The number of tips to create. Default is 2.
        rack_capacity : int
            The capacity of the rack. Default is 4.
        rack_id : str
            The identifier of the rack. Default is "PdpTipRack1".
        tip_id_inherit : bool
            Whether the tip identifiers should be inherited from the rack. Default is True.

        Returns
        -------
        rack : JuniorRack
            The rack with the empty tips.
        tips : list[JuniorPdpTip]
            The list of empty tips.

        """
        rack = JuniorRack.from_capacity(
            can_contain=[JuniorPdpTip.rdf_type, ], capacity=rack_capacity, container_id=rack_id
        )

        assert n_tips <= rack.slot_capacity, f"{n_tips} {rack.slot_capacity}"

        tips = []
        n_created = 0
        for k in range(rack.slot_capacity):
            idx = str(k+1)
            if tip_id_inherit:
                v = JuniorPdpTip(
                    identifier=f"{rack_id}/PdpTip-{idx}",
                    contained_by=rack.identifier,
                    contained_in_slot=idx,
                )
            else:
                v = JuniorVial(
                    contained_by=rack.identifier, contained_in_slot=idx,
                )
            rack.has_slot_content.add(v)
            tips.append(v)
            n_created += 1
            if n_created == n_tips:
                break
        return rack, tips

    @staticmethod
    def create_rack_with_empty_vials(
            n_vials: int = 2,
            rack_capacity: int = 4,
            vial_type: JUNIOR_VIAL_TYPE = "HRV",
            rack_id: str = "VialRack1",
            vial_id_inherit: bool = True
    ) -> tuple[JuniorRack, list[JuniorVial]]:
        """Create a rack with empty vials.

        Parameters
        ----------
        n_vials : int
            The number of vials to create. Default is 2.
        rack_capacity : int
            The capacity of the rack. Default is 4.
        vial_type : str
            The type of vial. Default is "HRV".
        rack_id : str
            The identifier of the rack. Default is "VialRack1".
        vial_id_inherit : bool
            Whether the vial identifiers should be inherited from the rack. Default is True.

        Returns
        -------
        rack : JuniorRack
            The rack with the empty vials.
        vials : list[JuniorVial]
            The list of empty vials.

        """

        rack = JuniorRack.from_capacity(
            can_contain=[JuniorVial.rdf_type, ], capacity=rack_capacity, container_id=rack_id
        )

        assert n_vials <= rack.slot_capacity, f"{n_vials} {rack.slot_capacity}"

        vials = []
        n_created = 0
        for k in range(rack.slot_capacity):
            idx = str(k+1)
            if vial_id_inherit:
                v = JuniorVial(
                    identifier=f"{rack_id}/vial-{idx}", contained_by=rack.identifier,
                    contained_in_slot=idx, vial_type=vial_type,
                )
            else:
                v = JuniorVial(
                    contained_by=rack.identifier, contained_in_slot=idx, vial_type=vial_type,
                )
            rack.has_slot_content.add(v)
            vials.append(v)
            n_created += 1
            if n_created == n_vials:
                break
        return rack, vials


class JuniorVpg(LabContainee, LabContainer, JuniorLabObject):
    """A vial plate gripper of the JUNIOR platform.

    Parameters
    ----------
    can_contain : list[str]
        The list of objects that can be placed in the vial plate gripper.

    """

    can_contain: Can_contain[LabObject] = [JuniorRack.rdf_type, ]

    @property
    def rack(self) -> JuniorRack | None:
        """
        The rack in the vial plate gripper.

        Returns
        -------
        vial_plate_gripper : JuniorVpg
            The vial plate grippe.

        """
        i = self.slot_content['SLOT']
        if i is None:
            return None
        return JUNIOR_LAB[i]


class JuniorPdp(LabContainee, LabContainer, JuniorLabObject):
    """The positive displacement pipette of the Junior platform.

    Parameters
    ----------
    can_contain : list[str]
        Whether the positive displacement pipette can contain other objects. Default is a list of
        strings containing the name of the positive displacement pipette tip.
    """

    can_contain: Can_contain[LabObject] = [JuniorPdpTip.rdf_type, ]

    @property
    def tip(self) -> JuniorPdpTip | None:
        """The positive displacement pipette tip.

        Returns
        -------
        tip : JuniorPdpTip
            The positive displacement pipette tip.

        """
        i = self.slot_content['SLOT']
        if i is None:
            return None
        return JUNIOR_LAB[i]


class Powder_param_known(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology

class JuniorSvt(LabContainee, LabContainer, JuniorLabObject):
    """The SV tool which is the z2 attachment used to hold a SV vial.

    Parameters
    ----------
    can_contain : list[str]
        Whether the SV tool can contain other objects. Default is a list of strings containing the
        name of the SV vial.
    powder_param_known : bool
        Whether the powder parameter is known. Default is False.

    """

    can_contain: Can_contain[LabObject] = [JuniorVial.rdf_type, ]

    powder_param_known: Powder_param_known[bool] = False

    @property
    def sv_vial(self) -> JuniorVial | None:
        """The SV vial.

        Returns
        -------
        sv_vial : JuniorVial
            The SV vial.

        """
        i = self.slot_content['SLOT']
        if i is None:
            return None
        return JUNIOR_LAB[i]


class JuniorZ1Needle(ChemicalContainer, LabContainee, JuniorLabObject):
    """The Z1 needle of the Junior platform."""
    pass


class JuniorWashBay(JuniorLabObject):
    """The wash bay for washing needles.

    Parameters
    ----------
    layout : JuniorLayout
        The layout of the wash bay. Default is None.

    """
    layout: Layout[JuniorLayout]


class Disposal_content(DatatypeProperty):
    # TODO should be an object property
    rdfs_isDefinedBy = JuniorOntology

class JuniorTipDisposal(JuniorLabObject):
    """The tip disposal for used PdpTips.

    Parameters
    ----------
    layout : JuniorLayout
        The layout of the tip disposal. Default is None.
    disposal_content : list[str]
        The list of objects that can be disposed of in the tip disposal. Default is an empty list.

    """
    layout: Layout[JuniorLayout]

    disposal_content: Optional[Disposal_content[str]] = set()


"""python
# this becomes useless when we have `model_post_init`...
from functools import wraps
from typing import TypeVar, ParamSpec, Callable
T = TypeVar('T')
P = ParamSpec('P')
def add_to_junior_lab(func: Callable[P, T], ) -> Callable[P, T]:
    # TODO figure out how to do type hinting with new paramspc
    @wraps(func)
    def add_to_lab(*args: P.args, **kwargs: P.kwargs) -> T:
        created = func(*args, **kwargs)
        try:
            JUNIOR_LAB.add_object(created)
        except AttributeError:
            for obj in created:
                JUNIOR_LAB.add_object(obj)
        return created

    return add_to_lab
"""
