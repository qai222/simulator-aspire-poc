from __future__ import annotations

from hardware_pydantic.junior.settings import JUNIOR_LAB, JuniorLabObject, JUNIOR_VIAL_TYPE, JuniorLayout
from hardware_pydantic.lab_objects import ChemicalContainer, LabContainee, LabContainer


class JuniorStirBar(LabContainee, JuniorLabObject):
    material: str = "TEFLON"

    is_spinning: bool = False


class JuniorVial(ChemicalContainer, LabContainee, LabContainer, JuniorLabObject):
    """ a vial on the JUNIOR platform, usually placed in a rack, it can contain a stir bar """

    can_contain: list[str] = [JuniorStirBar.__name__, ]

    vial_type: JUNIOR_VIAL_TYPE = "HRV"


class JuniorPdpTip(ChemicalContainer, LabContainee, JuniorLabObject):
    """ positive displacement pipette tip """
    pass


class JuniorRack(LabContainer, LabContainee, JuniorLabObject):
    """ a rack used to hold vials or positive displacement pipette tips """

    @staticmethod
    def create_rack_with_empty_tips(
            n_tips: int = 2, rack_capacity: int = 4,
            rack_id: str = "PdpTipRack1", tip_id_inherit: bool = True
    ) -> tuple[JuniorRack, list[JuniorPdpTip]]:

        rack = JuniorRack.from_capacity(
            can_contain=[JuniorPdpTip.__name__, ], capacity=rack_capacity, container_id=rack_id
        )

        assert n_tips <= rack.slot_capacity, f"{n_tips} {rack.slot_capacity}"

        tips = []
        n_created = 0
        for k in rack.slot_content:
            if tip_id_inherit:
                v = JuniorPdpTip(
                    identifier=f"PdpTip-{k} " + rack_id, contained_by=rack.identifier, contained_in_slot=k,
                )
            else:
                v = JuniorVial(
                    contained_by=rack.identifier, contained_in_slot=k,
                )
            rack.slot_content[k] = v.identifier
            tips.append(v)
            n_created += 1
            if n_created == n_tips:
                break
        return rack, tips

    @staticmethod
    def create_rack_with_empty_vials(
            n_vials: int = 2, rack_capacity: int = 4, vial_type: JUNIOR_VIAL_TYPE = "HRV",
            rack_id: str = "VialRack1", vial_id_inherit: bool = True
    ) -> tuple[JuniorRack, list[JuniorVial]]:

        rack = JuniorRack.from_capacity(
            can_contain=[JuniorVial.__name__, ], capacity=rack_capacity, container_id=rack_id
        )

        assert n_vials <= rack.slot_capacity, f"{n_vials} {rack.slot_capacity}"

        vials = []
        n_created = 0
        for k in rack.slot_content:
            if vial_id_inherit:
                v = JuniorVial(
                    identifier=f"vial-{k} " + rack_id, contained_by=rack.identifier,
                    contained_in_slot=k, vial_type=vial_type,
                )
            else:
                v = JuniorVial(
                    contained_by=rack.identifier, contained_in_slot=k, vial_type=vial_type,
                )
            rack.slot_content[k] = v.identifier
            vials.append(v)
            n_created += 1
            if n_created == n_vials:
                break
        return rack, vials


class JuniorVpg(LabContainee, LabContainer, JuniorLabObject):
    """ vial plate gripper """

    can_contain: list[str] = [JuniorRack.__name__, ]

    @property
    def rack(self) -> JuniorRack | None:
        i = self.slot_content['SLOT']
        if i is None:
            return None
        return JUNIOR_LAB[i]


class JuniorPdp(LabContainee, LabContainer, JuniorLabObject):
    """ positive displacement pipette """

    can_contain: list[str] = [JuniorPdpTip.__name__, ]

    @property
    def tip(self) -> JuniorPdpTip | None:
        i = self.slot_content['SLOT']
        if i is None:
            return None
        return JUNIOR_LAB[i]


class JuniorSvt(LabContainee, LabContainer, JuniorLabObject):
    """ sv tool: the z2 attachment used to hold a sv vial """

    can_contain: list[str] = [JuniorVial.__name__, ]

    powder_param_known: bool = False

    @property
    def sv_vial(self) -> JuniorVial | None:
        i = self.slot_content['SLOT']
        if i is None:
            return None
        return JUNIOR_LAB[i]


class JuniorZ1Needle(ChemicalContainer, LabContainee, JuniorLabObject):
    pass


class JuniorWashBay(JuniorLabObject):
    """ for washing needles """
    layout: JuniorLayout | None = None


class JuniorTipDisposal(JuniorLabObject):
    """ for used PdpTips """
    layout: JuniorLayout | None = None

    disposal_content: list[str] = []


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
