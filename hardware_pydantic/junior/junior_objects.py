from __future__ import annotations

from functools import wraps
from typing import TypeVar, ParamSpec, Callable

from hardware_pydantic.lab_objects import *

JUNIOR_LAB = Lab()
_JUNIOR_VIAL_TYPE = Literal["HRV", "MRV", "SV"]

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


class JuniorRack(Rack):
    slot_size: str | None = None

    @staticmethod
    # @add_to_junior_lab
    def create_rack_with_empty_vials(n_vials: int, rack_capacity: int, vial_type: _JUNIOR_VIAL_TYPE,
                                     rack_id: str = None) -> list[JuniorRack | JuniorVial]:
        created = []
        rack = JuniorRack.from_capacity(rack_capacity, rack_id=rack_id)
        created.append(rack)
        assert n_vials <= rack.capacity
        for k in rack.content:
            v = JuniorVial(position=k, position_relative=rack.identifier, type=vial_type)
            rack.content[k] = v
            created.append(v)
        return created

    @staticmethod
    def put_rack_in_a_slot(rack: JuniorRack, slot: 'JuniorSlot'):
        assert slot.can_hold == rack.__class__.__name__
        rack.position = slot.identifier
        slot.content = rack.identifier

    def model_post_init(self, *args) -> None:  # this should be better than `add_to_junior_lab`, you need pydantic 2.x
        JUNIOR_LAB.add_object(self)


class JuniorVial(Vial):
    type: _JUNIOR_VIAL_TYPE

    slot_size: str | None = None

    def model_post_init(self, *args) -> None:  # this should be better than `add_to_junior_lab`, you need pydantic 2.x
        JUNIOR_LAB.add_object(self)


class JuniorZ2Attachment(LabObject):
    position: str | None = None

    def model_post_init(self, *args) -> None:  # this should be better than `add_to_junior_lab`, you need pydantic 2.x
        JUNIOR_LAB.add_object(self)


class JuniorVPG(JuniorZ2Attachment):
    """ vial plate gripper """
    holding_rack: str | None


class JuniorPDT(JuniorZ2Attachment):
    content: dict[str, float | None]
    last_held: list[str]


class JuniorSvTool(JuniorZ2Attachment):
    vial_connected_to: str | None
    vial_last_held: str | None
