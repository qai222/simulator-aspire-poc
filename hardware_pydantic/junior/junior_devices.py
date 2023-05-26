from __future__ import annotations

from hardware_pydantic.devices import Heater
from hardware_pydantic.junior.junior_objects import *

_SLOT_SIZE_X = 80
_SLOT_SIZE_Y = 120
_SMALL_SLOT_SIZE_X = 20
_SMALL_SLOT_SIZE_Y = 20


class JuniorDevice(Device):

    def model_post_init(self, *args) -> None:  # this should be better than `add_to_junior_lab`, you need pydantic 2.x
        JUNIOR_LAB.add_object(self)


class JuniorSlot(JuniorDevice, Heater):
    """ I'm using `Device` here because some slots function as `Balance` or `Heater` """

    can_weigh: bool = False

    can_hold: str | None
    """ what `JuniorObject` it can hold? used for typing checking """

    can_heat: bool = False

    can_cool: bool = False

    can_stir: bool = False

    content: str | None = None
    """ the identifier of the object it currently holds """

    # layout related

    layout_position: tuple[float, float] | None = None
    """ left bot """

    layout_x: float = _SLOT_SIZE_X

    layout_y: float = _SLOT_SIZE_Y

    @staticmethod
    # @add_to_junior_lab
    def create_slot(
            identifier: str,
            layout_relation: Literal["above", "right_to"] = None,
            layout_relative: JuniorSlot = None,
            layout_x: float = _SLOT_SIZE_X, layout_y: float = _SLOT_SIZE_Y,
            can_cool=False, can_heat=False, can_stir=False, can_hold: str | None = JuniorRack.__name__, can_weigh=False,
            content: str = None
    ) -> JuniorSlot:
        if layout_relative is None:
            abs_layout_position = (0, 0)
        else:
            if layout_relation == "above":
                abs_layout_position = (
                    layout_relative.layout_position[0],
                    layout_relative.layout_position[1] + layout_relative.layout_y + 20
                )
            elif layout_relation == "right_to":
                abs_layout_position = (
                    layout_relative.layout_position[0] + layout_relative.layout_x + 20,
                    layout_relative.layout_position[1],
                )
            else:
                raise ValueError
        slot = JuniorSlot(
            identifier=identifier,
            can_hold=can_hold, can_cool=can_cool, can_heat=can_heat, can_stir=can_stir, can_weigh=can_weigh,
            layout_position=abs_layout_position, layout_x=layout_x, layout_y=layout_y,
            content=content,
        )
        return slot


_z1_needle_labels = ["a", "b", "c", "d", "e", "f", "g"]
_z1_needle_label = Literal["a", "b", "c", "d", "e", "f", "g"]


class JuniorArmZ1(Device):
    position_on_top_of: str
    """ the current position """

    needle_content: dict[_z1_needle_label, dict[str, float]] = {l: dict() for l in _z1_needle_labels}
    """ liquid composition of needles """

    needle_capacity: float = float('inf')

    moving_to: str | None = None
    """ where am I going? """

    can_access: list[str] = []
    """ a list of slot identifiers that this arm can access """

    def pre__transfer(self, use_needle: _z1_needle_label, from_vial: JuniorVial, to_vial: JuniorVial, amount: float) -> \
    tuple[list[LabObject], float]:
        if self.position_on_top_of != from_vial.identifier:
            raise PreActError
        if len(self.needle_content[use_needle]) != 0:
            raise PreActError
        if from_vial.content_sum < 1e-5:
            raise PreActError
        if amount > self.needle_capacity:
            raise PreActError
        if amount > from_vial.content_sum:
            raise PreActError
        # TODO sample from distributions
        aspirate_speed = 5
        dispense_speed = 5
        move_cost = 5
        return [from_vial, to_vial], amount / aspirate_speed + amount / dispense_speed + move_cost

    def post__transfer(self, from_vial: JuniorVial, to_vial: JuniorVial, amount: float):
        removed = from_vial.remove_content(amount)
        to_vial.add_content(removed)
        self.position_on_top_of = to_vial.identifier  # assuming vial always held in a rack

    # TODO action wash
    # TODO split action transfer into aspirate, move, dispense


