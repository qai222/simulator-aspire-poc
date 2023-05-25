from __future__ import annotations

from hardware_pydantic.lab_objects import *


class Heater(Device):
    set_point: float = 25
    set_point_max: float = 400
    reading: float = 25
    content: LabObject | None = None

    def pre__set_point(self, set_point: float = 25) -> tuple[list[LabObject], float]:
        if self.set_point > self.set_point_max:
            raise PreActError
        return [], 1e-5

    def post__set_point(self, set_point) -> None:
        self.set_point = set_point

    def pre__heat_process(self) -> tuple[list[LabObject], float]:
        # TODO heating and cooling rate should be different and it should not be a constant
        heat_rate = 10
        return [], abs(self.set_point - self.reading) / heat_rate

    def post__heat_process(self) -> None:
        self.reading = self.set_point


class LiquidDispenser(Device):
    # TODO subparts
    capacity: float = 40
    last_held: dict[str, float] = dict()

    def pre__transfer(self, from_vial: Vial, to_vial: Vial, amount: float) -> tuple[list[LabObject], float]:
        if amount > self.capacity:
            raise PreActError
        if amount > from_vial.content_sum:
            raise PreActError
        # TODO sample from a dist
        aspirate_speed = 5
        dispense_speed = 5
        return [from_vial, to_vial], amount / aspirate_speed + amount / dispense_speed

    def post__transfer(self, from_vial: Vial, to_vial: Vial, amount: float):
        removed = from_vial.remove_content(amount)
        to_vial.add_content(removed)
        self.last_held = removed


# TODO deprecate as "Cullen 2023-05-24: vial gripper is useless on Junior, will be physically removed"
class VialTransferor(Device):

    def pre__transfer(
            self,
            from_obj: Rack | Heater,
            to_obj: Rack | Heater,
            transferee: Vial,
            to_position: str | None
    ):
        # TODO dynamic duration
        return [to_obj, transferee], 5

    def post__transfer(
            self,
            from_obj: Rack | Heater,
            to_obj: Rack | Heater,
            transferee: Vial,
            to_position: str | None
    ) -> None:
        # TODO make more general
        # take it out
        assert transferee.position_relative == from_obj.identifier
        if isinstance(from_obj, Rack):
            assert transferee.position in from_obj.content
            from_obj.content[transferee.position] = None
        elif isinstance(from_obj, Heater):
            from_obj.content = None
        else:
            raise TypeError
        transferee.position = None
        transferee.position_relative = None
        # put it to
        if isinstance(to_obj, Rack):
            assert to_position in to_obj.content
            to_obj.content[to_position] = transferee.identifier
        elif isinstance(to_obj, Heater):
            to_obj.content = transferee.identifier
        transferee.position = to_position
        transferee.position_relative = to_obj.identifier
