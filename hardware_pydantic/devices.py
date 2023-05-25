from __future__ import annotations

from hardware_pydantic.lab_objects import *


class Heater(Device):
    set_point: float = 25
    reading: float = 25
    content: LabObject | None = None

    @action_method_logging
    def action__set_point(self, set_point: float = 25):
        self.set_point = set_point

    def projection__set_point(self, set_point: float = 25):
        return 1e-5

    @action_method_logging
    def action__heat_process(self):
        self.reading = self.set_point

    def projection__heat_process(self):
        # TODO heating and cooling rate should be different and it should not be a constant
        heat_rate = 10
        return abs(self.set_point - self.reading) / heat_rate


class Cooler(Heater):
    def projection__cool_process(self):
        # TODO heating and cooling rate should be different and it should not be a constant
        cool_rate = 15
        return abs(self.set_point - self.reading) / cool_rate


class Stirrer(Device):
    """Stirrer on the fixed deck."""
    set_point: float
    reading: float = 0
    content: LabObject | None = None

    @action_method_logging
    def action__set_point(self, set_point: float):
        self.set_point = set_point

    def projection__set_point(self, set_point: float):
        # todo: redefine the time to set up the stirrer later
        return 1e-5

    @action_method_logging
    def action__stirring_process(self):
        self.reading = self.set_point

    def projection__stirring_process(self):
        # todo: redefine stirring rate later
        stirring_increase_rate = 10
        return abs(self.set_point - self.reading) / stirring_increase_rate


class LiquidTransferer(Device):
    # todo: check differences among
    # PositiveDisplacementTool
    # extendable single tip (syringe pump) and multi-tip (liquid handler)
    @action_method_logging
    def action__transfer_between_vials(self, from_obj: Vial, to_obj: Vial, amount: float):
        # TODO sample from a dist
        removed = from_obj.remove_content(amount)
        to_obj.add_content(removed)

    def projection__transfer_between_vials(self, from_obj: Vial, to_obj: Vial, amount: float):
        # TODO sample from a dist
        transfer_speed = 5
        return amount / transfer_speed


class VialGripperArm(Device):
    def projection__transfer(
            self,
            from_obj: Rack | Heater,
            to_obj: Rack | Heater,
            transferee: Vial,
            to_position: str | None
    ) -> float:
        # TODO dynamic duration
        return 5

    @action_method_logging
    def action__transfer(
            self,
            from_obj: Rack | Heater,
            to_obj: Rack | Heater,
            vial: Vial,
            to_position: str | None
    ) -> None:
        # TODO make more general
        # take it out
        assert vial.position_relative == from_obj.identifier
        if isinstance(from_obj, Rack):
            assert vial.position in from_obj.content
            from_obj.content[vial.position] = None
        elif isinstance(from_obj, Heater):
            from_obj.content = None
        else:
            raise TypeError
        vial.position = None
        vial.position_relative = None
        # put it to
        if isinstance(to_obj, Rack):
            assert to_position in to_obj.content
            to_obj.content[to_position] = vial.identifier
        elif isinstance(to_obj, Heater):
            to_obj.content = vial.identifier
        vial.position = to_position
        vial.position_relative = to_obj.identifier


class PlateGripperArm(Device):
    pass


class SolidSVTool(Device):
    """Solid sample vial tool for transferring solids."""
    # todo: this device is often used together with a balance
    @action_method_logging
    def action__transfer(self, from_obj: Vial, to_obj: Vial, amount: float):
        # TODO sample from a dist
        removed = from_obj.remove_content(amount)
        to_obj.add_content(removed)

    def projection__transfer(self, from_obj: Vial, to_obj: Vial, amount: float):
        # todo: add adaptive projection to estimate the time of transferring solids
        # transfer speed in g/s
        transfer_speed = 2
        return amount / transfer_speed


class Balance(Device):
    """Balance for weighing solids."""
    reading: float = 0
    content: LabObject | None = None

    @action_method_logging
    def action__tare(self):
        self.reading = 0

    def projection__tare(self):
        return 0

    @action_method_logging
    def action__weigh(self, amount: float):
        self.reading = amount

    def projection__weigh(self):
        # todo: add adaptive projection to estimate the time of weighing solids
        # assuming the weighing takes 10 seconds
        return 10


class Evaporator:
    """Evaporator for evaporating solvents."""
    reading: dict[str, float] = {
        # temperature in Celsius
        "temperature": 25,
        # pressure in atm
        "pressure": 1,
        # rpm
        "rpm": 0,
    }
    set_point: dict[str, float] = {
        # temperature in Celsius
        "temperature": 25,
        # pressure in atm
        "pressure": 1,
        # rpm
        "rpm": 0,
    }
    content: LabObject | None = None

    @action_method_logging
    def action__set_point(self, set_point: dict[str, float]):
        self.set_point = set_point

    def projection__set_point(self, set_point: dict[str, float]):
        # todo: add adaptive projection to estimate the time of setting up the evaporator for
        # temperature, pressure, and rpm
        # assuming the setting up takes 60 seconds
        return 60
