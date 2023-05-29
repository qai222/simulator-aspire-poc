from __future__ import annotations

from hardware_pydantic.lab_objects import *


class Heater(Device):
    """Heating unit on the fixed deck."""
    # the header denotes the hotplate in the fixed deck of the Junior Unchained robot for now
    set_point: float = 25
    set_point_max: float = 400
    reading: float = 25
    content: LabObject | None = None

    def pre__set_point(self, set_point: float = 25) -> tuple[list[LabObject], float]:
        """Set the temperature to the set-point."""
        if self.set_point > self.set_point_max:
            raise PreActError
        return [], 1e-5

    def post__set_point(self, set_point) -> None:
        """Update the set-point after the temperature is set."""
        self.set_point = set_point

    def pre__heat_process(self) -> tuple[list[LabObject], float]:
        """Time needed to change the temperature to the set-point."""
        # TODO heating and cooling rate should be different and it should not be a constant
        heat_rate = 10
        return [], abs(self.set_point - self.reading) / heat_rate

    def post__heat_process(self) -> None:
        """The reading reflects the temperature of the heater during the heating process."""
        self.reading = self.set_point



class Cooler(Heater):
    """Cooling unit on the fixed deck.

    The cooler is inherited from the heater. The only difference is the cooling rate.
    """
    def post__cool_process(self):
        """Time elapsed reaching the reading point."""
        cool_rate = 15
        return [], abs(self.set_point - self.reading) / cool_rate


class Stirrer(Device):
    """Stirrer on the fixed deck."""
    set_point: float
    reading: float = 0
    content: LabObject | None = None

    def pre__set_point(self, set_point: float):
        """Change the stirring rate to the set-point."""
        self.set_point = set_point

    def post__set_point(self, set_point: float):
        """Time needed to change the stirring rate to the set-point."""
        # todo: redefine the time to set up the stirrer later
        return 1e-5

    def pre__stirring_process(self):
        """The reading reflects the stirring rate of the stirrer during the stirring process."""
        self.reading = self.set_point

    def post__stirring_process(self):
        """Time elapsed reaching the reading point."""
        # todo: redefine stirring rate later
        stirring_increase_rate = 10
        return abs(self.set_point - self.reading) / stirring_increase_rate


    def post__heat_process(self) -> None:
        self.reading = self.set_point

class LiquidDispenser(Device):
    """Generic hardware for liquid transfer."""
    # TODO subparts
    capacity: float = 40
    last_held: dict[str, float] = dict()

    def pre__transfer(self, from_vial: Vial, to_vial: Vial, amount: float) -> tuple[list[LabObject], float]:
        """Transfer liquid from one vial to another."""
        if amount > self.capacity:
            raise PreActError
        if amount > from_vial.content_sum:
            raise PreActError
        # TODO sample from a dist
        aspirate_speed = 5
        dispense_speed = 5
        return [from_vial, to_vial], amount / aspirate_speed + amount / dispense_speed

    def post__transfer(self, from_vial: Vial, to_vial: Vial, amount: float):
        """Time needed to transfer liquid from one vial to another."""
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
            vial: Vial,
            to_position: str | None
    ) -> None:
        """Transfer a vial from one place to another."""
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
    # note: this is not used for the current simulation
    def pre__transfer(self, from_obj: Vial, to_obj: Vial, amount: float):
        """Transfer solids from one vial to another."""
        # TODO sample from a dist
        removed = from_obj.remove_content(amount)
        to_obj.add_content(removed)

    def post__transfer(self, from_obj: Vial, to_obj: Vial, amount: float):
        """Time needed to transfer solids from one vial to another."""
        # todo: add adaptive post to estimate the time of transferring solids
        # transfer speed in g/s
        transfer_speed = 2
        return amount / transfer_speed


class Balance(Device):
    """Balance for weighing solids."""
    reading: float = 0
    content: LabObject | None = None

    def pre__tare(self):
        """Set the reading to zero."""
        self.reading = 0

    def post__tare(self):
        """Time needed to tare the balance."""
        return 1.e-6

    def pre__weigh(self, amount: float):
        """Set the reading to the amount."""
        self.reading = amount

    def post__weigh(self):
        """Time needed to weigh solids."""
        # todo: add adaptive post to estimate the time of weighing solids
        # assuming the weighing takes 10 seconds
        return 10


class Evaporator:
    """Evaporator for evaporating solvents."""
    # denotes v-10 evaporator for the moment
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

    def pre__set_point(self, set_point: dict[str, float]):
        """Set the pressure, temperature, rpm of the evaporator."""
        self.set_point.update(set_point)

    def post__set_point(self, set_point: dict[str, float]):
        """Time needed to set the pressure, temperature, rpm of the evaporator."""
        return 1.e-6

    def pre__evaporate_process(self):
        """Evaporate the solvent."""
        self.reading.update(self.set_point)

    def post__evaporate_process(self, set_point: dict[str, float]):
        """Time needed to reach the set-points for evaporation."""
        # todo: add adaptive post to estimate the time of setting up the evaporator for
        # temperature, pressure, and rpm
        # assuming the setting up takes 60 seconds
        return 60
