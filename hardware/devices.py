from hardware.base import *


class Balance(Device):
    max_capacity: float = 20000

    reading: float = 0
    reading_precision: float = 0.001
    reading_unit: str = "g"

    def validate_state(self, state: dict[str, Any]) -> bool:
        return self.reading < self.max_capacity


class HeaterSetPoint(Instruction):

    set_to: float

    action_method_name: str = "action_set_point"


class Heater(Device):
    max_capacity: float = 400

    set_point: float = 25
    reading: float = 25

    heating_rate: float = 10  # per min
    cooling_rate: float = 2

    # def validate_state(self, state: dict[str, Any]) -> bool:
    #     state['max_capacity']
    #     return self.reading < self.max_capacity

    def action_set_point(self, ins: HeaterSetPoint):
        new_state_self = {"set_point": ins.set_to}
        self.change_state(self, new_state_self)


heater = Heater()
action = HeaterSetPoint(set_to=200)
print(heater)
getattr(heater, action.action_method_name)(action)
print(heater)
"""
identifier='41ab503e-6324-4e78-aaa8-af1d81fd2a91' is_occupied=False max_capacity=400 set_point=25 reading=25 heating_rate=10 cooling_rate=2
identifier='41ab503e-6324-4e78-aaa8-af1d81fd2a91' is_occupied=False max_capacity=400 set_point=200.0 reading=25 heating_rate=10 cooling_rate=2
"""
