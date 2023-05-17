from __future__ import annotations

from hardware.lab_objects import *


class Balance(Device):

    def __init__(self, identifier: str = str_uuid(), occupied: bool = False, max_capacity: float = 2000,
                 reading: float = 0, reading_precision: float = 0.001, reading_unit: str = "g"):
        super().__init__(identifier=identifier, occupied=occupied)
        self.reading_unit = reading_unit
        self.reading_precision = reading_precision
        self.reading = reading
        self.max_capacity = max_capacity

    # def validate_state(self, state: dict[str, Any]) -> bool:
    #     return state['reading'] < state['max_capacity']


class Heater(Device):

    def __init__(
            self, identifier: str = str_uuid(), occupied: bool = False, max_capacity: float = 400,
            set_point: float = 25, reading: float = 25, reading_precision: float = 0.001, reading_unit: str = "C"
    ):
        super().__init__(identifier=identifier, occupied=occupied)
        self.set_point = set_point
        self.reading_unit = reading_unit
        self.reading_precision = reading_precision
        self.reading = reading
        self.max_capacity = max_capacity

    # def validate_state(self, state: dict[str, Any]) -> bool:
    #     return state['reading'] < state['max_capacity']

    @action_method_logging
    def action__set_point(self, set_point: float = 25):
        self.set_point = set_point
        return Device._default_action_duration

    @action_method_logging
    def action__heat_object(self):
        # TODO heating and cooling rate should be different and it should not be a constant
        heat_rate = 10
        return abs(self.set_point - self.reading) / heat_rate


class LiquidTransferor(Device):

    def __init__(
            self,
            identifier: str = str_uuid(),
            occupied: bool = False,
            max_capacity: float = 10,
            # content: dict = None,
    ):
        super().__init__(identifier=identifier, occupied=occupied)
        # if content is None:
        #     content = dict()
        # self.content = content
        self.max_capacity = max_capacity

    @action_method_logging
    def action__transfer_between_vials(self, from_obj: Vial, to_obj: Vial, amount: float):
        # TODO sample from a dist
        transfer_speed = 5
        removed = from_obj.remove_content(amount)
        to_obj.add_content(removed)
        return amount / transfer_speed
