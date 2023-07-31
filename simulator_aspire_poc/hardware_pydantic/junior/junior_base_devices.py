from __future__ import annotations

from hardware_pydantic.base import Device, PreActError, DEVICE_ACTION_METHOD_ACTOR_TYPE
from hardware_pydantic.junior.junior_objects import JuniorStirBar
from hardware_pydantic.junior.settings import *
from hardware_pydantic.lab_objects import LabContainer, ChemicalContainer


class JuniorBaseHeater(Device, LabContainer, JuniorLabObject):
    """ the heating component under a rack slot, it cannot be read directly """

    can_heat: bool = True
    """ if False all related actions would error out """

    set_point: float = 25
    """ current set point in C """

    set_point_max: float = 400
    """ allowed max set point """

    def action__set_point(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
            set_point: float = 25
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: set_point
        DESCRIPTION: set the temperature point of a heater
        PARAMS:
            - set_point: float = 25
        """
        if actor_type == 'pre':
            if not self.can_heat:
                raise PreActError
            if self.set_point > self.set_point_max:
                raise PreActError
        elif actor_type == 'post':
            self.set_point = set_point
        elif actor_type == 'proj':
            return [], 1e-6
        else:
            raise ValueError


class JuniorBaseStirrer(Device, LabContainer, JuniorLabObject):
    can_stir: bool = True
    """ if False all related actions would error out """

    stir_turned_on: bool = False

    def action__onoff_switch(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: start_stir
        DESCRIPTION: start/stop stirring, if there is a stirring bar then it starts/stops spinning
        PARAMS:
        """
        stirring_bars = []
        containees = LabContainer.get_all_containees(self, JUNIOR_LAB)
        for c in containees:
            cc = JUNIOR_LAB[c]
            if isinstance(cc, JuniorStirBar):
                stirring_bars.append(cc)

        if actor_type == 'pre':
            if not self.can_stir:
                raise PreActError
        elif actor_type == 'post':
            self.stir_turned_on = not self.stir_turned_on
            for sb in stirring_bars:
                sb.is_spinning = not sb.is_spinning
        elif actor_type == 'proj':
            return stirring_bars, 1e-6
        else:
            raise ValueError


class JuniorBaseLiquidDispenser(Device, JuniorLabObject):
    """ a generic liquid dispenser """

    def action__aspirate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            source_container: ChemicalContainer,
            dispenser_container: ChemicalContainer,
            amount: float,
            aspirate_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: aspirate
        DESCRIPTION: aspirate liquid from a ChemicalContainer to the dispenser_container (ex. PdpTip)
        PARAMS:
            - source_container: ChemicalContainer,
            - dispenser_container: ChemicalContainer,
            - amount: float,
            - aspirate_speed: float = 5,
        """
        if actor_type == 'pre':
            if amount > dispenser_container.volume_capacity:
                raise PreActError
            if amount > source_container.content_sum:
                raise PreActError
        elif actor_type == 'post':
            removed = source_container.remove_content(amount)
            dispenser_container.add_content(removed)
        elif actor_type == 'proj':
            return [source_container, dispenser_container], amount / aspirate_speed
        else:
            raise ValueError

    def action__dispense(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            destination_container: ChemicalContainer,
            dispenser_container: ChemicalContainer,
            amount: float,
            dispense_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: dispense
        DESCRIPTION: dispense liquid from the dispenser_container (ex. PdpTip) to a ChemicalContainer
        PARAMS:
            - actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            - destination_container: ChemicalContainer,
            - dispenser_container: ChemicalContainer,
            - amount: float,
            - dispense_speed: float = 5,
        """
        if actor_type == 'pre':
            if amount > dispenser_container.content_sum:
                raise PreActError
            if amount + destination_container.content_sum > destination_container.volume_capacity:
                raise PreActError
        elif actor_type == 'post':
            removed = dispenser_container.remove_content(amount)
            destination_container.add_content(removed)
        elif actor_type == 'proj':
            return [destination_container, dispenser_container], amount / dispense_speed
        else:
            raise ValueError
