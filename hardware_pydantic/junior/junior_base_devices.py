from __future__ import annotations

from hardware_pydantic.base import Device, PreActError, DEVICE_ACTION_METHOD_ACTOR_TYPE
from hardware_pydantic.junior.junior_objects import JuniorStirBar
from hardware_pydantic.junior.settings import *
from hardware_pydantic.lab_objects import LabContainer, ChemicalContainer
from hardware_pydantic.junior.utils import running_time_aspirate, running_time_dispensing

_eps = 1e-7

class JuniorBaseHeater(Device, LabContainer, JuniorLabObject):
    """The heating component under a rack slot. Please note it cannot be read directly.

    Parameters
    ----------
    can_heat : bool
        Tag to indicate if the object can be heated or not. If False all related actions would
        error out.
    set_point : float
        Current set point in Celsius.
    set_point_max : float
        Allowed max set point.

    """

    can_heat: bool = True
    set_point: float = 25
    set_point_max: float = 400

    def action__set_point(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
            set_point: float = 25
    ) -> tuple[list[LabObject], float] | None:
        """The action of setting the temperature point of a heater.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action.
        set_point : float, optional
            The temperature point to set, by default 25 Celsius.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by the action and the running time of the
            action if the actor type is 'proj'.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and the set point is out of range. The other case is if
            the actor type is 'post' and the lab object/device can not be heated.
        ValueError
            If the actor type is not 'pre', 'post' or 'proj'.

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
    """The stirrer unit under a rack slot.

    Parameters
    ----------
    can_stir : bool
        Tag to indicate if the object can be stirred or not. If False all related actions would
        error out. Default is True.
    stir_turned_on : bool
        Tag to indicate if the stirrer is turned on or not. Default is False.

    """
    can_stir: bool = True
    stir_turned_on: bool = False

    def action__onoff_switch(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
    ) -> tuple[list[LabObject], float] | None:
        """The action to switch on or off the stirrer.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE, optional
            The actor type of the action, 'pre', 'post' or 'proj'. Default is 'pre'.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by the action and the running time of the action.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and the lab object/device can not be stirred.
        ValueError
            If the actor type is not 'pre', 'post' or 'proj'.

        Notes
        -----
        If there is a stirring bar then it starts/stops spinning.

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
    """A generic liquid dispenser for the Junior platform."""

    def action__aspirate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            source_container: ChemicalContainer,
            dispenser_container: ChemicalContainer,
            amount: float,
    ) -> tuple[list[LabObject], float] | None:
        """The action of aspirating liquid from a container.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action.
        source_container : ChemicalContainer
            The container to aspirate from.
        dispenser_container : ChemicalContainer
            The container to dispense to.
        amount : float
            The amount to aspirate.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by the action and the running time of the
            aspirate action if the actor type is 'proj'.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and the amount is out of range. The other case is if
            required amount is larger than the content sum of the source container.
        ValueError
            If the actor type is not 'pre', 'post' or 'proj'.

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
            return [source_container, dispenser_container], running_time_aspirate(amount)
        else:
            raise ValueError

    def action__dispense(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            destination_container: ChemicalContainer,
            dispenser_container: ChemicalContainer,
            amount: float,
            scaling_factor: float = 1.0,
    ) -> tuple[list[LabObject], float] | None:
        """The action of dispensing liquid to a container.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action, one of 'pre', 'post' or 'proj'.
        destination_container : ChemicalContainer
            The container to dispense to.
        dispenser_container : ChemicalContainer
            The container to dispense from.
        amount : float
            The amount to dispense.
        scaling_factor : float, optional
            The scaling factor for the running time of the action. Default is 1.0.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by the action and the running time of the dispense
            action if the actor type is 'proj'.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and the amount is out of range.
        ValueError
            If the actor type is not 'pre', 'post' or 'proj'.

        """
        if actor_type == 'pre':
            if amount > dispenser_container.content_sum + _eps:
                raise PreActError(f"{amount} > {dispenser_container.content_sum}")
            if amount + destination_container.content_sum > destination_container.volume_capacity:
                raise PreActError
        elif actor_type == 'post':
            removed = dispenser_container.remove_content(amount)
            destination_container.add_content(removed)
        elif actor_type == 'proj':
            running_time = running_time_dispensing(amount) * scaling_factor

            return [destination_container, dispenser_container], running_time
        else:
            raise ValueError
