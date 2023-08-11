from __future__ import annotations

from hardware_pydantic.base import Device, PreActError, DEVICE_ACTION_METHOD_ACTOR_TYPE
from hardware_pydantic.tecan.settings import *
from hardware_pydantic.lab_objects import LabContainer, ChemicalContainer


"""Device classes for Tecan devices."""


class TecanBaseHeater(Device, LabContainer, TecanLabObject):
    """The heating component under a rack slot but it cannot be read directly.

    Attributes
    ----------
    can_heat : bool
        If False all related actions would error out. Defaults to True.
    set_point : float
        Current set point in Celsius. Defaults to 25.
    set_point_max : float
        Allowed max set point with the unit of Celsius. Defaults to 400.

    Notes
    -----
    The temperature unit is Celsius

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
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE, optional
            The actor type of the action, 'pre', 'post', or 'proj'. Defaults to 'pre'.
        set_point : float, optional
            The temperature point to set. Defaults to 25.


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



class TecanBaseLiquidDispenser(Device, TecanLabObject):
    """ a generic liquid dispenser """

    def action__aspirate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            source_container: ChemicalContainer,
            dispenser_container: ChemicalContainer,
            amount: float,
            aspirate_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        """The action of aspirating liquid from a ChemicalContainer to the dispenser_container
        (ex. PdpTip).

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action, 'pre', 'post', or 'proj'.
        source_container : ChemicalContainer
            The container to aspirate from.
        dispenser_container : ChemicalContainer
            The container to dispense to.
        amount : float
            The amount of liquid to aspirate.
        aspirate_speed : float, optional
            The speed of aspirating. Default is 5.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of LabObjects that are involved in the action and the time it takes to
            complete the action.

        Raises
        ------
        PreActError
            If the amount of liquid to aspirate is greater than the volume capacity or the amount
            of liquid in the source container is less than the amount to aspirate.
        ValueError
            If the actor_type is not 'pre', 'post', or 'proj'.

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
        """The action of dispensing liquid from the dispenser_container (ex. PdpTip) to a
        ChemicalContainer.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action, 'pre', 'post', or 'proj'.
        destination_container : ChemicalContainer
            The container to dispense to.
        dispenser_container : ChemicalContainer
            The container to dispense from.
        amount : float
            The amount of liquid to dispense.
        dispense_speed : float, optional
            The speed of dispensing. Default is 5.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of LabObjects that are involved in the action and the time it takes.

        Raises
        ------
        PreActError
            If the amount of liquid to dispense is greater than the volume capacity or the amount
            of liquid in the dispenser container is less than the amount to dispense.
        ValueError
            If the actor_type is not 'pre', 'post', or 'proj'.

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
