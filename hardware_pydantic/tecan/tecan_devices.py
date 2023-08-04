from __future__ import annotations

from hardware_pydantic.base import Device, DEVICE_ACTION_METHOD_ACTOR_TYPE, PreActError
from hardware_pydantic.tecan.settings import *
from hardware_pydantic.tecan.tecan_base_devices import TecanBaseHeater, TecanBaseLiquidDispenser
from hardware_pydantic.tecan.tecan_objects import *


class TecanSlot(TecanBaseHeater):
    """Slot in the Tecan deck, can contain plates, liquid tanks, etc.

    Attributes
    ----------
    can_contain : list[str]
        List of objects that can be contained in this slot.
    can_heat : bool
        Whether this slot can be heated. If True, then the slot can be heated to a certain
        temperature. Default is False.
    layout : TecanLayout | None
        The layout of the slot. Default is None.

    """
    can_contain: list[str] = [TecanPlate.__name__, ]

    can_heat: bool = False

    layout: TecanLayout | None = None

    def action__wait(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
            wait_time: float = 0
    ) -> tuple[list[LabObject], float] | None:
        """The action of waiting for a certain amount of time.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE, optional
            The actor type of the action. Default is 'pre'.
        wait_time : float, optional
            The amount of time to wait. Default is 0.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by this action and the amount of time.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and the slot cannot be heated.
        ValueError
            If the actor type is not 'pre', 'post', or 'proj'.

        """
        if actor_type == 'pre':
            if not self.can_heat:
                raise PreActError
        elif actor_type == 'post':
            return
        elif actor_type == 'proj':
            containees = self.get_all_containees(container=self, lab=TECAN_LAB)
            return [TECAN_LAB[i] for i in containees], wait_time
        else:
            raise ValueError

    @staticmethod
    def put_plate_in_a_slot(plate: TecanPlate, tecan_slot: TecanSlot):
        """Put a plate in a slot.

        Parameters
        ----------
        plate : TecanPlate
            The plate to put in the slot.
        tecan_slot : TecanSlot
            The slot to put the plate in.

        """
        if plate.contained_by is not None:
            prev_slot = TECAN_LAB[plate.contained_by]
            assert isinstance(prev_slot, TecanSlot)
            prev_slot.slot_content["SLOT"] = None
        assert plate.__class__.__name__ in tecan_slot.can_contain
        plate.contained_by = tecan_slot.identifier
        plate.contained_in_slot = "SLOT"
        tecan_slot.slot_content["SLOT"] = plate.identifier


class TecanArm(Device, LabContainer, TecanLabObject):
    """The Tecan arm.

    Attributes
    ----------
    position_on_top_of : str | None
        The current position, can only be a slot (not vial). Default is None.

    """
    position_on_top_of: str | None = None

    def action__move_to(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            move_to_slot: TecanSlot | TecanLiquidTank | TecanHotel,
    ) -> tuple[list[LabObject], float] | None:
        """The action of moving to a slot.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action.
        move_to_slot : TecanSlot | TecanLiquidTank | TecanHotel
            The slot to move to.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by this action and the amount of time.

        Raises
        ------
        ValueError
            If the actor type is not 'pre', 'post', or 'proj'.

        """
        if self.position_on_top_of == move_to_slot.identifier:
            move_cost = 1e-6
        else:
            move_cost = 5

        if actor_type == 'pre':
            return
        elif actor_type == 'post':
            self.position_on_top_of = move_to_slot.identifier
        elif actor_type == 'proj':
            containees = self.get_all_containees(container=self, lab=TECAN_LAB)
            return [TECAN_LAB[i] for i in containees], move_cost
        else:
            raise ValueError


class TecanArm1(TecanArm, TecanBaseLiquidDispenser):
    """The Tecan arm 1.

    Attributes
    ----------
    slot_content : dict[str, str]
        The content of the slot. Default is {}.
    can_contain : list[str]
        List of objects that can be contained in this slot. Default is [TecanArm1Needle.__name__, ].

    """
    slot_content: dict[str, str] = dict()

    can_contain: list[str] = [TecanArm1Needle.__name__, ]

    def action__concurrent_aspirate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            source_container: TecanLiquidTank,
            dispenser_containers: list[TecanArm1Needle],
            amounts: list[float],
            aspirate_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        """Action of aspirating liquid from a list of ChemicalContainer to a list of
        dispenser_container.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action.
        source_container : TecanLiquidTank
            The source container.
        dispenser_containers : list[TecanArm1Needle]
            The list of dispenser containers.
        amounts : list[float]
            The list of amounts to aspirate.
        aspirate_speed : float, optional
            The speed of aspiration. Default is 5.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by this action and the amount of time.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and the number of `dispenser_containers` is not equal to
            the number of amounts.

        """
        if actor_type == 'pre':
            if not len(dispenser_containers) == len(amounts):
                raise PreActError
        objs = []
        times = []
        for d, a in zip(dispenser_containers, amounts):
            res = self.action__aspirate(actor_type=actor_type, source_container=source_container,
                                        dispenser_container=d, amount=a, aspirate_speed=aspirate_speed)
            if actor_type == 'proj':
                objs += res[0]
                times.append(res[1])
        if actor_type == 'proj':
            return objs, max(times)

    def action__concurrent_dispense(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            destination_containers: list[ChemicalContainer],
            dispenser_containers: list[TecanArm1Needle],
            amounts: list[float],
            dispense_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        """The action of dispensing liquid from a list of dispenser_container to a list of
        ChemicalContainer.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action.
        destination_containers : list[ChemicalContainer]
            The list of destination containers.
        dispenser_containers : list[TecanArm1Needle]
            The list of dispenser containers.
        amounts : list[float]
            The list of amounts to dispense.
        dispense_speed : float, optional
            The speed of dispensing. Default is 5.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by this action and the amount of time.

        Raises
        ------
        PreActError
            If there is more than one type of destination containees or the number of
            `destination_containers`, `dispenser_containers`, and `amounts` are not equal.

        """
        if actor_type == 'pre':
            if len(set([TECAN_LAB[dc.contained_by] for dc in destination_containers])) != 1:
                raise PreActError
            if not len(destination_containers) == len(dispenser_containers) == len(amounts):
                raise PreActError
        objs = []
        times = []
        for s, d, a in zip(destination_containers, dispenser_containers, amounts):
            res = self.action__dispense(actor_type=actor_type, destination_container=s, dispenser_container=d, amount=a,
                                        dispense_speed=dispense_speed)
            if actor_type == 'proj':
                objs += res[0]
                times.append(res[1])
        if actor_type == 'proj':
            return objs, max(times)

    def action__wash(self, actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE, wash_bay: TecanWashBay):
        """The action of washing the needle.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action.
        wash_bay : TecanWashBay
            The wash bay.

        Returns
        -------
        list[LabObject]
            The list of objects that are affected by this action.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and the arm is not on top of the wash bay.

        """
        if actor_type == 'pre':
            if self.position_on_top_of != wash_bay.identifier:
                raise PreActError
        elif actor_type == 'post':
            for n in self.slot_content.values():
                TECAN_LAB[n].chemical_content = dict()
        elif actor_type == 'proj':
            containees = self.get_all_containees(container=self, lab=TECAN_LAB)
            return [TECAN_LAB[i] for i in containees], 10


class TecanArm2(TecanArm):
    """The Tecan Arm 2.

    Attributes
    ----------
    identifier : str
        The identifier of the arm.

    """
    can_contain: list[str] = [TecanPlate.__name__, ]

    @property
    def attachment(self) -> TecanPlate | None:
        """The attachment of the arm.

        Returns
        -------
        TecanPlate | None
            The attachment of the arm.

        """
        if self.slot_content['SLOT'] is None:
            return None
        return TECAN_LAB[self.slot_content['SLOT']]

    def action__pick_up_plate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            thing: TecanPlate,
    ) -> tuple[list[LabObject], float] | None:
        """The action of picking up a TecanPlate.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action.
        thing : TecanPlate
            The TecanPlate to pick up.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by this action and the amount of time.

        Raises
        ------
        PreActError
            It errors out with any of these two conditions:
            1. If the actor type is 'pre' and the arm is not on top of the thing or the
            arm.
            2. If the actor type is 'pre' and the arm already has an attachment.
        ValueError
            If the actor type is not 'pre', 'post', or 'proj'.

        """

        if actor_type == 'pre':
            thing_slot = TECAN_LAB[thing.contained_by]
            # TODO merge this with "move_to"
            if thing_slot.identifier != self.position_on_top_of:
                raise PreActError(
                    f"you are picking up from: {thing_slot.identifier} "
                    f"but the arm is on top of: {self.position_on_top_of}")
            if self.attachment is not None:
                raise PreActError("already has an attachment, cannot pick up")
        elif actor_type == 'post':
            LabContainee.move(containee=thing, dest_container=self, lab=TECAN_LAB, dest_slot="SLOT")
        elif actor_type == 'proj':
            pickup_cost = 7
            return [thing, ], pickup_cost
        else:
            raise ValueError

    def action__put_down_plate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            dest_slot: TecanSlot | TecanHotel,
            dest_slot_key: str,
    ) -> tuple[list[LabObject], float] | None:
        """The action of putting down a TecanPlate.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type of the action.
        dest_slot : TecanSlot | TecanHotel
            The destination slot.
        dest_slot_key : str
            The key of the destination slot.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of objects that are affected by this action and the amount of time.

        Raises
        ------
        PreActError
            It errors out with any of these three conditions when the actor type is 'pre':
            1. the arm is not on top of the destination slot;
            2. the slot is not empty;
            3. the slot is a TecanHotel and the slot key is not in the slot content.
        ValueError
            If the actor type is not 'pre', 'post', or 'proj'.

        """
        thing = self.attachment
        objs = [thing, dest_slot]
        if isinstance(dest_slot, TecanSlot):
            dest_slot_key = "SLOT"

        if actor_type == 'pre':
            if dest_slot.identifier != self.position_on_top_of:
                raise PreActError
            if dest_slot.slot_content[dest_slot_key] is not None:
                raise PreActError
            if isinstance(dest_slot, TecanHotel) and dest_slot_key not in dest_slot.slot_content:
                raise PreActError

        elif actor_type == 'post':
            LabContainee.move(containee=thing, dest_container=dest_slot, lab=TECAN_LAB, dest_slot=dest_slot_key)
        elif actor_type == 'proj':
            put_down_cost = 7
            return objs, put_down_cost
        else:
            raise ValueError
