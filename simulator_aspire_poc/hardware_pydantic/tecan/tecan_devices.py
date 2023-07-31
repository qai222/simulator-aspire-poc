from __future__ import annotations

from simulator_aspire_poc.hardware_pydantic.base import Device, DEVICE_ACTION_METHOD_ACTOR_TYPE, PreActError
from simulator_aspire_poc.hardware_pydantic.tecan.settings import *
from simulator_aspire_poc.hardware_pydantic.tecan.tecan_base_devices import TecanBaseHeater, TecanBaseLiquidDispenser
from simulator_aspire_poc.hardware_pydantic.tecan.tecan_objects import *


class TecanSlot(TecanBaseHeater):
    can_contain: list[str] = [TecanPlate.__name__, ]

    can_heat: bool = False

    layout: TecanLayout | None = None

    def action__wait(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
            wait_time: float = 0
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: wait
        DESCRIPTION: hold everything in this slot for a while, ex. heat/cool/stir
        PARAMS:
            - wait_time: float = 0
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
        if plate.contained_by is not None:
            prev_slot = TECAN_LAB[plate.contained_by]
            assert isinstance(prev_slot, TecanSlot)
            prev_slot.slot_content["SLOT"] = None
        assert plate.__class__.__name__ in tecan_slot.can_contain
        plate.contained_by = tecan_slot.identifier
        plate.contained_in_slot = "SLOT"
        tecan_slot.slot_content["SLOT"] = plate.identifier


class TecanArm(Device, LabContainer, TecanLabObject):
    position_on_top_of: str | None = None
    """ the current position, can only be a slot (not vial) """

    def action__move_to(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            move_to_slot: TecanSlot | TecanLiquidTank | TecanHotel,
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: wait
        DESCRIPTION: hold everything in this slot for a while, ex. heat/cool/stir
        PARAMS:
            - wait_time: float = 0
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
        """
        ACTION: concurrent_aspirate
        DESCRIPTION: aspirate liquid from a list of ChemicalContainer to a list of dispenser_container
        PARAMS:
            - actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            - source_container: TecanLiquidTank,
            - dispenser_containers: list[ChemicalContainer],
            - amounts: list[float],
            - aspirate_speed: float = 5,
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
        """
        ACTION: concurrent_dispense
        DESCRIPTION: dispense liquid from a list of dispenser_container to a list of ChemicalContainer
        PARAMS:
            - actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            - destination_containers: list[ChemicalContainer],
            - dispenser_containers: list[TecanZ1Needle],
            - amounts: list[float],
            - dispense_speed: float = 5,
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
    can_contain: list[str] = [TecanPlate.__name__, ]

    @property
    def attachment(self) -> TecanPlate | None:
        if self.slot_content['SLOT'] is None:
            return None
        return TECAN_LAB[self.slot_content['SLOT']]

    def action__pick_up_plate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            thing: TecanPlate,
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: pick_up
        DESCRIPTION: pick up an attachment or a sv vial or a rack or a pdp tip
        PARAMS:
            - thing: TecanSvt | TecanPdp | TecanVpg | TecanVial | TecanPdpTip | TecanRack
        """

        if actor_type == 'pre':
            thing_slot = TECAN_LAB[thing.contained_by]
            # TODO merge this with "move_to"
            if thing_slot.identifier != self.position_on_top_of:
                raise PreActError(
                    f"you are picking up from: {thing_slot.identifier} but the arm is on top of: {self.position_on_top_of}")
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
        """
        ACTION: put_down
        DESCRIPTION: put down an attachment or a sv vial or a rack or a pdp tip
        PARAMS:
            - dest_slot: TecanSlot | TecanTipDisposal
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
