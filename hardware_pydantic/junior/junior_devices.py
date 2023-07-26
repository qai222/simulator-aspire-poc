from __future__ import annotations

from hardware_pydantic.base import Device, DEVICE_ACTION_METHOD_ACTOR_TYPE, PreActError
from hardware_pydantic.junior.junior_base_devices import JuniorBaseHeater, JuniorBaseStirrer, JuniorBaseLiquidDispenser
from hardware_pydantic.junior.junior_objects import JuniorRack, JuniorZ1Needle, JuniorWashBay, JuniorSvt, JuniorPdp, \
    JuniorVpg, JuniorVial, JuniorPdpTip, JuniorTipDisposal
from hardware_pydantic.junior.settings import *
from hardware_pydantic.lab_objects import LabContainer, LabContainee, ChemicalContainer


class JuniorSlot(JuniorBaseHeater, JuniorBaseStirrer):
    """
    a vial or plate slot, does not include wash bay and tip disposal
    subclassing `Device` here because some slots function as `Balance` or `Heater`
    """

    can_weigh: bool = False
    """ is this a balance? """

    can_heat: bool = False
    """ can it heat? """

    can_cool: bool = False
    """ can it coll? """

    can_stir: bool = False
    """ can it stir? """

    layout: JuniorLayout | None = None

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
            containees = self.get_all_containees(container=self, lab=JUNIOR_LAB)
            return [JUNIOR_LAB[i] for i in containees], wait_time
        else:
            raise ValueError

    @staticmethod
    def put_rack_in_a_slot(rack: JuniorRack, slot: JuniorSlot):
        if rack.contained_by is not None:
            prev_slot = JUNIOR_LAB[rack.contained_by]
            assert isinstance(prev_slot, JuniorSlot)
            prev_slot.slot_content["SLOT"] = None
        assert rack.__class__.__name__ in slot.can_contain
        rack.contained_by = slot.identifier
        rack.contained_in_slot = "SLOT"
        slot.slot_content["SLOT"] = rack.identifier


class JuniorArmPlatform(Device, LabContainer, JuniorLabObject):
    position_on_top_of: str | None = None
    """ the current position, can only be a slot (not vial) """

    anchor_arm: str | None = None
    """ which arm is used to define xy position? """

    def action__move_to(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            anchor_arm: JuniorArmZ1 | JuniorArmZ2,
            move_to_slot: JuniorSlot,
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
            if anchor_arm.identifier not in self.get_all_containees(self, JUNIOR_LAB):
                raise PreActError
        elif actor_type == 'post':
            self.position_on_top_of = move_to_slot.identifier
            self.anchor_arm = anchor_arm.identifier
        elif actor_type == 'proj':
            containees = self.get_all_containees(container=self, lab=JUNIOR_LAB)
            return [JUNIOR_LAB[i] for i in containees], move_cost
        else:
            raise ValueError


class JuniorArmZ1(LabContainer, LabContainee, JuniorBaseLiquidDispenser):
    allowed_concurrency: list[int] = [1, 4, 6]

    slot_content: dict[str, str] = dict()

    @property
    def arm_platform(self) -> JuniorArmPlatform:
        return JUNIOR_LAB[self.contained_by]

    def action__concurrent_aspirate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            source_containers: list[ChemicalContainer],
            dispenser_containers: list[JuniorZ1Needle],
            amounts: list[float],
            aspirate_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: concurrent_aspirate
        DESCRIPTION: aspirate liquid from a list of ChemicalContainer to a list of dispenser_container
        PARAMS:
            - actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            - source_containers: list[ChemicalContainer],
            - dispenser_containers: list[ChemicalContainer],
            - amounts: list[float],
            - aspirate_speed: float = 5,
        """
        if actor_type == 'pre':
            if len(set([JUNIOR_LAB[sc.contained_by] for sc in source_containers])) != 1:
                raise PreActError
            if not len(source_containers) == len(dispenser_containers) == len(amounts):
                raise PreActError
            if len(source_containers) not in self.allowed_concurrency:
                raise PreActError
        objs = []
        times = []
        for s, d, a in zip(source_containers, dispenser_containers, amounts):
            res = self.action__aspirate(actor_type=actor_type, source_container=s, dispenser_container=d, amount=a,
                                        aspirate_speed=aspirate_speed)
            if actor_type == 'proj':
                objs += res[0]
                times.append(res[1])
        if actor_type == 'proj':
            return objs, max(times)

    def action__concurrent_dispense(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            destination_containers: list[ChemicalContainer],
            dispenser_containers: list[JuniorZ1Needle],
            amounts: list[float],
            dispense_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: concurrent_dispense
        DESCRIPTION: dispense liquid from a list of dispenser_container to a list of ChemicalContainer
        PARAMS:
            - actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            - destination_containers: list[ChemicalContainer],
            - dispenser_containers: list[JuniorZ1Needle],
            - amounts: list[float],
            - dispense_speed: float = 5,
        """
        if actor_type == 'pre':
            if len(set([JUNIOR_LAB[dc.contained_by] for dc in destination_containers])) != 1:
                raise PreActError
            if not len(destination_containers) == len(dispenser_containers) == len(amounts):
                raise PreActError
            if len(destination_containers) not in self.allowed_concurrency:
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

    def action__wash(self,
                     actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
                     wash_bay: JuniorWashBay,
                     wash_volume: float = 10,
                     flush_volume: float = 10,
                     ):
        """The unit of wash_volume and flush_volume is mL."""
        if actor_type == 'pre':
            if JUNIOR_LAB[self.contained_by].position_on_top_of != wash_bay.identifier:
                raise PreActError
        elif actor_type == 'post':
            for n in self.slot_content.values():
                JUNIOR_LAB[n].chemical_content = dict()
        elif actor_type == 'proj':
            containees = self.get_all_containees(container=self, lab=JUNIOR_LAB)
            return [JUNIOR_LAB[i] for i in containees], 6.0270*wash_volume + 32.0000*flush_volume


class JuniorArmZ2(LabContainer, LabContainee, JuniorBaseLiquidDispenser):

    @property
    def arm_platform(self) -> JuniorArmPlatform:
        return JUNIOR_LAB[self.contained_by]

    @property
    def attachment(self) -> None | JuniorSvt | JuniorVpg | JuniorPdp:
        if self.slot_content['SLOT'] is None:
            return None
        return JUNIOR_LAB[self.slot_content['SLOT']]

    def action__pick_up(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            thing: JuniorSvt | JuniorPdp | JuniorVpg | JuniorVial | JuniorPdpTip | JuniorRack,
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: pick_up
        DESCRIPTION: pick up an attachment or a sv vial or a rack or a pdp tip
        PARAMS:
            - thing: JuniorSvt | JuniorPdp | JuniorVpg | JuniorVial | JuniorPdpTip | JuniorRack
        """

        if actor_type == 'pre':

            thing_slot = LabContainee.get_container(thing, JUNIOR_LAB, upto=JuniorSlot)
            # TODO merge this with "ArmPlatform.move_to"
            if thing_slot.identifier != self.arm_platform.position_on_top_of:
                raise PreActError(f"you are picking up from: {thing_slot.identifier} but the arm is on top of: {self.arm_platform.position_on_top_of}")

            if isinstance(thing, (JuniorSvt, JuniorPdp, JuniorVpg)):
                if self.attachment is not None:
                    raise PreActError(f"you are picking up: {thing.__class__.__name__} but the current attachment is: {self.attachment.__class__.__name__}")
            else:
                if isinstance(thing, JuniorVial) and not isinstance(self.attachment, JuniorSvt):
                    raise PreActError
                elif isinstance(thing, JuniorPdpTip) and not isinstance(self.attachment, JuniorPdp):
                    raise PreActError
                elif isinstance(thing, JuniorRack) and not isinstance(self.attachment, JuniorVpg):
                    raise PreActError
        elif actor_type == 'post':
            if isinstance(thing, (JuniorSvt, JuniorPdp, JuniorVpg)):
                LabContainee.move(containee=thing, dest_container=self, lab=JUNIOR_LAB, dest_slot="SLOT")
            else:
                LabContainee.move(containee=thing, dest_container=self.attachment, lab=JUNIOR_LAB, dest_slot="SLOT")
        elif actor_type == 'proj':
            # putting down SV Powder Dispense Tool
            if isinstance(thing, JuniorSvt):
                pickup_cost = 29
            elif isinstance(thing, JuniorPdp):
                pickup_cost = 5.9231
            else:
                pickup_cost = 10
            if isinstance(thing, (JuniorSvt, JuniorPdp, JuniorVpg)):
                return [thing, ], pickup_cost
            else:
                return [thing, self.attachment, ], pickup_cost
        else:
            raise ValueError

    def action__put_down(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            dest_slot: JuniorSlot | JuniorTipDisposal
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: put_down
        DESCRIPTION: put down an attachment or a sv vial or a rack or a pdp tip
        PARAMS:
            - dest_slot: JuniorSlot | JuniorTipDisposal
        """

        if self.attachment.slot_content['SLOT'] is not None:
            thing = JUNIOR_LAB[self.attachment.slot_content['SLOT']]
            objs = [thing, self.attachment, dest_slot]
        else:
            thing = self.attachment
            objs = [thing, dest_slot]

        if actor_type == 'pre':
            if dest_slot.identifier != self.arm_platform.position_on_top_of:
                raise PreActError
            if not isinstance(dest_slot, JuniorTipDisposal):
                if dest_slot.slot_content['SLOT'] is not None:
                    raise PreActError
        elif actor_type == 'post':
            if not isinstance(dest_slot, JuniorTipDisposal):
                LabContainee.move(containee=thing, dest_container=dest_slot, lab=JUNIOR_LAB, dest_slot="SLOT")
                if isinstance(thing, JuniorVial):
                    self.attachment.powder_param_known = False
            else:
                thing_container = JUNIOR_LAB[thing.contained_by]
                thing_container: LabContainer
                assert thing_container.slot_content[thing.contained_in_slot] == thing.identifier
                thing_container.slot_content[thing.contained_in_slot] = None
                dest_slot.disposal_content.append(thing.identifier)
                thing.contained_by = None
                thing.contained_in_slot = None  # disposal doesn't have slot labels
        elif actor_type == 'proj':
            # putting down SV Powder Dispense Tool
            if isinstance(thing, JuniorSvt):
                put_down_cost = 35
            # putting down PDP
            elif isinstance(thing, JuniorPdp):
                put_down_cost = 12
            else:
                put_down_cost = 7

            return objs, put_down_cost
        else:
            raise ValueError

    def action__aspirate_pdp(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            source_container: ChemicalContainer,
            amount: float,
            aspirate_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        source_slot = LabContainee.get_container(source_container, JUNIOR_LAB, upto=JuniorSlot)
        if actor_type == 'pre':
            if source_slot.identifier != self.arm_platform.position_on_top_of:
                raise PreActError
            if not isinstance(self.attachment, JuniorPdp):
                raise PreActError
            if not isinstance(JUNIOR_LAB[self.attachment.slot_content['SLOT']], JuniorPdpTip):
                raise PreActError
        return self.action__aspirate(
            actor_type=actor_type, source_container=source_container,
            dispenser_container=JUNIOR_LAB[self.attachment.slot_content['SLOT']],
            amount=amount, aspirate_speed=aspirate_speed,
        )

    def action__dispense_pdp(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            destination_container: ChemicalContainer,
            amount: float,
            dispense_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        dest_slot = LabContainee.get_container(destination_container, JUNIOR_LAB, upto=JuniorSlot)
        if actor_type == 'pre':
            if dest_slot.identifier != self.arm_platform.position_on_top_of:
                raise PreActError
            if not isinstance(self.attachment, JuniorPdp):
                raise PreActError
            if not isinstance(JUNIOR_LAB[self.attachment.slot_content['SLOT']], JuniorPdpTip):
                raise PreActError
        return self.action__dispense(
            actor_type=actor_type, destination_container=destination_container,
            dispenser_container=JUNIOR_LAB[self.attachment.slot_content['SLOT']],
            amount=amount, dispense_speed=dispense_speed
        )

    def action__dispense_sv(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            destination_container: ChemicalContainer,
            amount: float,
            dispense_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        dest_slot = LabContainee.get_container(destination_container, JUNIOR_LAB, upto=JuniorSlot)
        if actor_type == 'pre':
            if dest_slot.identifier != self.arm_platform.position_on_top_of:
                raise PreActError(f"destination slot is: {dest_slot.identifier} but the arm is on top of: {self.arm_platform.position_on_top_of}")
            if not isinstance(self.attachment, JuniorSvt):
                raise PreActError
            if not isinstance(JUNIOR_LAB[self.attachment.slot_content['SLOT']], JuniorVial):
                raise PreActError
        elif actor_type == 'proj':
            # 185.0, 236.0 seconds for 18mg and 55 mg respectively
            if self.attachment.powder_param_known:
                dispense_speed = dispense_speed * 10
            else:
                dispense_speed = dispense_speed
        elif actor_type == 'post':
            self.attachment.powder_param_known = True
        return self.action__dispense(
            actor_type=actor_type, destination_container=destination_container,
            dispenser_container=JUNIOR_LAB[self.attachment.slot_content['SLOT']],
            amount=amount, dispense_speed=dispense_speed
        )


if __name__ == '__main__':
    slot = JuniorSlot(identifier="slot1", can_contain=[JuniorRack.__name__])
    jr, vials = JuniorRack.create_rack_with_empty_vials()
    JuniorSlot.put_rack_in_a_slot(jr, slot)
    # print(LabContainer.get_all_containees(slot, JUNIOR_LAB))
    # print(JUNIOR_LAB)
    print(LabContainee.get_container(jr, JUNIOR_LAB, upto=JuniorSlot))
