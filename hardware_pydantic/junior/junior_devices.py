from __future__ import annotations

from hardware_pydantic.base import Device, DEVICE_ACTION_METHOD_ACTOR_TYPE, PreActError
from hardware_pydantic.junior.junior_base_devices import JuniorBaseHeater, JuniorBaseStirrer, \
    JuniorBaseLiquidDispenser
from hardware_pydantic.junior.junior_objects import JuniorRack, JuniorZ1Needle, JuniorWashBay, \
    JuniorSvt, JuniorPdp, \
    JuniorVpg, JuniorVial, JuniorPdpTip, JuniorTipDisposal
from hardware_pydantic.junior.settings import *
from hardware_pydantic.lab_objects import LabContainer, LabContainee, ChemicalContainer
from hardware_pydantic.junior.utils import running_time_washing


"""Devices on the Junior platform at NCATS."""


class JuniorSlot(JuniorBaseHeater, JuniorBaseStirrer):
    """A slot on the Junior platform at NCATS.

    Parameters
    ----------
    can_weigh : bool
        Whether this slot can weigh or not. Defaults to False.
    can_heat : bool
        Whether this slot can heat or not. Defaults to False.
    can_cool : bool
        Whether this slot can cool or not. Defaults to False.
    can_stir : bool
        Whether this slot can stir or not. Defaults to False.
    layout : JuniorLayout, optional
        The layout of this slot, by default None.

    Notes
    -----
    A vial or plate slot, does not include wash bay and tip disposal. We subclass the
    `Device` here because some slots function as `Balance` or `Heater`.

    """

    can_weigh: bool = False
    can_heat: bool = False
    can_cool: bool = False
    can_stir: bool = False
    layout: JuniorLayout | None = None

    def action__wait(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
            wait_time: float = 0
    ) -> tuple[list[LabObject], float] | None:
        """The wait action for a slot which holds everything in this slot for a given period of
        time.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE, optional
            The actor type, one of 'pre', 'post', or 'proj'. Default is 'pre'.
        wait_time : float, optional
            The wait time. Default is 0.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of lab objects and the time cost.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and this slot cannot heat.
        ValueError
            If the actor type is not one of 'pre', 'post', or 'proj'.

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
        """Put a rack in a slot.

        Parameters
        ----------
        rack : JuniorRack
            The rack to put in a slot.
        slot : JuniorSlot
            The slot to put the rack in.

        """
        if rack.contained_by is not None:
            prev_slot = JUNIOR_LAB[rack.contained_by]
            assert isinstance(prev_slot, JuniorSlot)
            prev_slot.slot_content["SLOT"] = None
        assert rack.__class__.__name__ in slot.can_contain
        rack.contained_by = slot.identifier
        rack.contained_in_slot = "SLOT"
        slot.slot_content["SLOT"] = rack.identifier


class JuniorArmPlatform(Device, LabContainer, JuniorLabObject):
    """The arm platform on the Junior platform at NCATS.

    Parameters
    ----------
    position_on_top_of : str, optional
        The current position, can only be a slot (not vial). Default is None.
    anchor_arm : str, optional
        Which arm is used to define xy position. Default is None.

    """
    position_on_top_of: str | None = None
    anchor_arm: str | None = None

    def action__move_to(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            anchor_arm: JuniorArmZ1 | JuniorArmZ2,
            move_to_slot: JuniorSlot,
    ) -> tuple[list[LabObject], float] | None:
        """Action to move to a slot.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type, one of 'pre', 'post', or 'proj'.
        anchor_arm : JuniorArmZ1 | JuniorArmZ2
            Which arm is used to define xy position.
        move_to_slot : JuniorSlot
            The slot to move to.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of lab objects and the time cost.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and the anchor arm is not in the same slot as this arm.
        ValueError
            If the actor type is not one of 'pre', 'post', or 'proj'.
        """
        # it takes time zero to move to the same slot
        if self.position_on_top_of == move_to_slot.identifier:
            move_cost = 1e-6
        # it takes 5 seconds to move to a different slot
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
    """The Z1 arm on the Junior platform at NCATS.

    Parameters
    ----------
    allowed_concurrency : list[int], optional
        The allowed concurrency. Default is [1, 4, 6].
    slot_content : dict[str, str], optional
        The slot content. Default is empty dictionary.

    """
    allowed_concurrency: list[int] = [1, 4, 6]

    slot_content: dict[str, str] = dict()

    @property
    def arm_platform(self) -> JuniorArmPlatform:
        """The arm platform this arm is on.

        Returns
        -------
        JuniorArmPlatform
            The arm platform this arm is on.
        """
        return JUNIOR_LAB[self.contained_by]

    def action__concurrent_aspirate(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            source_containers: list[ChemicalContainer],
            dispenser_containers: list[JuniorZ1Needle],
            amounts: list[float],
    ) -> tuple[list[LabObject], float] | None:
        """The action to aspirate liquid from a list of ChemicalContainer to a list of
        dispenser_container.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type, one of 'pre', 'post', or 'proj'.
        source_containers : list[ChemicalContainer]
            The source containers.
        dispenser_containers : list[JuniorZ1Needle]
            The dispenser containers.
        amounts : list[float]
            The amounts to aspirate.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of lab objects and the time cost.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and it will error out for any of the following reasons:
            1. if two or more lab objects holding the container;
            2. if the number of source containers, dispenser containers, and amounts are not the
            same;
            3. if the number of source containers is not in the allowed concurrency list;

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
            res = self.action__aspirate(actor_type=actor_type,
                                        source_container=s,
                                        dispenser_container=d,
                                        amount=a,
                                        )
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
    ) -> tuple[list[LabObject], float] | None:
        """The action to dispense liquid from a list of dispenser_container to a list of
        ChemicalContainers.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type, one of 'pre', 'post', or 'proj'.
        destination_containers : list[ChemicalContainer]
            The destination containers.
        dispenser_containers : list[JuniorZ1Needle]
            The dispenser containers.
        amounts : list[float]
            The amounts to dispense.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of lab objects and the time cost.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and it will error out for any of the following reasons:
            1. if two or more lab objects holding the container;
            2. if the number of destination containers, dispenser containers, and amounts are not
            the same;
            3. if the number of destination containers is not in the allowed concurrency list;

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
            res = self.action__dispense(actor_type=actor_type,
                                        destination_container=s,
                                        dispenser_container=d,
                                        amount=a,
                                        )
            if actor_type == 'proj':
                objs += res[0]
                times.append(res[1])
        if actor_type == 'proj':
            return objs, max(times)

    def action__wash(self,
                     actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
                     wash_bay: JuniorWashBay,
                     wash_volume: float = 1,
                     flush_volume: float = 1,
                     ):
        """Wash action.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type, one of 'pre', 'post', or 'proj'.
        wash_bay : JuniorWashBay
            The wash bay.
        wash_volume : float, optional
            The wash volume. Default is 1 mL.
        flush_volume : float, optional
            The flush volume. Default is 1 mL.

        Returns
        -------
        list[ChemicalContainer], float
            The list of containers and the time cost.

        Raises
        ------
        PreActError
            If the action type is 'pre' and the position of the arm is not on top of the wash bay.

        Notes
        -----
        The unit of wash_volume and flush_volume is mL.

        """
        if actor_type == 'pre':
            if JUNIOR_LAB[self.contained_by].position_on_top_of != wash_bay.identifier:
                raise PreActError
        elif actor_type == 'post':
            for n in self.slot_content.values():
                JUNIOR_LAB[n].chemical_content = dict()
        elif actor_type == 'proj':
            containees = self.get_all_containees(container=self, lab=JUNIOR_LAB)
            return [JUNIOR_LAB[i] for i in
                    containees], running_time_washing(wash_volume, flush_volume)


class JuniorArmZ2(LabContainer, LabContainee, JuniorBaseLiquidDispenser):
    """The Z2 arm of the Junior liquid handler."""

    @property
    def arm_platform(self) -> JuniorArmPlatform:
        """The arm platform of Z2 arm.

        Returns
        -------
        JuniorArmPlatform
            The arm platform of Z2 arm.

        """
        return JUNIOR_LAB[self.contained_by]

    @property
    def attachment(self) -> None | JuniorSvt | JuniorVpg | JuniorPdp:
        """The attachment on the Z2 arm.

        Returns
        -------
        None | JuniorSvt | JuniorVpg | JuniorPdp
            The attachment on the Z2 arm.

        """
        if self.slot_content['SLOT'] is None:
            return None
        return JUNIOR_LAB[self.slot_content['SLOT']]

    def action__pick_up(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            thing: JuniorSvt | JuniorPdp | JuniorVpg | JuniorVial | JuniorPdpTip | JuniorRack,
    ) -> tuple[list[LabObject], float] | None:
        """The action of picking up an attachment or a sv vial or a rack or a pdp tip.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type, one of 'pre', 'post', or 'proj'.
        thing : JuniorSvt | JuniorPdp | JuniorVpg | JuniorVial | JuniorPdpTip | JuniorRack
            The thing to pick up.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of lab objects and the time cost.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and it will error out for any of the following reasons:
            1. if the arm is not on top of the thing;
            2. the `thing` is not a JuniorSvt, JuniorPdp, JuniorVpg, or JuniorVial and the current
            attachment is not None;
            3. if the `thing` is a JuniorVial but the current attachment is not a JuniorSvt;
            4. if the `thing` is a JuniorPdpTip but the current attachment is not a JuniorPdp;
            5. if the `thing` is a JuniorRack but the current attachment is not a JuniorVpg.
        ValueError
            If actor type is not one of 'pre', 'post', or 'proj'.

        """

        if actor_type == 'pre':

            thing_slot = LabContainee.get_container(thing, JUNIOR_LAB, upto=JuniorSlot)
            # TODO merge this with "ArmPlatform.move_to"
            if thing_slot.identifier != self.arm_platform.position_on_top_of:
                raise PreActError(
                    f"you are picking up from: {thing_slot.identifier} but the arm is on top of: "
                    f"{self.arm_platform.position_on_top_of}")

            if isinstance(thing, (JuniorSvt, JuniorPdp, JuniorVpg)):
                if self.attachment is not None:
                    raise PreActError(
                        f"you are picking up: {thing.__class__.__name__} but the current "
                        f"attachment is: {self.attachment.__class__.__name__}")
            else:
                if isinstance(thing, JuniorVial) and not isinstance(self.attachment, JuniorSvt):
                    raise PreActError
                elif isinstance(thing, JuniorPdpTip) and not isinstance(self.attachment, JuniorPdp):
                    raise PreActError
                elif isinstance(thing, JuniorRack) and not isinstance(self.attachment, JuniorVpg):
                    raise PreActError
        elif actor_type == 'post':
            if isinstance(thing, (JuniorSvt, JuniorPdp, JuniorVpg)):
                LabContainee.move(containee=thing, dest_container=self, lab=JUNIOR_LAB,
                                  dest_slot="SLOT")
            else:
                LabContainee.move(containee=thing,
                                  dest_container=self.attachment,
                                  lab=JUNIOR_LAB,
                                  dest_slot="SLOT")
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
        """The action of putting down an attachment or a sv vial or a rack or a pdp tip.

        Parameters
        ----------
        actor_type : DEVICE_ACTION_METHOD_ACTOR_TYPE
            The actor type, one of 'pre', 'post', or 'proj'.
        dest_slot : JuniorSlot | JuniorTipDisposal
            The destination slot.

        Returns
        -------
        tuple[list[LabObject], float] | None
            The list of lab objects and the time cost if the actor type is 'proj'.

        Raises
        ------
        PreActError
            If the actor type is 'pre' and it will error out for any of the following reasons:
            1. if the arm is not on top of the thing;
            2. When the destination slot is not the JuniorTipDisposal and the destination slot is
            not empty, it errors out.
        ValueError
            If actor type is not one of 'pre', 'post', or 'proj'.

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
                LabContainee.move(containee=thing, dest_container=dest_slot, lab=JUNIOR_LAB,
                                  dest_slot="SLOT")
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
            amount=amount, )

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
            amount=amount,
        )

    def action__dispense_sv(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            destination_container: ChemicalContainer,
            amount: float,
            dispense_speed: float = 5,
    ) -> tuple[list[LabObject], float] | None:
        dest_slot = LabContainee.get_container(destination_container, JUNIOR_LAB, upto=JuniorSlot)
        scaling_factor = 1.0

        if actor_type == 'pre':
            if dest_slot.identifier != self.arm_platform.position_on_top_of:
                raise PreActError(f"destination slot is: {dest_slot.identifier} but the arm is on "
                                  f"top of: {self.arm_platform.position_on_top_of}")
            if not isinstance(self.attachment, JuniorSvt):
                raise PreActError
            if not isinstance(JUNIOR_LAB[self.attachment.slot_content['SLOT']], JuniorVial):
                raise PreActError
        elif actor_type == 'proj':
            # 185.0, 236.0 seconds for 18mg and 55 mg respectively
            if self.attachment.powder_param_known:
                scaling_factor = scaling_factor / 10

        elif actor_type == 'post':
            self.attachment.powder_param_known = True
        return self.action__dispense(
            actor_type=actor_type, destination_container=destination_container,
            dispenser_container=JUNIOR_LAB[self.attachment.slot_content['SLOT']],
            amount=amount, scaling_factor=scaling_factor,
        )


if __name__ == '__main__':
    slot = JuniorSlot(identifier="slot1", can_contain=[JuniorRack.__name__])
    jr, vials = JuniorRack.create_rack_with_empty_vials()
    JuniorSlot.put_rack_in_a_slot(jr, slot)
    # print(LabContainer.get_all_containees(slot, JUNIOR_LAB))
    # print(JUNIOR_LAB)
    print(LabContainee.get_container(jr, JUNIOR_LAB, upto=JuniorSlot))
