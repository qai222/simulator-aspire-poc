from __future__ import annotations

from hardware_pydantic.devices import Heater
from hardware_pydantic.junior.junior_objects import *

_SLOT_SIZE_X = 80
_SLOT_SIZE_Y = 120
_SMALL_SLOT_SIZE_X = 40
_SMALL_SLOT_SIZE_Y = 20


class JuniorDevice(Device):

    def model_post_init(self, *args) -> None:  # this should be better than `add_to_junior_lab`, you need pydantic 2.x
        JUNIOR_LAB.add_object(self)


class JuniorSlot(JuniorDevice, Heater):
    """ I'm using `Device` here because some slots function as `Balance` or `Heater` """

    can_weigh: bool = False

    can_hold: str | None
    """ what `JuniorObject` it can hold? used for typing checking """

    can_heat: bool = False

    can_cool: bool = False

    can_stir: bool = False

    content: str | None = None
    """ the identifier of the object it currently holds """

    # layout related

    layout_position: tuple[float, float] | None = None
    """ left bot """

    layout_x: float = _SLOT_SIZE_X

    layout_y: float = _SLOT_SIZE_Y

    def pre__wait(self, wait_time: float):
        """ wait for certain amount of time """
        # TODO should define a `container` superclass that can get its children recursively
        objs = []
        if self.content is None:
            return [], wait_time
        slot_content = JUNIOR_LAB[self.content]
        if isinstance(slot_content, JuniorRack):
            for k, v in slot_content.content.items():
                if v is not None:
                    objs.append(v)
        return objs, wait_time

    def post__wait(self, wait_time: float):
        return

    @staticmethod
    # @add_to_junior_lab
    def create_slot(
            identifier: str,
            layout_relation: Literal["above", "right_to"] = None,
            layout_relative: JuniorSlot = None,
            layout_x: float = _SLOT_SIZE_X, layout_y: float = _SLOT_SIZE_Y,
            can_cool=False, can_heat=False, can_stir=False, can_hold: str | None = JuniorRack.__name__, can_weigh=False,
            content: str = None
    ) -> JuniorSlot:
        if layout_relative is None:
            abs_layout_position = (0, 0)
        else:
            if layout_relation == "above":
                abs_layout_position = (
                    layout_relative.layout_position[0],
                    layout_relative.layout_position[1] + layout_relative.layout_y + 20
                )
            elif layout_relation == "right_to":
                abs_layout_position = (
                    layout_relative.layout_position[0] + layout_relative.layout_x + 20,
                    layout_relative.layout_position[1],
                )
            else:
                raise ValueError
        slot = JuniorSlot(
            identifier=identifier,
            can_hold=can_hold, can_cool=can_cool, can_heat=can_heat, can_stir=can_stir, can_weigh=can_weigh,
            layout_position=abs_layout_position, layout_x=layout_x, layout_y=layout_y,
            content=content,
        )
        return slot


class JuniorArm(JuniorDevice):
    position_on_top_of: str
    """ the current position, can only be a slot (not vial) """

    # # not useful fn
    # moving_to: str | None = None
    # """ where am I going? """

    can_access: list[str] = []
    """ a list of slot identifiers that this arm can access """

    def pre__move_to(self, move_to_slot: JuniorSlot):
        if move_to_slot.identifier not in self.can_access:
            raise PreActError
        return [move_to_slot, ], 5

    def post__move_to(self, move_to_slot: JuniorSlot):
        self.position_on_top_of = move_to_slot.identifier


class JuniorArmZ1(JuniorArm):

    # needle_content: dict[str, dict[str, float]] = {str(l + 1): dict() for l in range(7)}
    # """ liquid composition of needles """
    #
    # needle_capacities: dict[str, float] = {str(l + 1): float('inf') for l in range(7)}

    def pre__transfer_liquid(
            self,
            use_needles: list[str],
            from_vials: list[JuniorVial],
            to_vials: list[JuniorVial],
            amounts: list[float]
    ) -> tuple[list[LabObject], float]:
        from_rack_id = list(set([v.position_relative for v in from_vials]))[0]
        from_rack = JUNIOR_LAB.dict_object[from_rack_id]
        from_rack: JuniorRack
        from_slot = JUNIOR_LAB.dict_object[from_rack.position]

        if from_slot.identifier not in self.can_access:
            raise PreActError
        if self.position_on_top_of == from_slot.identifier:
            move_cost_1 = 0
        else:
            move_cost_1 = 5

        # TODO check empty needles
        # TODO check amount capacities
        # TODO check to_vials overflow
        # TODO check concurrency
        # TODO sample from distributions
        aspirate_speed = 5
        dispense_speed = 5
        move_cost_2 = 5
        return from_vials + to_vials, max(amounts) / aspirate_speed + max(
            amounts) / dispense_speed + move_cost_1 + move_cost_2

    def post__transfer_liquid(self, use_needles: list[str], from_vials: list[JuniorVial], to_vials: list[JuniorVial],
                              amounts: list[float]):
        for from_vial, to_vial, amount in zip(from_vials, to_vials, amounts):
            removed = from_vial.remove_content(amount)
            to_vial.add_content(removed)
            to_rack_id = list(set([v.position_relative for v in to_vials]))[0]
            to_rack = JUNIOR_LAB.dict_object[to_rack_id]
            to_rack: JuniorRack
            to_slot = JUNIOR_LAB.dict_object[to_rack.position]
            self.position_on_top_of = to_slot.identifier  # assuming vial always held in a rack

    # TODO action wash
    # TODO split action transfer into aspirate, move, dispense


class JuniorArmZ2(JuniorArm):
    attached_head: str | None = None
    """ various attachments, at most one at a time """

    @property
    def head(self) -> None | JuniorSvTool | JuniorVPG | JuniorPDT:
        if self.attached_head is None:
            return None
        return JUNIOR_LAB[self.attached_head]

    def pre__pick_up(self, obj: JuniorZ2Attachment | JuniorVial | JuniorRack):
        """ it can pick up 1. various heads 2. sv vial 3. rack """
        if self.attached_head is not None:
            if isinstance(obj, JuniorVial) and isinstance(JUNIOR_LAB[self.attached_head], JuniorSvTool):
                pass
            else:
                raise PreActError

        cost = 1

        if isinstance(obj, JuniorVial):
            slot_id = obj.position_relative
        else:
            slot_id = obj.position
        slot = JUNIOR_LAB[slot_id]

        if not isinstance(slot, JuniorSlot):
            raise PreActError

        if self.position_on_top_of != slot_id:
            cost += 5

        if slot_id not in self.can_access:
            raise PreActError

        if isinstance(obj, JuniorVial):
            if obj.type != "SV":
                raise PreActError
            if not isinstance(JUNIOR_LAB[self.attached_head], JuniorSvTool):
                raise PreActError
            if JUNIOR_LAB[self.attached_head].vial_connected_to is not None:
                raise PreActError

        if isinstance(obj, JuniorRack):
            if not isinstance(JUNIOR_LAB[self.attached_head], JuniorVPG):
                raise PreActError

        return [obj, slot], cost

    def post__pick_up(self, obj: JuniorZ2Attachment | JuniorVial | JuniorRack):
        if isinstance(obj, JuniorVial):
            slot_id = obj.position_relative
        else:
            slot_id = obj.position
        slot = JUNIOR_LAB[slot_id]
        slot: JuniorSlot
        if isinstance(obj, JuniorZ2Attachment):
            self.attached_head = obj.identifier
        elif isinstance(obj, JuniorVial):
            head = JUNIOR_LAB[self.attached_head]
            head: JuniorSvTool
            head.vial_connected_to = obj.identifier
        elif isinstance(obj, JuniorRack):
            head = JUNIOR_LAB[self.attached_head]
            head: JuniorVPG
            head.holding_rack = obj
        else:
            raise TypeError
        slot.content = None
        if isinstance(obj, JuniorVial):
            obj.position = None
            obj.position_relative = self.identifier
        else:
            obj.position = self.identifier
        self.position_on_top_of = slot_id

    def pre__dispense_sv(self, to_vial: JuniorVial, amount: float):

        sv_head_id = self.attached_head
        sv_head = JUNIOR_LAB[sv_head_id]
        sv_head: JuniorSvTool
        from_vial = JUNIOR_LAB[sv_head.vial_connected_to]

        move_cost = 0

        rack_holding_vial = JUNIOR_LAB[to_vial.position_relative]
        rack_holding_vial: JuniorRack

        if self.position_on_top_of != rack_holding_vial.position:
            move_cost += 5

        if not isinstance(JUNIOR_LAB[self.attached_head], JuniorZ2Attachment):
            raise PreActError
        if from_vial.content_sum < 1e-5:
            raise PreActError
        if amount > from_vial.content_sum:
            raise PreActError
        # TODO sample from distributions
        # TODO "powder param" fit time cost
        dispense_speed = 5
        return [sv_head, from_vial, to_vial], amount / dispense_speed + move_cost

    def post__dispense_sv(self, to_vial: JuniorVial, amount: float):
        sv_head_id = self.attached_head
        sv_head = JUNIOR_LAB[sv_head_id]
        sv_head: JuniorSvTool
        from_vial = JUNIOR_LAB[sv_head.vial_connected_to]

        rack_holding_vial = JUNIOR_LAB[to_vial.position_relative]
        rack_holding_vial: JuniorRack

        if self.position_on_top_of != rack_holding_vial.position:
            self.position_on_top_of = rack_holding_vial.position

        removed = from_vial.remove_content(amount)
        to_vial.add_content(removed)

    def pre__put_down(self, to_slot: JuniorSlot):
        """ it can put down 1. various heads 2. sv vial 3. rack """
        if self.attached_head is None:
            raise PreActError

        if to_slot.content is not None:
            raise PreActError

        if to_slot.identifier not in self.can_access:
            raise PreActError

        cost = 0
        objs = [to_slot, ]

        if isinstance(self.head, JuniorSvTool):
            if self.head.vial_connected_to is not None and to_slot.can_hold == JuniorVial.__name__:
                cost += 5
                objs.append(JUNIOR_LAB[self.head.vial_connected_to])
            elif self.head.vial_connected_to is None and to_slot.can_hold == JuniorSvTool.__name__:
                cost += 5
                objs.append(self.head)
            else:
                raise PreActError

        elif isinstance(self.head, JuniorZ2Attachment):
            cost += 5
            objs.append(JUNIOR_LAB[self.head])

        else:
            raise PreActError

        # TODO check PDT
        # TODO check VPG
        return objs, cost

    def post__put_down(self, to_slot: JuniorSlot):
        """ it can put down 1. various heads 2. sv vial 3. rack """

        self.position_on_top_of = to_slot.identifier

        if isinstance(self.head, JuniorSvTool):
            if self.head.vial_connected_to is not None and to_slot.can_hold == JuniorVial.__name__:
                to_slot.content = self.head.vial_connected_to
                self.head.vial_connected_to = None
            elif self.head.vial_connected_to is None and to_slot.can_hold == JuniorSvTool.__name__:
                to_slot.content = self.head.identifier
                self.attached_head = None
            else:
                raise PostActError

        elif isinstance(self.head, JuniorZ2Attachment):
            to_slot.content = self.head.identifier
            self.attached_head = None

        else:
            raise PostActError

    def pre__transfer_liquid_pdt(self, from_vial: JuniorVial, to_vial: JuniorVial, amount: float, ):

        from_rack_id = from_vial.position_relative
        from_rack = JUNIOR_LAB.dict_object[from_rack_id]
        from_rack: JuniorRack
        from_slot = JUNIOR_LAB.dict_object[from_rack.position]

        if from_slot.identifier not in self.can_access:
            raise PreActError
        if self.position_on_top_of == from_slot.identifier:
            move_cost_1 = 0
        else:
            move_cost_1 = 5

        if not isinstance(self.head, JuniorPDT):
            raise PreActError

        # TODO check this for all transfer
        to_rack_id = to_vial.position_relative
        to_rack = JUNIOR_LAB[to_rack_id]
        to_rack: JuniorRack
        to_slot = JUNIOR_LAB[to_rack.position]
        if not isinstance(to_slot, JuniorSlot):
            raise PreActError

        if JUNIOR_LAB['Z1 ARM'].position_on_top_of == to_rack.position:
            raise PreActError

        aspirate_speed = 5
        dispense_speed = 5
        move_cost_2 = 5
        return [self.head, from_vial,
                to_vial], amount / aspirate_speed + amount / dispense_speed + move_cost_1 + move_cost_2

    def post__transfer_liquid_pdt(self, from_vial: JuniorVial, to_vial: JuniorVial, amount: float, ):
        removed = from_vial.remove_content(amount)
        to_vial.add_content(removed)
        to_rack_id = from_vial.position_relative
        to_rack = JUNIOR_LAB.dict_object[to_rack_id]
        to_rack: JuniorRack
        to_slot = JUNIOR_LAB.dict_object[to_rack.position]
        self.position_on_top_of = to_slot.identifier  # assuming vial always held in a rack

    # TODO action wash
    # TODO split action transfer into aspirate, move, dispense
