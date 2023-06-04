from hardware_pydantic.junior.junior_devices import *
from hardware_pydantic.junior.junior_devices import _SLOT_SIZE_Y, _SMALL_SLOT_SIZE_X, _SMALL_SLOT_SIZE_Y, _SLOT_SIZE_X


class JuniorInstruction(Instruction):

    def model_post_init(self, *args) -> None:
        JUNIOR_LAB.add_instruction(self)


def create_junior_base():
    assert len(JUNIOR_LAB.dict_object) == 0, "JUNIOR BASE HAS ALREADY BEEN INIT!!!"

    RACK_SLOT_OFF_DECK_1 = JuniorSlot.create_slot(identifier="RACK SLOT OFF-1")
    RACK_SLOT_OFF_DECK_2 = JuniorSlot.create_slot(identifier="RACK SLOT OFF-2", layout_relation="above",
                                                  layout_relative=RACK_SLOT_OFF_DECK_1)
    RACK_SLOT_OFF_DECK_3 = JuniorSlot.create_slot(identifier="RACK SLOT OFF-3", layout_relation="above",
                                                  layout_relative=RACK_SLOT_OFF_DECK_2)

    # WASH_BAY = JuniorSlot.create_slot(identifier="WASH BAY", layout_relation="right_to",
    #                                   layout_relative=RACK_SLOT_OFF_DECK_1, layout_x=_SMALL_SLOT_SIZE_X,
    #                                   layout_y=_SLOT_SIZE_Y * 3, can_hold=None)

    RACK_SLOT_2_3_1 = JuniorSlot.create_slot(identifier="RACK SLOT 1", layout_relation="right_to",
                                             # layout_relative=WASH_BAY,
                                             layout_relative=RACK_SLOT_OFF_DECK_1,
                                             can_cool=True, can_heat=True, can_stir=True
                                             )
    RACK_SLOT_2_3_2 = JuniorSlot.create_slot(identifier="RACK SLOT 2", layout_relation="above",
                                             layout_relative=RACK_SLOT_2_3_1, can_cool=True, can_heat=True,
                                             can_stir=True)
    RACK_SLOT_2_3_3 = JuniorSlot.create_slot(identifier="RACK SLOT 3", layout_relation="above",
                                             layout_relative=RACK_SLOT_2_3_2, can_cool=True, can_heat=True,
                                             can_stir=True)

    PDT_SLOT_1 = JuniorSlot.create_slot(
        identifier="PDT SLOT 1", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
        layout_relation="right_to", layout_relative=RACK_SLOT_2_3_1, can_hold=JuniorPDT.__name__
    )
    PDT_SLOT_2 = JuniorSlot.create_slot(
        identifier="PDT SLOT 2", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
        layout_relation="above", layout_relative=PDT_SLOT_1, can_hold=JuniorPDT.__name__
    )
    PDT_SLOT_3 = JuniorSlot.create_slot(
        identifier="PDT SLOT 3", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
        layout_relation="above", layout_relative=PDT_SLOT_2, can_hold=JuniorPDT.__name__
    )

    SV_VIAL_SLOTS = []

    num_sv_vial_per_row = 3

    for i in range(12):
        irow = i // num_sv_vial_per_row
        icol = i % num_sv_vial_per_row
        if i == 0:
            sv_vial_slot = JuniorSlot.create_slot(
                identifier=f"SVV SLOT {i + 1}", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
                layout_relation="above", layout_relative=PDT_SLOT_3, can_hold=JuniorVial.__name__
            )
        else:
            last_slot = SV_VIAL_SLOTS[i - 1]
            if irow == (i - 1) // num_sv_vial_per_row:
                relation = "right_to"
                relative = last_slot
            else:
                relation = "above"
                relative = SV_VIAL_SLOTS[i - num_sv_vial_per_row]
            sv_vial_slot = JuniorSlot.create_slot(
                identifier=f"SVV SLOT {i + 1}", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
                layout_relation=relation, layout_relative=relative, can_hold=JuniorVial.__name__
            )
        SV_VIAL_SLOTS.append(sv_vial_slot)

    SV_TOOL_SLOT = JuniorSlot.create_slot(
        identifier="SV TOOL SLOT", layout_x=_SLOT_SIZE_X, layout_y=_SLOT_SIZE_Y,
        layout_relation="above", layout_relative=SV_VIAL_SLOTS[9], can_hold=JuniorSvTool.__name__
    )

    BALANCE = JuniorSlot.create_slot(
        identifier="BALANCE SLOT", layout_x=_SLOT_SIZE_X, layout_y=_SLOT_SIZE_Y,
        layout_relation="right_to", layout_relative=SV_TOOL_SLOT, can_hold=JuniorVial.__name__, can_weigh=True
    )

    VPG_SLOT = JuniorSlot.create_slot(
        identifier="VPG SLOT", layout_x=_SLOT_SIZE_X, layout_y=_SLOT_SIZE_Y,
        layout_relation="right_to", layout_relative=BALANCE, can_hold=JuniorVPG.__name__,
    )

    # TIP_DISPOSAL = JuniorSlot.create_slot(identifier="DISPOSAL", layout_relation="right_to",
    #                                       layout_relative=VPG_SLOT, layout_x=_SMALL_SLOT_SIZE_X,
    #                                       layout_y=_SLOT_SIZE_Y, can_hold=None)

    # Z1_BAY = JuniorSlot.create_slot(
    #     identifier="Z1 BAY", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
    #     layout_relation="above", layout_relative=RACK_SLOT_OFF_DECK_3, can_hold=JuniorArmZ1.__name__
    # )
    #
    # Z2_BAY = JuniorSlot.create_slot(
    #     identifier="Z2 BAY", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
    #     layout_relation="right_to", layout_relative=Z1_BAY, can_hold=JuniorArmZ2.__name__
    # )

    Z1_ARM = JuniorArmZ1(
        identifier="Z1 ARM",
        # position_on_top_of=Z1_BAY.identifier,
        position_on_top_of=RACK_SLOT_OFF_DECK_1.identifier,
        can_access=[
            s.identifier for s in
            [RACK_SLOT_OFF_DECK_1, RACK_SLOT_OFF_DECK_2, RACK_SLOT_OFF_DECK_3, RACK_SLOT_2_3_1, RACK_SLOT_2_3_2,
             RACK_SLOT_2_3_3]
        ]
    )
    # Z1_BAY.content = Z1_ARM.identifier

    Z2_ARM = JuniorArmZ2(
        identifier="Z2 ARM",
        # position_on_top_of=Z2_BAY.identifier,
        position_on_top_of=RACK_SLOT_2_3_1.identifier,
        can_access=[
            s.identifier for s in
            [RACK_SLOT_2_3_1, RACK_SLOT_2_3_2, RACK_SLOT_2_3_3, SV_TOOL_SLOT, VPG_SLOT, PDT_SLOT_1, PDT_SLOT_2,
             PDT_SLOT_3, BALANCE] + SV_VIAL_SLOTS
        ]
    )
    # Z2_BAY.content = Z1_ARM.identifier

    SV_TOOL = JuniorSvTool(identifier="SV TOOL", position=SV_TOOL_SLOT.identifier, vial_connected_to=None)
    SV_TOOL_SLOT.content = SV_TOOL.identifier

    VPG = JuniorVPG(identifier="VPG", position=VPG_SLOT.identifier, holding_rack=None)
    VPG_SLOT.content = VPG.identifier

    PDT_1 = JuniorPDT(identifier="PDT 1", position=PDT_SLOT_1.identifier)
    PDT_2 = JuniorPDT(identifier="PDT 2", position=PDT_SLOT_2.identifier)
    PDT_3 = JuniorPDT(identifier="PDT 3", position=PDT_SLOT_3.identifier)

    PDT_SLOT_1.content = PDT_1.identifier
    PDT_SLOT_2.content = PDT_2.identifier
    PDT_SLOT_3.content = PDT_3.identifier

    return JUNIOR_LAB
