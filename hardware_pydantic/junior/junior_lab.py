from hardware_pydantic.junior.junior_devices import *
from hardware_pydantic.junior.junior_devices import _SLOT_SIZE_Y, _SMALL_SLOT_SIZE_X, _SMALL_SLOT_SIZE_Y, _SLOT_SIZE_X


def create_junior_base():
    assert len(JUNIOR_LAB.dict_object) == 0, "JUNIOR BASE HAS ALREADY BEEN INIT!!!"

    RACK_SLOT_OFF_DECK_1 = JuniorSlot.create_slot(identifier="RACK_SLOT_OFF_DECK_1")
    RACK_SLOT_OFF_DECK_2 = JuniorSlot.create_slot(identifier="RACK_SLOT_OFF_DECK_2", layout_relation="above",
                                                  layout_relative=RACK_SLOT_OFF_DECK_1)
    RACK_SLOT_OFF_DECK_3 = JuniorSlot.create_slot(identifier="RACK_SLOT_OFF_DECK_3", layout_relation="above",
                                                  layout_relative=RACK_SLOT_OFF_DECK_2)

    WASH_BAY = JuniorSlot.create_slot(identifier="WASH_BAY", layout_relation="right_to",
                                      layout_relative=RACK_SLOT_OFF_DECK_1, layout_x=_SMALL_SLOT_SIZE_X,
                                      layout_y=_SLOT_SIZE_Y * 3, can_hold=None)

    RACK_SLOT_2_3_1 = JuniorSlot.create_slot(identifier="RACK_SLOT_2_3_1", layout_relation="right_to",
                                             layout_relative=WASH_BAY, can_cool=True, can_heat=True, can_stir=True)
    RACK_SLOT_2_3_2 = JuniorSlot.create_slot(identifier="RACK_SLOT_2_3_2", layout_relation="above",
                                             layout_relative=RACK_SLOT_2_3_1, can_cool=True, can_heat=True,
                                             can_stir=True)
    RACK_SLOT_2_3_3 = JuniorSlot.create_slot(identifier="RACK_SLOT_2_3_3", layout_relation="above",
                                             layout_relative=RACK_SLOT_2_3_2, can_cool=True, can_heat=True,
                                             can_stir=True)

    PDT_SLOT_1 = JuniorSlot.create_slot(
        identifier="PDT_SLOT_1", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
        layout_relation="right_to", layout_relative=RACK_SLOT_2_3_1, can_hold=JuniorPDT.__name__
    )
    PDT_SLOT_2 = JuniorSlot.create_slot(
        identifier="PDT_SLOT_2", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
        layout_relation="above", layout_relative=PDT_SLOT_1, can_hold=JuniorPDT.__name__
    )
    PDT_SLOT_3 = JuniorSlot.create_slot(
        identifier="PDT_SLOT_3", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
        layout_relation="above", layout_relative=PDT_SLOT_2, can_hold=JuniorPDT.__name__
    )

    SV_VIAL_SLOTS = []

    num_sv_vial_per_row = 3

    for i in range(12):
        irow = i // num_sv_vial_per_row
        icol = i % num_sv_vial_per_row
        if i == 0:
            sv_vial_slot = JuniorSlot.create_slot(
                identifier=f"SV_VIAL_SLOT_{i + 1}", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
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
                identifier=f"SV_VIAL_SLOT_{i + 1}", layout_x=_SMALL_SLOT_SIZE_X, layout_y=_SMALL_SLOT_SIZE_Y,
                layout_relation=relation, layout_relative=relative, can_hold=JuniorVial.__name__
            )
        SV_VIAL_SLOTS.append(sv_vial_slot)

    SV_TOOL_SLOT = JuniorSlot.create_slot(
        identifier="SV_TOOL_SLOT", layout_x=_SLOT_SIZE_X, layout_y=_SLOT_SIZE_Y,
        layout_relation="above", layout_relative=SV_VIAL_SLOTS[9], can_hold=JuniorSvTool.__name__
    )

    BALANCE = JuniorSlot.create_slot(
        identifier="BALANCE", layout_x=_SLOT_SIZE_X, layout_y=_SLOT_SIZE_Y,
        layout_relation="right_to", layout_relative=SV_TOOL_SLOT, can_hold=JuniorVial.__name__, can_weigh=True
    )

    VPG_SLOT = JuniorSlot.create_slot(
        identifier="VPG_SLOT", layout_x=_SLOT_SIZE_X, layout_y=_SLOT_SIZE_Y,
        layout_relation="right_to", layout_relative=BALANCE, can_hold=JuniorVPG.__name__,
    )

    TIP_DISPOSAL = JuniorSlot.create_slot(identifier="TIP_DISPOSAL", layout_relation="right_to",
                                          layout_relative=VPG_SLOT, layout_x=_SMALL_SLOT_SIZE_X,
                                          layout_y=_SLOT_SIZE_Y, can_hold=None)
    return JUNIOR_LAB
