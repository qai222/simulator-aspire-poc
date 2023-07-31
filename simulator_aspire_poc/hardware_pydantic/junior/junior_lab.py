from simulator_aspire_poc.hardware_pydantic.junior.junior_devices import *
from simulator_aspire_poc.hardware_pydantic.junior.settings import *


def create_junior_base():
    assert len(JUNIOR_LAB.dict_object) == 0, "JUNIOR BASE HAS ALREADY BEEN INIT!!!"

    slot_off_1 = JuniorSlot(
        identifier="SLOT OFF-1", can_contain=[JuniorRack.__name__, ],
        layout=JuniorLayout.from_relative_layout()
    )
    slot_off_2 = JuniorSlot(
        identifier="SLOT OFF-2", can_contain=[JuniorRack.__name__, ],
        layout=JuniorLayout.from_relative_layout("above", slot_off_1.layout)
    )
    slot_off_3 = JuniorSlot(
        identifier="SLOT OFF-3", can_contain=[JuniorRack.__name__, ],
        layout=JuniorLayout.from_relative_layout("above", slot_off_2.layout)
    )

    wash_bay = JuniorWashBay(
        identifier="WASH BAY",
        layout=JuniorLayout.from_relative_layout("right_to", slot_off_1.layout, JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL,
                                                 JUNIOR_LAYOUT_SLOT_SIZE_Y * 3)
    )

    slot_2_3_1 = JuniorSlot(
        identifier="SLOT 2-3-1", can_contain=[JuniorRack.__name__, ], can_heat=True, can_cool=True, can_stir=True,
        layout=JuniorLayout.from_relative_layout("right_to", wash_bay.layout)
    )
    slot_2_3_2 = JuniorSlot(
        identifier="SLOT 2-3-2", can_contain=[JuniorRack.__name__, ], can_heat=True, can_stir=True,
        layout=JuniorLayout.from_relative_layout("above", slot_2_3_1.layout)
    )
    slot_2_3_3 = JuniorSlot(
        identifier="SLOT 2-3-3", can_contain=[JuniorRack.__name__, ], can_heat=True, can_stir=True,
        layout=JuniorLayout.from_relative_layout("above", slot_2_3_2.layout)
    )

    slot_pdt_1 = JuniorSlot(
        identifier="PDT SLOT 1", can_contain=[JuniorPdp.__name__, ],
        layout=JuniorLayout.from_relative_layout("right_to", slot_2_3_1.layout, JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL*2,
                                                 JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
    )
    slot_pdt_2 = JuniorSlot(
        identifier="PDT SLOT 2", can_contain=[JuniorPdp.__name__, ],
        layout=JuniorLayout.from_relative_layout("above", slot_pdt_1.layout, JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL*2,
                                                 JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
    )
    slot_pdt_3 = JuniorSlot(
        identifier="PDT SLOT 3", can_contain=[JuniorPdp.__name__, ],
        layout=JuniorLayout.from_relative_layout("above", slot_pdt_2.layout, JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL*2,
                                                 JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
    )

    sv_vial_slots = []

    num_sv_vial_per_row = 3

    for i in range(12):
        irow = i // num_sv_vial_per_row
        icol = i % num_sv_vial_per_row
        if i == 0:
            sv_vial_slot = JuniorSlot(
                identifier=f"SVV SLOT {i + 1}", can_contain=[JuniorVial.__name__, ],
                layout=JuniorLayout.from_relative_layout('above', slot_pdt_3.layout, JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL*2,
                                                         JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
            )
        else:
            last_slot = sv_vial_slots[i - 1]
            if irow == (i - 1) // num_sv_vial_per_row:
                relation = "right_to"
                relative = last_slot
            else:
                relation = "above"
                relative = sv_vial_slots[i - num_sv_vial_per_row]
            sv_vial_slot = JuniorSlot(
                identifier=f"SVV SLOT {i + 1}", can_contain=[JuniorVial.__name__, ],
                layout=JuniorLayout.from_relative_layout(relation, relative.layout, JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL*2,
                                                         JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
            )
        sv_vial_slots.append(sv_vial_slot)

    sv_tool_slot = JuniorSlot(
        identifier="SV TOOL SLOT", can_contain=[JuniorSvt.__name__, ],
        layout=JuniorLayout.from_relative_layout('above', sv_vial_slots[9].layout),
    )

    balance = JuniorSlot(
        identifier="BALANCE SLOT", can_contain=[JuniorRack.__name__, ], can_weigh=True,
        layout=JuniorLayout.from_relative_layout('right_to', sv_tool_slot.layout),
    )

    vpg_slot = JuniorSlot(
        identifier="VPG SLOT", can_contain=[JuniorVpg.__name__, ],
        layout=JuniorLayout.from_relative_layout('right_to', balance.layout, )
    )

    tip_disposal = JuniorTipDisposal(
        identifier="DISPOSAL",
        layout=JuniorLayout.from_relative_layout('right_to', vpg_slot.layout, layout_x=JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL)
    )

    arm_z1 = JuniorArmZ1(
        identifier='Z1 ARM', contained_by='ARM PLATFORM', contained_in_slot="z1",
        can_contain=[JuniorZ1Needle.__name__, ],
        slot_content={
            str(i + 1): JuniorZ1Needle(identifier=f"Z1 Needle {i + 1}", contained_by='Z1 ARM',
                                       contained_in_slot=str(i + 1), material="STEEL").identifier for i in range(7)
        },
    )

    arm_z2 = JuniorArmZ2(
        identifier='Z2 ARM', contained_by='ARM PLATFORM', contained_in_slot='z2',
        can_contain=[JuniorSvt.__name__, JuniorVpg.__name__, JuniorPdp.__name__, ],
    )

    arm_platform = JuniorArmPlatform(
        identifier='ARM PLATFORM', can_contain=[JuniorArmZ1.__name__, JuniorArmZ2.__name__, ],
        position_on_top_of=slot_off_1.identifier, anchor_arm=arm_z1.identifier,
        slot_content={"z1": arm_z1.identifier, "z2": arm_z2.identifier},
    )

    sv_tool = JuniorSvt(identifier="SV TOOL", contained_by=sv_tool_slot.identifier, powder_param_known=False)
    sv_tool_slot.slot_content['SLOT'] = sv_tool.identifier

    vpg = JuniorVpg(identifier="VPG", contained_by=vpg_slot.identifier)
    vpg_slot.slot_content['SLOT'] = vpg.identifier

    pdp_1 = JuniorPdp(identifier='PDT 1', contained_by=slot_pdt_1.identifier)
    pdp_2 = JuniorPdp(identifier='PDT 2', contained_by=slot_pdt_2.identifier)
    pdp_3 = JuniorPdp(identifier='PDT 3', contained_by=slot_pdt_3.identifier)
    slot_pdt_1.slot_content['SLOT'] = pdp_1.identifier
    slot_pdt_2.slot_content['SLOT'] = pdp_2.identifier
    slot_pdt_3.slot_content['SLOT'] = pdp_3.identifier

    return JUNIOR_LAB
