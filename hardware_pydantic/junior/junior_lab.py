from hardware_pydantic.junior.junior_devices import *
from hardware_pydantic.junior.settings import *


class JuniorBenchtop(BaseClass):
    rdfs_isDefinedBy = JuniorOntology
    SLOT_OFF_1: JuniorSlot
    SLOT_OFF_2: JuniorSlot
    SLOT_OFF_3: JuniorSlot

    WASH_BAY: JuniorWashBay

    SLOT_2_3_1: JuniorSlot
    SLOT_2_3_2: JuniorSlot
    SLOT_2_3_3: JuniorSlot

    SLOT_PDT_1: JuniorSlot
    SLOT_PDT_2: JuniorSlot
    SLOT_PDT_3: JuniorSlot

    SV_VIAL_SLOTS: list[JuniorSlot]

    SV_TOOL_SLOT: JuniorSlot

    BALANCE: JuniorSlot

    VPG_SLOT: JuniorSlot

    TIP_DISPOSAL: JuniorTipDisposal

    ARM_Z1: JuniorArmZ1

    ARM_Z2: JuniorArmZ2

    ARM_PLATFORM: JuniorArmPlatform

    SV_TOOL: JuniorSvt

    VPG: JuniorVpg

    PDP_1: JuniorPdp
    PDP_2: JuniorPdp
    PDP_3: JuniorPdp


def create_junior_base():
    assert len(JUNIOR_LAB.dict_object) == 0, "JUNIOR BASE HAS ALREADY BEEN INIT!!!"

    slot_off_1 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/SLOT-OFF-1", can_contain=[JuniorRack.rdf_type, ],
        layout=JuniorLayout.from_relative_layout()
    )
    slot_off_2 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/SLOT-OFF-2", can_contain=[JuniorRack.rdf_type, ],
        layout=JuniorLayout.from_relative_layout("above", list(slot_off_1.layout)[0])
    )
    slot_off_3 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/SLOT-OFF-3", can_contain=[JuniorRack.rdf_type, ],
        layout=JuniorLayout.from_relative_layout("above", list(slot_off_2.layout)[0])
    )

    wash_bay = JuniorWashBay(
        identifier=f"{JuniorOntology.namespace_iri}/WASH-BAY",
        layout=JuniorLayout.from_relative_layout("right_to", list(slot_off_1.layout)[0], JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL,
                                                 JUNIOR_LAYOUT_SLOT_SIZE_Y * 3)
    )

    slot_2_3_1 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/SLOT-2-3-1", can_contain=[JuniorRack.rdf_type, ], can_heat=True, can_cool=True, can_stir=True,
        layout=JuniorLayout.from_relative_layout("right_to", list(wash_bay.layout)[0])
    )
    slot_2_3_2 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/SLOT-2-3-2", can_contain=[JuniorRack.rdf_type, ], can_heat=True, can_stir=True,
        layout=JuniorLayout.from_relative_layout("above", list(slot_2_3_1.layout)[0])
    )
    slot_2_3_3 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/SLOT-2-3-3", can_contain=[JuniorRack.rdf_type, ], can_heat=True, can_stir=True,
        layout=JuniorLayout.from_relative_layout("above", list(slot_2_3_2.layout)[0])
    )

    slot_pdt_1 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/PDT-SLOT-1", can_contain=[JuniorPdp.rdf_type, ],
        layout=JuniorLayout.from_relative_layout("right_to", list(slot_2_3_1.layout)[0], JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL * 2,
                                                 JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
    )
    slot_pdt_2 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/PDT-SLOT-2", can_contain=[JuniorPdp.rdf_type, ],
        layout=JuniorLayout.from_relative_layout("above", list(slot_pdt_1.layout)[0], JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL * 2,
                                                 JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
    )
    slot_pdt_3 = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/PDT-SLOT-3", can_contain=[JuniorPdp.rdf_type, ],
        layout=JuniorLayout.from_relative_layout("above", list(slot_pdt_2.layout)[0], JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL * 2,
                                                 JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
    )

    sv_vial_slots = []

    num_sv_vial_per_row = 3

    for i in range(12):
        irow = i // num_sv_vial_per_row
        icol = i % num_sv_vial_per_row
        if i == 0:
            sv_vial_slot = JuniorSlot(
                identifier=f"{JuniorOntology.namespace_iri}/SVV-SLOT-{i + 1}", can_contain=[JuniorVial.rdf_type, ],
                layout=JuniorLayout.from_relative_layout('above', list(slot_pdt_3.layout)[0],
                                                         JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL * 2,
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
                identifier=f"{JuniorOntology.namespace_iri}/SVV-SLOT-{i + 1}", can_contain=[JuniorVial.rdf_type, ],
                layout=JuniorLayout.from_relative_layout(relation, list(relative.layout)[0], JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL * 2,
                                                         JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL),
            )
        sv_vial_slots.append(sv_vial_slot)

    sv_tool_slot = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/SV-TOOL-SLOT", can_contain=[JuniorSvt.rdf_type, ],
        layout=JuniorLayout.from_relative_layout('above', list(sv_vial_slots[9].layout)[0]),
    )

    balance = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/BALANCE-SLOT", can_contain=[JuniorRack.rdf_type, ], can_weigh=True,
        layout=JuniorLayout.from_relative_layout('right_to', list(sv_tool_slot.layout)[0]),
    )

    vpg_slot = JuniorSlot(
        identifier=f"{JuniorOntology.namespace_iri}/VPG-SLOT", can_contain=[JuniorVpg.rdf_type, ],
        layout=JuniorLayout.from_relative_layout('right_to', list(balance.layout)[0], )
    )

    tip_disposal = JuniorTipDisposal(
        identifier=f"{JuniorOntology.namespace_iri}/DISPOSAL",
        layout=JuniorLayout.from_relative_layout('right_to', list(vpg_slot.layout)[0], layout_x=JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL)
    )

    arm_z1 = JuniorArmZ1(
        identifier=f'{JuniorOntology.namespace_iri}/Z1-ARM', contained_by=f'{JuniorOntology.namespace_iri}/ARM-PLATFORM', contained_in_slot="z1",
        can_contain=[JuniorZ1Needle.rdf_type, ],
        has_slot_content=[
            JuniorZ1Needle(identifier=f"{JuniorOntology.namespace_iri}/Z1-Needle-{i + 1}", contained_by=f'{JuniorOntology.namespace_iri}/Z1-ARM',
                           contained_in_slot=str(i + 1), material="STEEL") for i in range(7)
        ],
    )

    arm_z2 = JuniorArmZ2(
        identifier=f'{JuniorOntology.namespace_iri}/Z2-ARM', contained_by=f'{JuniorOntology.namespace_iri}/ARM-PLATFORM', contained_in_slot='z2',
        can_contain=[JuniorSvt.rdf_type, JuniorVpg.rdf_type, JuniorPdp.rdf_type, ],
    )

    arm_platform = JuniorArmPlatform(
        identifier=f'{JuniorOntology.namespace_iri}/ARM-PLATFORM', can_contain=[JuniorArmZ1.rdf_type, JuniorArmZ2.rdf_type, ],
        has_position_on_top_of=slot_off_1.identifier, has_anchor_arm=arm_z1.identifier,
        has_slot_content=[arm_z1, arm_z2],
    )

    sv_tool = JuniorSvt(identifier=f"{JuniorOntology.namespace_iri}/SV-TOOL", contained_by=sv_tool_slot.identifier, powder_param_known=False,
                        contained_in_slot='SLOT')
    sv_tool_slot.has_slot_content.add(sv_tool)

    vpg = JuniorVpg(identifier=f"{JuniorOntology.namespace_iri}/VPG", contained_by=vpg_slot.identifier, contained_in_slot='SLOT')
    vpg_slot.has_slot_content.add(vpg)

    pdp_1 = JuniorPdp(identifier=f'{JuniorOntology.namespace_iri}/PDT-1', contained_by=slot_pdt_1.identifier, contained_in_slot='SLOT')
    pdp_2 = JuniorPdp(identifier=f'{JuniorOntology.namespace_iri}/PDT-2', contained_by=slot_pdt_2.identifier, contained_in_slot='SLOT')
    pdp_3 = JuniorPdp(identifier=f'{JuniorOntology.namespace_iri}/PDT-3', contained_by=slot_pdt_3.identifier, contained_in_slot='SLOT')
    slot_pdt_1.has_slot_content.add(pdp_1)
    slot_pdt_2.has_slot_content.add(pdp_2)
    slot_pdt_3.has_slot_content.add(pdp_3)

    return JuniorBenchtop(
        SLOT_OFF_1=slot_off_1,
        SLOT_OFF_2=slot_off_2,
        SLOT_OFF_3=slot_off_3,

        WASH_BAY=wash_bay,

        SLOT_2_3_1=slot_2_3_1,
        SLOT_2_3_2=slot_2_3_2,
        SLOT_2_3_3=slot_2_3_3,

        SLOT_PDT_1=slot_pdt_1,
        SLOT_PDT_2=slot_pdt_2,
        SLOT_PDT_3=slot_pdt_3,

        SV_VIAL_SLOTS=sv_vial_slots,

        SV_TOOL_SLOT=sv_tool_slot,

        BALANCE=balance,

        VPG_SLOT=vpg_slot,

        TIP_DISPOSAL=tip_disposal,

        ARM_Z1=arm_z1,

        ARM_Z2=arm_z2,

        ARM_PLATFORM=arm_platform,

        SV_TOOL=sv_tool,

        VPG=vpg,

        PDP_1=pdp_1,
        PDP_2=pdp_2,
        PDP_3=pdp_3,
    )
