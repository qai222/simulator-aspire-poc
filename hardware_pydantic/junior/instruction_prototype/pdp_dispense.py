from hardware_pydantic.junior.junior_lab import *


def pdp_dispense(
        junior_benchtop: JuniorBenchtop,
        src_vial: JuniorVial, src_slot: JuniorSlot,
        tips: list[JuniorPdpTip], tips_slot: JuniorSlot,
        dest_vials: list[JuniorVial], dest_vials_slot: JuniorSlot,
        amount: float,
        include_dropoff_pdp=True,
        include_pickup_pdp=True,
):
    ins_list = []

    if include_pickup_pdp:
        ins1 = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": junior_benchtop.SLOT_PDT_1,
            },
            description=f"move to slot: PDT SLOT 1"
        )

        ins2 = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="pick_up",
            action_parameters={"thing": junior_benchtop.PDP_1},
            description=f"pick up: {junior_benchtop.PDP_1.identifier}",
        )
        ins_list += [ins1, ins2]

    for tip, dest_vial in zip(tips, dest_vials):
        i_a = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": tips_slot,
            },
            description=f"move to slot: {tips_slot.identifier}"
        )
        i_b = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="pick_up",
            action_parameters={"thing": tip},
            description=f"pick up: {tip.identifier}",
        )
        i_c = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": src_slot,
            },
            description=f"move to slot: {src_slot.identifier}"
        )
        i_d = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="aspirate_pdp",
            action_parameters={
                "source_container": src_vial,
                "amount": amount,
                # "aspirate_speed": speed,
            },
            description=f"aspirate_pdp from: {src_vial.identifier} amount: {amount}"
        )
        i_e = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": dest_vials_slot,
            },
            description=f"move to slot: {dest_vials_slot.identifier}"
        )
        i_f = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="dispense_pdp",
            action_parameters={
                "destination_container": dest_vial,
                "amount": amount,
                # "dispense_speed": speed,
            },
            description=f"dispense_pdp to: {dest_vial.identifier}"
        )
        i_g = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": junior_benchtop.TIP_DISPOSAL,
            },
            description=f"move to slot: DISPOSAL"
        )
        i_h = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="put_down",
            action_parameters={
                "dest_slot": junior_benchtop.TIP_DISPOSAL,
            },
            description="put down: DISPOSAL"
        )
        ins_list += [i_a, i_b, i_c, i_d, i_e, i_f, i_g, i_h]

    if include_dropoff_pdp:
        ins_xx = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": junior_benchtop.SLOT_PDT_1,
            },
            description=f"move to slot: {junior_benchtop.SLOT_PDT_1.identifier}"
        )
        ins_yy = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="put_down",
            action_parameters={
                "dest_slot": junior_benchtop.SLOT_PDT_1,
            },
            description=f"put down: {junior_benchtop.SLOT_PDT_1.identifier}"
        )
        ins_list += [ins_xx, ins_yy]

    JuniorInstruction.path_graph(ins_list)
    return ins_list
