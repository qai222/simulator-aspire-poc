from hardware_pydantic.junior.junior_lab import *


def pick_drop_rack_to(
        junior_benchtop: JuniorBenchtop,
        rack: JuniorRack, src_slot: JuniorSlot, dest_slot: JuniorSlot
) -> list[JuniorInstruction]:
    ins1 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z2,
            "move_to_slot": junior_benchtop.VPG_SLOT,
        },
        description=f"move to slot: {junior_benchtop.VPG_SLOT.identifier}"
    )

    ins2 = JuniorInstruction(
        device=junior_benchtop.ARM_Z2, action_name="pick_up",
        action_parameters={
            "thing": junior_benchtop.VPG,
        },
        description=f"pick up: {junior_benchtop.VPG.identifier}"
    )

    ins3 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z2,
            "move_to_slot": src_slot,
        },
        description=f"move to slot: {src_slot.identifier}"
    )
    ins4 = JuniorInstruction(
        device=junior_benchtop.ARM_Z2, action_name="pick_up",
        action_parameters={
            "thing": rack,
        },
        description=f"pick up: {rack.identifier}"
    )
    ins5 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z2,
            "move_to_slot": dest_slot,
        },
        description=f"move to slot: {dest_slot.identifier}"
    )
    ins6 = JuniorInstruction(
        device=junior_benchtop.ARM_Z2, action_name="put_down",
        action_parameters={
            "dest_slot": dest_slot,
        },
        description=f"put down: {dest_slot.identifier}"
    )

    ins7 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z2,
            "move_to_slot": junior_benchtop.VPG_SLOT,
        },
        description=f"move to slot: {junior_benchtop.VPG_SLOT.identifier}"
    )

    ins8 = JuniorInstruction(
        device=junior_benchtop.ARM_Z2, action_name="put_down",
        action_parameters={
            "dest_slot": junior_benchtop.VPG_SLOT,
        },
        description=f"put down: {junior_benchtop.VPG_SLOT.identifier}"
    )

    ins_list = [ins1, ins2, ins3, ins4, ins5, ins6, ins7, ins8]
    JuniorInstruction.path_graph(ins_list)
    return ins_list
