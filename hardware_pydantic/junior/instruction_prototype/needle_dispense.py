from hardware_pydantic.junior.junior_lab import *


def needle_dispense(
        junior_benchtop: JuniorBenchtop,
        src_vials: list[JuniorVial],
        src_slot: JuniorSlot,
        dest_vials: list[JuniorVial],
        dest_vials_slot: JuniorSlot,
        amounts: list[float],
):
    z1_needles = [JUNIOR_LAB[f"Z1 Needle {i + 1}"] for i in range(len(amounts))]
    ins1 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z1,
            "move_to_slot": src_slot,
        },
        description=f"move to slot: {src_slot.identifier}"
    )
    ins2 = JuniorInstruction(
        device=junior_benchtop.ARM_Z1, action_name="concurrent_aspirate",
        action_parameters={
            "source_containers": src_vials,
            "dispenser_containers": z1_needles,
            "amounts": amounts,
        },
        description=f"concurrent aspirate from: {','.join([v.identifier for v in src_vials])}"
    )
    ins3 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z1,
            "move_to_slot": dest_vials_slot,
        },
        description=f"move to slot: {dest_vials_slot.identifier}"
    )
    ins4 = JuniorInstruction(
        device=junior_benchtop.ARM_Z1, action_name="concurrent_dispense",
        action_parameters={
            "destination_containers": dest_vials,
            "dispenser_containers": z1_needles,
            "amounts": amounts,
        },
        description=f"concurrent dispense to: {','.join([v.identifier for v in dest_vials])}"
    )
    ins5 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z1,
            "move_to_slot": junior_benchtop.WASH_BAY,
        },
        description=f"move to slot: WASH BAY"
    )
    ins6 = JuniorInstruction(
        device=junior_benchtop.ARM_Z1, action_name="wash",
        action_parameters={
            "wash_bay": junior_benchtop.WASH_BAY,
        },
        description="wash needles"
    )
    ins_list = [ins1, ins2, ins3, ins4, ins5, ins6]
    JuniorInstruction.path_graph(ins_list)
    return ins_list
