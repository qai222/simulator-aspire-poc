from hardware_pydantic.junior.junior_lab import *


def solid_dispense(
        junior_benchtop: JuniorBenchtop,
        sv_vial: JuniorVial,
        sv_vial_slot: JuniorSlot,
        dest_vials: list[JuniorVial],
        amount: float,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True,
):
    ins3 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z2,
            "move_to_slot": sv_vial_slot,
        },
        description=f"move to slot: {sv_vial_slot.identifier}"
    )

    ins4 = JuniorInstruction(
        device=junior_benchtop.ARM_Z2, action_name="pick_up",
        action_parameters={"thing": sv_vial},
        description=f"pick up: {sv_vial.identifier}",
    )

    ins5 = JuniorInstruction(
        device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": junior_benchtop.ARM_Z2,
            "move_to_slot": junior_benchtop.BALANCE,
        },
        description=f"move to slot: {junior_benchtop.BALANCE.identifier}"
    )

    if include_pickup_svtool:
        ins1 = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": junior_benchtop.SV_TOOL_SLOT,
            },
            description=f"move to slot: {junior_benchtop.SV_TOOL_SLOT.identifier}"
        )

        ins2 = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="pick_up",
            action_parameters={"thing": junior_benchtop.SV_TOOL},
            description=f"pick up: {junior_benchtop.SV_TOOL.identifier}",
        )
        ins_list = [ins1, ins2, ins3, ins4, ins5]
    else:
        ins_list = [ins3, ins4, ins5]

    for dest_vial in dest_vials:
        ins6 = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="dispense_sv",
            action_parameters={
                "destination_container": dest_vial,
                "amount": amount,
                # "dispense_speed": speed,
            },
            description=f"dispense_sv to: {dest_vial.identifier}",
        )
        ins_list.append(ins6)

    if include_dropoff_svvial:
        ins7 = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": sv_vial_slot,
            },
            description=f"move to slot: {sv_vial_slot.identifier}"
        )

        ins8 = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="put_down",
            action_parameters={
                "dest_slot": sv_vial_slot,
            },
            description=f"put down: {sv_vial_slot.identifier}"
        )
        ins_list.append(ins7)
        ins_list.append(ins8)

    if include_dropoff_svtool:
        ins9 = JuniorInstruction(
            device=junior_benchtop.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": junior_benchtop.ARM_Z2,
                "move_to_slot": junior_benchtop.SV_TOOL_SLOT,
            },
            description=f"move to slot: {junior_benchtop.SV_TOOL_SLOT.identifier}"
        )

        ins10 = JuniorInstruction(
            device=junior_benchtop.ARM_Z2, action_name="put_down",
            action_parameters={
                "dest_slot": junior_benchtop.SV_TOOL_SLOT,
            },
            description=f"put down: {junior_benchtop.SV_TOOL_SLOT.identifier}"
        )
        ins_list.append(ins9)
        ins_list.append(ins10)

    JuniorInstruction.path_graph(ins_list)

    return ins_list
