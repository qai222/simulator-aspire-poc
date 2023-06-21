from hardware_pydantic.junior import *

"""
following the notes of N-Sulfonylation 
"""

create_junior_base()

# CONCURRENCY = 4
CONCURRENCY = 1

# RACK A: holding HRVs with DCM, on off deck initially
RACK_A, RACK_A_VIALS = JuniorRack.create_rack_with_empty_vials(
    n_vials=CONCURRENCY, rack_capacity=6, vial_type="HRV", rack_id="RACK A"
)
RACK_A: JuniorRack
RACK_A_VIALS: list[JuniorVial]
for v in RACK_A_VIALS:
    v.chemical_content = {"DCM": 1000}
JuniorSlot.put_rack_in_a_slot(RACK_A, JUNIOR_LAB['SLOT OFF-1'])

# RACK B: holding HRVs for reactions, at 2-3-2 initially, one for RSO2Cl stock solution, another for pyridine source
RACK_B, RACK_B_VIALS = JuniorRack.create_rack_with_empty_vials(
    n_vials=CONCURRENCY + 1, rack_capacity=8, vial_type="HRV", rack_id="RACK B"
)
RACK_B: JuniorRack
RACK_B_VIALS: list[JuniorVial]
JuniorSlot.put_rack_in_a_slot(RACK_B, JUNIOR_LAB['SLOT 2-3-2'])

# RACK C: holding one MRV for reaction, at 2-3-1 initially
RACK_C, RACK_C_VIALS = JuniorRack.create_rack_with_empty_vials(
    n_vials=CONCURRENCY, rack_capacity=6, vial_type="MRV", rack_id="RACK C"
)
RACK_C: JuniorRack
RACK_C_VIALS: list[JuniorVial]
JuniorSlot.put_rack_in_a_slot(RACK_C, JUNIOR_LAB['SLOT 2-3-1'])

# RACK D: holding PDP tips, at 2-3-3 initially
RACK_D, RACK_D_TIPS = JuniorRack.create_rack_with_empty_tips(
    n_tips=CONCURRENCY, rack_capacity=6, rack_id="RACK D", tip_id_inherit=True
)
RACK_D: JuniorRack
RACK_D_TIPS: list[JuniorPdpTip]
JuniorSlot.put_rack_in_a_slot(RACK_D, JUNIOR_LAB['SLOT 2-3-3'])

# SV VIALS, one for solid amine (aniline), another for RSO2Cl, each sits in a SVV SLOT
SVV_1 = JuniorVial(
    identifier="SV VIAL 1", contained_by=JUNIOR_LAB['SVV SLOT 1'].identifier,
    chemical_content={'solid amine': 1000},
    vial_type='SV',
)
SVV_2 = JuniorVial(
    identifier="SV VIAL 2", contained_by=JUNIOR_LAB['SVV SLOT 2'].identifier,
    chemical_content={'sulfonyl chloride': 1000},
    vial_type='SV',
)
JUNIOR_LAB['SVV SLOT 1'].slot_content['SLOT'] = SVV_1.identifier
JUNIOR_LAB['SVV SLOT 2'].slot_content['SLOT'] = SVV_2.identifier

# INSTRUCTIONS
Z1_ARM = JUNIOR_LAB['Z1 ARM']
Z2_ARM = JUNIOR_LAB['Z2 ARM']
ARM_PLATFORM = JUNIOR_LAB['ARM PLATFORM']

Z1NEEDLES = [JUNIOR_LAB[f"Z1 Needle {i + 1}"] for i in range(CONCURRENCY)]

VPG = JUNIOR_LAB['VPG']
VPG_SLOT = JUNIOR_LAB['VPG SLOT']
SV_TOOL = JUNIOR_LAB['SV TOOL']
SV_TOOL_SLOT = JUNIOR_LAB['SV TOOL SLOT']
BALANCE_SLOT = JUNIOR_LAB['BALANCE SLOT']

PDP_1 = JUNIOR_LAB['PDT 1']

DCM_VIALS = RACK_A_VIALS
MRV_VIALS = RACK_C_VIALS
PYRIDINE_VIAL = RACK_B_VIALS[0]
RSO2Cl_STOCK_SOLUTION_VIALS = RACK_B_VIALS[1:]
PYRIDINE_VIAL.chemical_content = {"pyridine": 1000}


def pick_drop_rack_to(rack: JuniorRack, src_slot: JuniorSlot, dest_slot: JuniorSlot):
    ins1 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z2_ARM,
            "move_to_slot": VPG_SLOT,
        },
        description=f"move to slot: {VPG_SLOT.identifier}"
    )

    ins2 = JuniorInstruction(
        device=Z2_ARM, action_name="pick_up",
        action_parameters={
            "thing": VPG,
        },
        description=f"pick up: {VPG.identifier}"
    )

    ins3 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z2_ARM,
            "move_to_slot": src_slot,
        },
        description=f"move to slot: {src_slot.identifier}"
    )
    ins4 = JuniorInstruction(
        device=Z2_ARM, action_name="pick_up",
        action_parameters={
            "thing": rack,
        },
        description=f"pick up: {rack.identifier}"
    )
    ins5 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z2_ARM,
            "move_to_slot": dest_slot,
        },
        description=f"move to slot: {dest_slot.identifier}"
    )
    ins6 = JuniorInstruction(
        device=Z2_ARM, action_name="put_down",
        action_parameters={
            "dest_slot": dest_slot,
        },
        description=f"put down: {dest_slot.identifier}"
    )

    ins7 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z2_ARM,
            "move_to_slot": VPG_SLOT,
        },
        description=f"move to slot: {VPG_SLOT.identifier}"
    )

    ins8 = JuniorInstruction(
        device=Z2_ARM, action_name="put_down",
        action_parameters={
            "dest_slot": VPG_SLOT,
        },
        description=f"put down: {VPG_SLOT.identifier}"
    )

    ins_list = [ins1, ins2, ins3, ins4, ins5, ins6, ins7, ins8]
    JuniorInstruction.path_graph(ins_list)
    return ins_list


def solid_dispense(
        sv_vial: JuniorVial,
        sv_vial_slot: JuniorSlot,
        dest_vials: list[JuniorVial],
        amount: float, speed: float,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True,
):
    ins3 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z2_ARM,
            "move_to_slot": sv_vial_slot,
        },
        description=f"move to slot: {sv_vial_slot.identifier}"
    )

    ins4 = JuniorInstruction(
        device=Z2_ARM, action_name="pick_up",
        action_parameters={"thing": sv_vial},
        description=f"pick up: {sv_vial.identifier}",
    )

    ins5 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z2_ARM,
            "move_to_slot": BALANCE_SLOT,
        },
        description=f"move to slot: {BALANCE_SLOT.identifier}"
    )

    if include_pickup_svtool:
        ins1 = JuniorInstruction(
            device=ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": Z2_ARM,
                "move_to_slot": SV_TOOL_SLOT,
            },
            description=f"move to slot: {SV_TOOL_SLOT.identifier}"
        )

        ins2 = JuniorInstruction(
            device=Z2_ARM, action_name="pick_up",
            action_parameters={"thing": SV_TOOL},
            description=f"pick up: {SV_TOOL.identifier}",
        )
        ins_list = [ins1, ins2, ins3, ins4, ins5]
    else:
        ins_list = [ins3, ins4, ins5]

    for dest_vial in dest_vials:
        ins6 = JuniorInstruction(
            device=Z2_ARM, action_name="dispense_sv",
            action_parameters={
                "destination_container": dest_vial,
                "amount": amount,
                "dispense_speed": speed,
            },
            description=f"dispense_sv to: {dest_vial.identifier}",
        )
        ins_list.append(ins6)

    if include_dropoff_svvial:
        ins7 = JuniorInstruction(
            device=ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": Z2_ARM,
                "move_to_slot": sv_vial_slot,
            },
            description=f"move to slot: {sv_vial_slot.identifier}"
        )

        ins8 = JuniorInstruction(
            device=Z2_ARM, action_name="put_down",
            action_parameters={
                "dest_slot": sv_vial_slot,
            },
            description=f"put down: {sv_vial_slot.identifier}"
        )
        ins_list.append(ins7)
        ins_list.append(ins8)

    if include_dropoff_svtool:
        ins9 = JuniorInstruction(
            device=ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": Z2_ARM,
                "move_to_slot": SV_TOOL_SLOT,
            },
            description=f"move to slot: {SV_TOOL_SLOT.identifier}"
        )

        ins10 = JuniorInstruction(
            device=Z2_ARM, action_name="put_down",
            action_parameters={
                "dest_slot": SV_TOOL_SLOT,
            },
            description=f"put down: {SV_TOOL_SLOT.identifier}"
        )
        ins_list.append(ins9)
        ins_list.append(ins10)

    JuniorInstruction.path_graph(ins_list)

    return ins_list


def needle_dispense(
        src_vials: list[JuniorVial],
        src_slot: JuniorSlot,
        dest_vials: list[JuniorVial],
        dest_vials_slot: JuniorSlot,
        amount: float, speed: float,
):
    ins1 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z1_ARM,
            "move_to_slot": src_slot,
        },
        description=f"move to slot: {src_slot.identifier}"
    )
    ins2 = JuniorInstruction(
        device=Z1_ARM, action_name="concurrent_aspirate",
        action_parameters={
            "source_containers": src_vials,
            "dispenser_containers": Z1NEEDLES,
            "amounts": [amount, ] * CONCURRENCY,
            "aspirate_speed": speed,
        },
        description=f"concurrent aspirate from: {','.join([v.identifier for v in src_vials])}"
    )
    ins3 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z1_ARM,
            "move_to_slot": dest_vials_slot,
        },
        description=f"move to slot: {dest_vials_slot.identifier}"
    )
    ins4 = JuniorInstruction(
        device=Z1_ARM, action_name="concurrent_dispense",
        action_parameters={
            "destination_containers": dest_vials,
            "dispenser_containers": Z1NEEDLES,
            "dispense_speed": speed,
            "amounts": [amount, ] * CONCURRENCY,
        },
        description=f"concurrent dispense to: {','.join([v.identifier for v in dest_vials])}"
    )
    ins5 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z1_ARM,
            "move_to_slot": JUNIOR_LAB['WASH BAY'],
        },
        description=f"move to slot: WASH BAY"
    )
    ins6 = JuniorInstruction(
        device=Z1_ARM, action_name="wash",
        action_parameters={
            "wash_bay": JUNIOR_LAB['WASH BAY'],
        },
        description="wash needles"
    )
    ins_list = [ins1, ins2, ins3, ins4, ins5, ins6]
    JuniorInstruction.path_graph(ins_list)
    return ins_list


def pdp_dispense(
        src_vial: JuniorVial, src_slot: JuniorSlot,
        tips: list[JuniorPdpTip], tips_slot: JuniorSlot,
        dest_vials: list[JuniorVial], dest_vials_slot: JuniorSlot,
        amount: float, speed: float
):
    ins1 = JuniorInstruction(
        device=ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": Z2_ARM,
            "move_to_slot": JUNIOR_LAB['PDT SLOT 1'],
        },
        description=f"move to slot: PDT SLOT 1"
    )

    ins2 = JuniorInstruction(
        device=Z2_ARM, action_name="pick_up",
        action_parameters={"thing": PDP_1},
        description=f"pick up: {PDP_1.identifier}",
    )

    ins_list = [ins1, ins2]

    for tip, dest_vial in zip(tips, dest_vials):
        i_a = JuniorInstruction(
            device=ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": Z2_ARM,
                "move_to_slot": tips_slot,
            },
            description=f"move to slot: {tips_slot.identifier}"
        )
        i_b = JuniorInstruction(
            device=Z2_ARM, action_name="pick_up",
            action_parameters={"thing": tip},
            description=f"pick up: {tip.identifier}",
        )
        i_c = JuniorInstruction(
            device=ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": Z2_ARM,
                "move_to_slot": src_slot,
            },
            description=f"move to slot: {src_slot.identifier}"
        )
        i_d = JuniorInstruction(
            device=Z2_ARM, action_name="aspirate_pdp",
            action_parameters={
                "source_container": src_vial,
                "amount": amount,
                "aspirate_speed": speed,
            },
            description=f"aspirate_pdp from: {src_vial.identifier}"
        )
        i_e = JuniorInstruction(
            device=ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": Z2_ARM,
                "move_to_slot": dest_vials_slot,
            },
            description=f"move to slot: {dest_vials_slot.identifier}"
        )
        i_f = JuniorInstruction(
            device=Z2_ARM, action_name="dispense_pdp",
            action_parameters={
                "destination_container": dest_vial,
                "amount": amount,
                "dispense_speed": speed,
            },
            description=f"dispense_pdp to: {dest_vial.identifier}"
        )
        i_g = JuniorInstruction(
            device=ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": Z2_ARM,
                "move_to_slot": JUNIOR_LAB['DISPOSAL'],
            },
            description=f"move to slot: DISPOSAL"
        )
        i_h = JuniorInstruction(
            device=Z2_ARM, action_name="put_down",
            action_parameters={
                "dest_slot": JUNIOR_LAB['DISPOSAL'],
            },
            description="put down: DISPOSAL"
        )
        ins_list += [i_a, i_b, i_c, i_d, i_e, i_f, i_g, i_h]
    JuniorInstruction.path_graph(ins_list)
    return ins_list


ins_list1 = pick_drop_rack_to(RACK_B, JUNIOR_LAB['SLOT 2-3-2'], BALANCE_SLOT)

ins_list2 = solid_dispense(SVV_2, JUNIOR_LAB['SVV SLOT 2'], RSO2Cl_STOCK_SOLUTION_VIALS, 10, 0.2,
                           include_pickup_svtool=True, include_dropoff_svvial=True, include_dropoff_svtool=True)
ins_list2[0].preceding_instructions.append(ins_list1[-1].identifier)

ins_list3 = pick_drop_rack_to(RACK_B, BALANCE_SLOT, JUNIOR_LAB['SLOT 2-3-2'])
ins_list3[0].preceding_instructions.append(ins_list2[-1].identifier)

ins_list4 = pick_drop_rack_to(RACK_C, JUNIOR_LAB['SLOT 2-3-1'], BALANCE_SLOT)
ins_list4[0].preceding_instructions.append(ins_list3[-1].identifier)

ins_list5 = solid_dispense(SVV_1, JUNIOR_LAB['SVV SLOT 1'], MRV_VIALS, 10, 0.2, include_pickup_svtool=True,
                           include_dropoff_svvial=True, include_dropoff_svtool=True)
ins_list5[0].preceding_instructions.append(ins_list4[-1].identifier)

ins_list6 = pick_drop_rack_to(RACK_C, BALANCE_SLOT, JUNIOR_LAB['SLOT 2-3-1'])
ins_list6[0].preceding_instructions.append(ins_list5[-1].identifier)

ins_list7 = needle_dispense(DCM_VIALS, JUNIOR_LAB['SLOT OFF-1'], MRV_VIALS, JUNIOR_LAB['SLOT 2-3-1'], 10, 0.2)
ins_list7[0].preceding_instructions.append(ins_list6[-1].identifier)

ins_list8 = needle_dispense(DCM_VIALS, JUNIOR_LAB['SLOT OFF-1'], RSO2Cl_STOCK_SOLUTION_VIALS, JUNIOR_LAB['SLOT 2-3-2'],
                            10, 0.2)
ins_list8[0].preceding_instructions.append(ins_list7[-1].identifier)

ins_list9 = pdp_dispense(PYRIDINE_VIAL, JUNIOR_LAB['SLOT 2-3-2'], RACK_D_TIPS, JUNIOR_LAB['SLOT 2-3-3'], MRV_VIALS,
                         JUNIOR_LAB['SLOT 2-3-1'], 10, 0.2)
ins_list9[0].preceding_instructions.append(ins_list8[-1].identifier)

ins_stir1 = JuniorInstruction(
    device=JUNIOR_LAB['SLOT 2-3-1'], action_name="wait", action_parameters={"wait_time": 300},
    description="wait for 5 min"
)
ins_stir2 = JuniorInstruction(
    device=JUNIOR_LAB['SLOT 2-3-2'], action_name="wait", action_parameters={"wait_time": 300},
    description="wait for 5 min"
)
ins_stir3 = JuniorInstruction(
    device=JUNIOR_LAB['SLOT 2-3-3'], action_name="wait", action_parameters={"wait_time": 300},
    description="wait for 5 min"
)

for i in [ins_stir1, ins_stir2, ins_stir3]:
    i.preceding_instructions.append(ins_list9[-1].identifier)

ins_list10 = needle_dispense(RSO2Cl_STOCK_SOLUTION_VIALS, JUNIOR_LAB['SLOT 2-3-2'], MRV_VIALS, JUNIOR_LAB['SLOT 2-3-1'],
                             10, 0.2)
ins_list10[0].preceding_instructions = [ins_stir1.identifier, ins_stir2.identifier, ins_stir3.identifier]

ins_stir21 = JuniorInstruction(
    device=JUNIOR_LAB['SLOT 2-3-1'], action_name="wait", action_parameters={"wait_time": 7200},
    description="wait for 120 min"
)
ins_stir22 = JuniorInstruction(
    device=JUNIOR_LAB['SLOT 2-3-2'], action_name="wait", action_parameters={"wait_time": 7200},
    description="wait for 120 min"
)
ins_stir23 = JuniorInstruction(
    device=JUNIOR_LAB['SLOT 2-3-3'], action_name="wait", action_parameters={"wait_time": 7200},
    description="wait for 120 min"
)

for i in [ins_stir21, ins_stir22, ins_stir23]:
    i.preceding_instructions.append(ins_list10[-1].identifier)

diagram = JUNIOR_LAB.instruction_graph

diagram.layout(algo="rt_circular")
diagram.dump_file(filename="sim_junior_instruction.drawio", folder="./")

from casymda_hardware.model import *
import simpy

env = simpy.Environment()

model = Model(env, JUNIOR_LAB, wdir=os.path.abspath("./"), model_name=f"con-{CONCURRENCY}")

env.run()
