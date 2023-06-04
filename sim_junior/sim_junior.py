from hardware_pydantic.junior import *
from hardware_pydantic.junior import JuniorInstruction as Instruction


def junior_setup():
    junior_lab = create_junior_base()

    # RACK A: holding HRVs with DCM, on off deck initially
    rack_and_vials = JuniorRack.create_rack_with_empty_vials(n_vials=1, rack_capacity=1, vial_type="HRV",
                                                             rack_id="RACK A")
    RACK_A, RACK_A_VIALS = rack_and_vials[0], rack_and_vials[1:]
    RACK_A: JuniorRack
    RACK_A_VIALS: list[JuniorVial]
    for v in RACK_A_VIALS:
        v.content = {"DCM": 10}
    JuniorRack.put_rack_in_a_slot(RACK_A, junior_lab['RACK SLOT OFF-1'])

    # RACK B: holding HRVs for reactions, at 2-3-2 initially, one for RSO2Cl stock solution, another for pyridine source
    rack_and_vials = JuniorRack.create_rack_with_empty_vials(n_vials=2, rack_capacity=2, vial_type="HRV",
                                                             rack_id="RACK B")
    RACK_B, RACK_B_VIALS = rack_and_vials[0], rack_and_vials[1:]
    RACK_B: JuniorRack
    pyridine_vail = junior_lab[RACK_B.content['1']]
    pyridine_vail.content = {"pyridine": 10}
    JuniorRack.put_rack_in_a_slot(RACK_B, junior_lab['RACK SLOT 2'])

    # RACK C: holding one MRV for reaction, at 2-3-1 initially
    rack_and_vials = JuniorRack.create_rack_with_empty_vials(n_vials=1, rack_capacity=1, vial_type="MRV",
                                                             rack_id="RACK C")
    RACK_C, RACK_C_VIALS = rack_and_vials[0], rack_and_vials[1:]
    RACK_C: JuniorRack
    JuniorRack.put_rack_in_a_slot(RACK_C, junior_lab['RACK SLOT 1'])

    # IGNORE PDT TIP RACK FN

    SVV_1 = JuniorVial(identifier="SV VIAL 1", position_relative=junior_lab['SVV SLOT 1'].identifier, type="SV",
                       content={"solid amine": 10})
    SVV_2 = JuniorVial(identifier="SV VIAL 2", position_relative=junior_lab['SVV SLOT 2'].identifier, type="SV",
                       content={"sulfonyl chloride": 10})
    junior_lab['SVV SLOT 1'].content = SVV_1.identifier
    junior_lab['SVV SLOT 2'].content = SVV_2.identifier

    return junior_lab


def junior_instructions():
    junior_lab = junior_setup()

    j_rack_a = junior_lab['RACK A']
    j_rack_b = junior_lab['RACK B']
    j_rack_c = junior_lab['RACK C']
    j_rack_c: JuniorRack

    j_vial_rso2cl = junior_lab[j_rack_b.content['0']]
    j_vial_pyridine = junior_lab[j_rack_b.content['1']]
    j_vial_mrv = junior_lab[j_rack_c.content['0']]
    j_vial_dcm = junior_lab[j_rack_a.content['0']]

    j_sv_vial_1 = junior_lab['SV VIAL 1']
    j_sv_vial_2 = junior_lab['SV VIAL 2']
    j_z1_arm = junior_lab['Z1 ARM']
    j_z2_arm = junior_lab['Z2 ARM']
    j_sv_tool = junior_lab['SV TOOL']

    # TODO this completely ignore the fact you need first pick up the rack, put it down on the balance, then put the rack back...
    # 1a dispense RSO2Cl from `SV VIAL 2` to `HRV 1` on rack c
    ins_1a1 = Instruction(
        identifier="1a1",
        device=j_z2_arm, action_name="pick_up",
        action_parameters={
            "obj": j_sv_tool,
        },
        description="pick up sv tool",
    )

    ins_1a2 = Instruction(
        identifier="1a2",
        device=j_z2_arm, action_name="pick_up",
        action_parameters={
            "obj": j_sv_vial_2,
        }, preceding_instructions=[ins_1a1.identifier],
        description="pick up RSO2Cl sv vial",
    )

    ins_1a3 = Instruction(
        identifier="1a3",
        device=j_z2_arm, action_name="dispense_sv",
        action_parameters={
            "to_vial": j_vial_rso2cl,
            "amount": 0.1
        }, preceding_instructions=[ins_1a2.identifier],
        # TODO somewhere a bug making rack content dict[str, Vial]...
        description="dispense RSO2Cl to " + j_vial_rso2cl.identifier
    )

    # 1b dispense solid amine from `SV VIAL 2` to `HRV 2` on rack c
    # first put down RSO2Cl vial
    ins_1b1 = Instruction(
        identifier="1b1",
        device=j_z2_arm, action_name="put_down",
        action_parameters={
            "to_slot": junior_lab['SVV SLOT 2']
        },
        preceding_instructions=[ins_1a3.identifier, ins_1a1.identifier],
        description="put down RSO2Cl sv vial"
    )

    ins_1b2 = Instruction(
        identifier="1b2",
        device=j_z2_arm, action_name="pick_up",
        action_parameters={
            "obj": j_sv_vial_1,
        }, preceding_instructions=[ins_1b1.identifier],
        description="pick up solid amine sv vial"
    )

    ins_1b3 = Instruction(
        identifier="1b3",
        device=j_z2_arm, action_name="dispense_sv",
        action_parameters={
            "to_vial": j_vial_mrv,
            "amount": 0.1
        }, preceding_instructions=[ins_1b2.identifier],
        description="dispense solid amine to " + j_vial_mrv.identifier
    )

    ins_2a = Instruction(
        identifier="2a",
        device=j_z2_arm, action_name="put_down",
        action_parameters={
            "to_slot": junior_lab['SVV SLOT 1']
        }, preceding_instructions=[ins_1b3.identifier],
        description="put down sv vial this also free this slot for ARM Z1: " + j_z1_arm.identifier
    )

    ins_2b = Instruction(
        identifier="2b",
        device=j_z1_arm, action_name="transfer_liquid",
        action_parameters={
            "use_needles": ["1"],
            "from_vials": [j_vial_dcm],
            "to_vials": [j_vial_rso2cl],
            "amounts": [2.0],
        }, preceding_instructions=[ins_1a3.identifier, ins_1b3.identifier, ins_2a.identifier],
        description="dispense DCM to RSO2Cl vial to make stock solution"
    )

    ins_3 = Instruction(
        identifier="3",
        device=j_z1_arm, action_name="transfer_liquid",
        action_parameters={
            "use_needles": ["1"],
            "from_vials": [j_vial_dcm],
            "to_vials": [j_vial_mrv],
            "amounts": [2.0],
        }, preceding_instructions=[ins_1a3.identifier, ins_1b3.identifier],
        description="dispense DCM to mrv vial (has aniline)"
    )
    ins_4a = Instruction(
        identifier="4a",
        device=j_z2_arm, action_name="put_down",
        action_parameters={
            "to_slot": junior_lab['SV TOOL SLOT']
        }, preceding_instructions=[ins_2a.identifier],
        description="put down sv tool"
    )

    ins_4b = Instruction(
        identifier="4b",
        device=j_z2_arm, action_name="pick_up",
        action_parameters={
            "obj": junior_lab['PDT 1'],
        }, preceding_instructions=[ins_4a.identifier, ],
        description="pick up pdt 1 for pyridine"
    )

    ins_4c_dep = Instruction(
        identifier="4c_dep",
        device=j_z1_arm, action_name="move_to",
        action_parameters={
            "move_to_slot": junior_lab['RACK SLOT OFF-3']
        }, preceding_instructions=[ins_3.identifier],  # last time we use z1,
        description="move z1 to off deck so z2 has access"
    )

    ins_4c = Instruction(
        identifier="4c",
        device=j_z2_arm, action_name="transfer_liquid_pdt",
        action_parameters={
            "from_vial": j_vial_pyridine,
            "to_vial": j_vial_mrv,
            "amount": 0.05,
        }, preceding_instructions=[ins_4b.identifier, ins_4c_dep.identifier],
        description="use pdt to transfer pyridine"
    )

    # SKIP INS 5

    ins6 = Instruction(
        identifier="6",
        device=j_z2_arm, action_name="transfer_liquid_pdt",
        action_parameters={
            "from_vial": j_vial_rso2cl,
            "to_vial": j_vial_mrv,
            "amount": 1.0,
        }, preceding_instructions=[ins_4c.identifier, ins_2b.identifier],
        description="use pdt to transfer rso2cl stock solution"
    )

    ins7 = Instruction(
        identifier="7",
        device=junior_lab['RACK SLOT 1'], action_name="wait",
        action_parameters={"wait_time": 3600},
        preceding_instructions=[ins6.identifier, ],
        description="wait until reaction finishes",
    )

    return junior_lab


jlab = junior_instructions()

from N2G import drawio_diagram

diagram = drawio_diagram()
diagram.add_diagram("Page-1")
for ins in jlab.dict_instruction.values():
    diagram.add_node(id=f"{ins.identifier}\n{ins.description}")
for ins in jlab.dict_instruction.values():
    for dep in ins.preceding_instructions:
        pre_ins = jlab.dict_instruction[dep]
        this_ins_node = f"{ins.identifier}\n{ins.description}"
        pre_ins_node = f"{pre_ins.identifier}\n{pre_ins.description}"
        diagram.add_link(pre_ins_node, this_ins_node, style="endArrow=classic")
diagram.layout(algo="kk")
diagram.dump_file(filename="sim_junior_instruction.drawio", folder="./")

import pickle

with open(f"/home/qai/workplace/simulator-aspire-poc/sim_junior/lab_states/state_0.pkl", "wb") as f:
    output = {"simulation time": 0, "lab": jlab, "log": ["SIMULATION INITIALIZED"]}
    pickle.dump(output, f)

from casymda_hardware.model import *
import simpy

env = simpy.Environment()

model = Model(env, jlab)

env.run()
