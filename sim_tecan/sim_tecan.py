from hardware_pydantic.tecan import *
"""
randomly adding three chemicals to a plate
"""

create_tecan_base()


PLATE_1, PLATE_1_WELLS = TecanPlate.create_plate_with_empty_wells(n_wells=96, plate_id="PLATE 1")
PLATE_2, PLATE_2_WELLS = TecanPlate.create_plate_with_empty_wells(n_wells=96, plate_id="PLATE 2")
PLATE_3, PLATE_3_WELLS = TecanPlate.create_plate_with_empty_wells(n_wells=96, plate_id="PLATE 3")

HEATER_1 = TECAN_LAB['HEATER-1']
HEATER_2 = TECAN_LAB['HEATER-2']

TANK_1 = TECAN_LAB['TANK-1']
TANK_2 = TECAN_LAB['TANK-2']
TANK_3 = TECAN_LAB['TANK-3']

DISPENSING_SLOT_1 = TECAN_LAB['D-SLOT-1']
DISPENSING_SLOT_2 = TECAN_LAB['D-SLOT-2']
DISPENSING_SLOT_3 = TECAN_LAB['D-SLOT-3']

TecanSlot.put_plate_in_a_slot(PLATE_1, DISPENSING_SLOT_1)
TecanSlot.put_plate_in_a_slot(PLATE_2, DISPENSING_SLOT_2)
TecanSlot.put_plate_in_a_slot(PLATE_3, DISPENSING_SLOT_3)

ARM_1 = TECAN_LAB['ARM-1']
ARM_2 = TECAN_LAB['ARM-2']
ARM_1.position_on_top_of = TANK_1.identifier
ARM_2.position_on_top_of = HEATER_2.identifier

TANK_1.chemical_content["CHEMICAL A"] = 100
TANK_2.chemical_content["CHEMICAL B"] = 100
TANK_3.chemical_content["CHEMICAL C"] = 100

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def pick_drop_plate_to(plate: TecanPlate, dest_slot: TecanSlot, dest_slot_key: str="SLOT"):
    src_slot = TECAN_LAB[plate.contained_by]
    ins1 = TecanInstruction(
        device=ARM_2, action_parameters={"move_to_slot": src_slot}, action_name="move_to",
        description=f"{ARM_2.identifier} move to slot: {src_slot.identifier}",
    )
    ins2 = TecanInstruction(
        device=ARM_2, action_parameters={"thing": plate}, action_name="pick_up_plate",
        description=f"{ARM_2.identifier} pick up: {plate.identifier}",
    )
    ins3 = TecanInstruction(
        device=ARM_2, action_parameters={"move_to_slot": dest_slot}, action_name="move_to",
        description=f"{ARM_2.identifier} move to slot: {dest_slot.identifier}",
    )
    ins4 = TecanInstruction(
        device=ARM_2, action_parameters={"dest_slot": dest_slot, "dest_slot_key": dest_slot_key}, action_name="put_down_plate",
        description=f"put down: {dest_slot.identifier}",
    )
    ins_list = [ins1, ins2, ins3, ins4]
    TecanInstruction.path_graph(ins_list)
    return ins_list


def needle_dispense(
        src_tank: TecanLiquidTank,
        dest_wells: list[TecanPlateWell],
        amounts_asp: list[float],
        amounts_disp: list[float],
        speed: float,

        skip_src_tank = False, skip_wash=False,
):
    needles = [TECAN_LAB[v] for v in ARM_1.slot_content.values()]
    # TODO IMPORTANT! the following is dangerous as the slot is evaluated before the simulation
    dest_wells_slot = TECAN_LAB[dest_wells[0].contained_by].contained_by
    dest_wells_slot = TECAN_LAB[dest_wells_slot]
    concurrency = len(needles)

    if not skip_src_tank:
        ins1 = TecanInstruction(
            device=ARM_1, action_parameters={"move_to_slot": src_tank}, action_name="move_to",
            description=f"{ARM_1.identifier} move to slot: {src_tank.identifier}",
        )
        ins2 = TecanInstruction(
            device=ARM_1, action_name="concurrent_aspirate",
            action_parameters={
                "source_container": src_tank,
                "dispenser_containers": needles,
                "amounts": amounts_asp,
                "aspirate_speed": speed,
            },
            description=f"concurrent aspirate from: {src_tank.identifier}"
        )
        ins_list = [ins1, ins2]
    else:
        ins_list = []


    ins3 = TecanInstruction(
        device=ARM_1, action_parameters={"move_to_slot": dest_wells_slot}, action_name="move_to",
        description=f"{ARM_1.identifier} move to slot: {dest_wells_slot.identifier}",
    )

    ins_dispense = []
    for chunk in chunks(dest_wells, concurrency):
        ins_d = TecanInstruction(
        device=ARM_1, action_name="concurrent_dispense",
        action_parameters={
            "destination_containers": chunk,
            "dispenser_containers": needles,
            "dispense_speed": speed,
            "amounts": amounts_disp,
        },
        description=f"concurrent dispense to: {','.join([v.identifier for v in dest_wells])}"
    )
        ins_dispense.append(ins_d)

    ins_list += [ins3, ]
    ins_list += ins_dispense

    if not skip_wash:
        ins5 = TecanInstruction(
            device=ARM_1, action_name="move_to",
            action_parameters={
                "move_to_slot": TECAN_LAB['WASH-BAY'],
            },
            description=f"{ARM_1.identifier} move to slot: WASH-BAY"
        )
        ins6 = TecanInstruction(
            device=ARM_1, action_name="wash",
            action_parameters={
                "wash_bay": TECAN_LAB['WASH-BAY'],
            },
            description="wash needles"
        )
        ins_list += [ins5, ins6]
    TecanInstruction.path_graph(ins_list)
    return ins_list

def heat_plate(plate: TecanPlate, heater: TecanSlot, wait_time: float = 60):

    ins_lst = pick_drop_plate_to(plate, heater)
    ins = TecanInstruction(
        device=HEATER_1, action_name="wait", action_parameters={"wait_time": wait_time},
        description=f"heat {plate.identifier} for {wait_time} s on {heater.identifier}",
    )
    ins_lst.append(ins)
    TecanInstruction.path_graph(ins_lst)
    return ins_lst

INS_LST1 = needle_dispense(src_tank=TANK_1, dest_wells=PLATE_1_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=False, skip_wash=True)
INS_LST2 = needle_dispense(src_tank=TANK_1, dest_wells=PLATE_2_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=True, skip_wash=True)
INS_LST3 = needle_dispense(src_tank=TANK_1, dest_wells=PLATE_3_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=True, skip_wash=False)

INS_LST4 = needle_dispense(src_tank=TANK_2, dest_wells=PLATE_1_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=False, skip_wash=True)
INS_LST5 = needle_dispense(src_tank=TANK_2, dest_wells=PLATE_2_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=True, skip_wash=True)
INS_LST6 = needle_dispense(src_tank=TANK_2, dest_wells=PLATE_3_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=True, skip_wash=False)

INS_LST7 = needle_dispense(src_tank=TANK_3, dest_wells=PLATE_1_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=False, skip_wash=True)
INS_LST8 = needle_dispense(src_tank=TANK_3, dest_wells=PLATE_2_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=True, skip_wash=True)
INS_LST9 = needle_dispense(src_tank=TANK_3, dest_wells=PLATE_3_WELLS,amounts_asp=[40,] * 8, amounts_disp=[.1,] * 8, speed=5, skip_src_tank=True, skip_wash=False)

TecanInstruction.path_graph(
    INS_LST1 + INS_LST2 + INS_LST3 + INS_LST4 + INS_LST5 + INS_LST6 + INS_LST7 + INS_LST8 + INS_LST9
)

# INS_LST10 = heat_plate(PLATE_1, HEATER_1)
# INS_LST10[0].preceding_instructions.append(INS_LST7[-1].identifier)
#
# INS_LST11 = heat_plate(PLATE_2, HEATER_2)
# INS_LST11[0].preceding_instructions.append(INS_LST8[-1].identifier)
#
# INS_LST12 = heat_plate(PLATE_3, HEATER_1)
# INS_LST12[0].preceding_instructions.append(INS_LST9[-3].identifier)

diagram = TECAN_LAB.instruction_graph

diagram.layout(algo="rt_circular")
diagram.dump_file(filename="sim_tecan_instruction.drawio", folder="./")

from casymda_hardware.model import *
import simpy

env = simpy.Environment()

model = Model(env, TECAN_LAB, wdir=os.path.abspath("./"), model_name=f"tecan_dummy")

env.run()
