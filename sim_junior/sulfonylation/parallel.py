import json

import simpy

from casymda_hardware.model import *
from hardware_pydantic.junior.benchtop.sulfonylation_parallel import setup_benchtop_for_sulfonylation
from hardware_pydantic.junior.instruction_prototype import *

# setup physical devices specific to the platform, such as robot arms and container slots.
JUNIOR_BENCHTOP = create_junior_base()

# setup physical objects specific to this reaction, mainly consumables (chemicals, vials, also racks).
SULFONYL_BENCHTOP = setup_benchtop_for_sulfonylation(
    junior_benchtop=JUNIOR_BENCHTOP,
    dcm_init_volume=10,  # in mL

    sulfonyl_init_amount=5,  # in g
    # four different amines react with the same sulfonyl
    solid_amines={
        "solid_amine_1": 5,  # in g
        "solid_amine_2": 5,  # in g
        "solid_amine_3": 5,  # in g
        "solid_amine_4": 5,  # in g
    },
    pyridine_init_volume=10,  # in mL
)


@ins_list_path_graph
def get_ins_lst_sulfonyl_dispense():
    """ solid dispense sulfonly solid to prepare stock solution that will be added to reactor vials """
    ins_list1 = pick_drop_rack_to(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.RACK_REACTANT, JUNIOR_BENCHTOP.SLOT_2_3_2,
                                  JUNIOR_BENCHTOP.BALANCE)
    ins_list2 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=SULFONYL_BENCHTOP.SULFONYL_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[0],
        dest_vials=[SULFONYL_BENCHTOP.SULFONYL_VIAL, ],
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list3 = pick_drop_rack_to(
        JUNIOR_BENCHTOP,
        SULFONYL_BENCHTOP.RACK_REACTANT, JUNIOR_BENCHTOP.BALANCE,
        JUNIOR_BENCHTOP.SLOT_2_3_2, )
    lst = ins_list1 + ins_list2 + ins_list3
    return lst


@ins_list_path_graph
def get_ins_lst_amine_dispense():
    """ solid amines directly dispensed to reactor vials """
    ins_list_move_to = pick_drop_rack_to(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.RACK_REACTOR, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                         JUNIOR_BENCHTOP.BALANCE)
    ins_list_move_back = pick_drop_rack_to(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.RACK_REACTOR, JUNIOR_BENCHTOP.BALANCE,
                                           JUNIOR_BENCHTOP.SLOT_2_3_1, )

    dispense_lst = []
    for i in range(len(SULFONYL_BENCHTOP.AMINE_SVS)):
        amine_svv = SULFONYL_BENCHTOP.AMINE_SVS[i]
        amine_svv_slot = JUNIOR_BENCHTOP.SV_VIAL_SLOTS[i + 1]
        reactor_vial = SULFONYL_BENCHTOP.REACTOR_VIALS[i]
        ins_list_actual_dispense = solid_dispense(
            junior_benchtop=JUNIOR_BENCHTOP,
            sv_vial=amine_svv,
            sv_vial_slot=amine_svv_slot,
            dest_vials=[reactor_vial, ],
            amount=0.5,
            include_pickup_svtool=True,
            include_dropoff_svvial=True, include_dropoff_svtool=True)
        dispense_lst += ins_list_actual_dispense
    lst = ins_list_move_to + dispense_lst + ins_list_move_back
    return lst


@ins_list_path_graph
def get_ins_lst_dcm_dispense():
    """ add solvent to 4 (# of amines) reactor vials and one sulfonyl vial using needles on Z1 arm """
    ins_list7 = needle_dispense(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.DCM_VIALS[:-1], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                SULFONYL_BENCHTOP.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                [1, ] * len(SULFONYL_BENCHTOP.REACTOR_VIALS))
    ins_list8 = needle_dispense(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.DCM_VIALS[-1:], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                [SULFONYL_BENCHTOP.SULFONYL_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2, [1, ])
    lst = ins_list7 + ins_list8
    return lst


@ins_list_path_graph
def get_ins_lst_reaction():
    """ capping, stir and let the reactions proceed """
    ins_cap = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 30 * len(SULFONYL_BENCHTOP.REACTOR_VIALS)},
        description="capping reactors"
    )
    ins_stir = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait", action_parameters={"wait_time": 300},
        description="wait for 5 min"
    )
    ins_reaction = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait", action_parameters={"wait_time": 60 * 120},
        description="wait for 120 min"
    )
    lst = [ins_cap, ins_stir, ins_reaction]
    return lst


def define_instructions():
    """
    define all instructions, note instruction objects
    are automatically added to the `JuniorLab` once they are instantiated
    """
    ins_lst_sulfonyl_dispense = get_ins_lst_sulfonyl_dispense()
    ins_lst_amine_dispense = get_ins_lst_amine_dispense()
    ins_lst_dcm_dispense = get_ins_lst_dcm_dispense()
    ins_lst_pyridine_dispense = pdp_dispense(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.PYRIDINE_VIAL,
                                             JUNIOR_BENCHTOP.SLOT_2_3_2,
                                             SULFONYL_BENCHTOP.PDP_TIPS,
                                             JUNIOR_BENCHTOP.SLOT_2_3_3, SULFONYL_BENCHTOP.REACTOR_VIALS,
                                             JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)
    ins_lst_reaction = get_ins_lst_reaction()

    chain_ins_lol(
        [
            ins_lst_sulfonyl_dispense,
            ins_lst_amine_dispense,
            ins_lst_dcm_dispense,
            ins_lst_pyridine_dispense,
            ins_lst_reaction,
        ]
    )


def simulate(name):
    define_instructions()

    # dump as `drawio` editable
    diagram = JUNIOR_LAB.instruction_graph
    diagram.layout(algo="rt_circular")
    diagram.dump_file(filename=f"{name}.drawio", folder="./")

    # dump as json
    with open(f"{name}.json", "w") as f:
        json.dump([v.as_dict(identifier_only=True) for v in JUNIOR_LAB.dict_instruction.values()], f, indent=2)

    env = simpy.Environment()
    Model(env, JUNIOR_LAB, wdir=os.path.abspath("./"), model_name=name)
    env.run()


if __name__ == '__main__':
    import os.path

    simulate(os.path.basename(__file__).rstrip(".py"))
