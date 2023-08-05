import simpy

from casymda_hardware.model import *
from hardware_pydantic.junior.benchtop.tips_pn_quinone import setup_quinone_benchtop
from hardware_pydantic.junior.instruction_prototype import *

JUNIOR_BENCHTOP = create_junior_base()

REACTION_BENCHTOP = setup_quinone_benchtop(
    # TODO change to reasonable amounts
    junior_benchtop=JUNIOR_BENCHTOP,
    n_reactors=4,
    water_init_volume=15,
    ethanol_init_volume=15,
    diketone_init_amount=100,
    naoh_init_amount=100,
    aldehyde_init_amount=100,
)


@ins_list_path_graph
def get_ins_lst_solid_dispense():  # naoh (2), diketone (0), aldehyde (1)
    ins_list1 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.RACK_REACTANT, JUNIOR_BENCHTOP.SLOT_2_3_2,
                                  JUNIOR_BENCHTOP.BALANCE)
    ins_list2_1 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.NAOH_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[2],
        dest_vials=[REACTION_BENCHTOP.NAOH_VIAL, ],
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list2_2 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.DIKETONE_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[0],
        dest_vials=[REACTION_BENCHTOP.DIKETONE_VIAL, ],
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list3 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.RACK_REACTANT, JUNIOR_BENCHTOP.BALANCE,
                                  JUNIOR_BENCHTOP.SLOT_2_3_2, )
    ins_list4 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.RACK_REACTOR, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                  JUNIOR_BENCHTOP.BALANCE)
    ins_list5 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.ALDEHYDE_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[1],
        dest_vials=REACTION_BENCHTOP.REACTOR_VIALS,
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list6 = pick_drop_rack_to(JUNIOR_BENCHTOP,
                                  REACTION_BENCHTOP.RACK_REACTOR, JUNIOR_BENCHTOP.BALANCE,
                                  JUNIOR_BENCHTOP.SLOT_2_3_1, )
    lst = ins_list1 + ins_list2_1 + ins_list2_2 + ins_list3 + ins_list4 + ins_list5 + ins_list6
    return lst


@ins_list_path_graph
def get_ins_lst_liquid_dispense():
    # water to naoh
    ins_list7 = needle_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.WATER_VIALS, JUNIOR_BENCHTOP.SLOT_OFF_1,
                                [REACTION_BENCHTOP.NAOH_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2,
                                [1, ])
    # ethanol to diketone
    ins_list8 = needle_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.ETHANOL_VIALS[-1:], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                [REACTION_BENCHTOP.DIKETONE_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2, [1, ])
    # ethanol to aldehyde
    ins_list9 = needle_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.ETHANOL_VIALS[:-1], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                REACTION_BENCHTOP.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_2,
                                [1, ] * len(REACTION_BENCHTOP.REACTOR_VIALS))

    lst = ins_list7 + ins_list8 + ins_list9
    return lst


@ins_list_path_graph
def get_ins_lst_reaction():
    ins_cap = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 30 * len(REACTION_BENCHTOP.REACTOR_VIALS)},
        description="capping reactors"
    )
    ins_reaction = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait", action_parameters={"wait_time": 60 * 240},
        description="wait for 240 min"
    )
    lst = [ins_cap, ins_reaction]
    return lst


def define_instructions():
    ins_lst_solid_dispense = get_ins_lst_solid_dispense()
    ins_lst_liquid_dispense = get_ins_lst_liquid_dispense()
    ins_lst_diketone_sln_dispense = pdp_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.DIKETONE_VIAL,
                                                 JUNIOR_BENCHTOP.SLOT_2_3_2,
                                                 REACTION_BENCHTOP.PDP_TIPS[: len(REACTION_BENCHTOP.REACTOR_VIALS)],
                                                 JUNIOR_BENCHTOP.SLOT_2_3_3, REACTION_BENCHTOP.REACTOR_VIALS,
                                                 JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)
    ins_lst_stir = [JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait", action_parameters={"wait_time": 300},
        description="wait for 5 min"
    ), ]
    # TODO there is a redundant `move_to` action to pick up pdp, even tho pdp is already on z2
    ins_lst_naoh_sln_dispense = pdp_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.NAOH_VIAL, JUNIOR_BENCHTOP.SLOT_2_3_2,
                                             REACTION_BENCHTOP.PDP_TIPS[len(REACTION_BENCHTOP.REACTOR_VIALS):],
                                             JUNIOR_BENCHTOP.SLOT_2_3_3, REACTION_BENCHTOP.REACTOR_VIALS,
                                             JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)

    ins_lst_reaction = get_ins_lst_reaction()

    chain_ins_lol(
        [
            ins_lst_solid_dispense,
            ins_lst_liquid_dispense,
            ins_lst_diketone_sln_dispense,
            ins_lst_stir,
            ins_lst_naoh_sln_dispense,
            ins_lst_reaction,
        ]
    )


def simulate(name):
    define_instructions()

    diagram = JUNIOR_LAB.instruction_graph
    diagram.layout(algo="rt_circular")
    diagram.dump_file(filename=f"{name}.drawio", folder="./")
    env = simpy.Environment()
    Model(env, JUNIOR_LAB, wdir=os.path.abspath("./"), model_name=name)
    env.run()


if __name__ == '__main__':
    import os.path

    simulate(os.path.basename(__file__).rstrip(".py"))
