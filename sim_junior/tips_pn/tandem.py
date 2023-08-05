from hardware_pydantic.junior.benchtop.tips_pn_tandem import setup_tips_pn_benchtop
from hardware_pydantic.junior.instruction_prototype import *
import simpy
from casymda_hardware.model import *

JUNIOR_BENCHTOP = create_junior_base()

REACTION_BENCHTOP = setup_tips_pn_benchtop(
    # TODO change to reasonable amounts
    junior_benchtop=JUNIOR_BENCHTOP,

    quinone_n_reactors=4,
    quinone_water_init_volume=15,
    quinone_ethanol_init_volume=15,
    quinone_diketone_init_amount=100,
    quinone_naoh_init_amount=100,
    quinone_aldehyde_init_amount=100,

    n_reactors=4,
    thf_init_volume=15,
    hcl_init_volume=15,
    silyl_init_volume=15,
    grignard_init_amount=100,
)


@ins_list_path_graph
def get_ins_lst_solid_dispense():  # naoh (2), diketone (0), aldehyde (1)
    ins_list1 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.quinone_benchtop.RACK_REACTANT, JUNIOR_BENCHTOP.SLOT_2_3_2,
                                  JUNIOR_BENCHTOP.BALANCE)
    ins_list2_1 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.quinone_benchtop.NAOH_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[2],
        dest_vials=[REACTION_BENCHTOP.quinone_benchtop.NAOH_VIAL, ],
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list2_2 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.quinone_benchtop.DIKETONE_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[0],
        dest_vials=[REACTION_BENCHTOP.quinone_benchtop.DIKETONE_VIAL, ],
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list3 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.quinone_benchtop.RACK_REACTANT, JUNIOR_BENCHTOP.BALANCE,
                                  JUNIOR_BENCHTOP.SLOT_2_3_2, )
    ins_list4 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.quinone_benchtop.RACK_REACTOR, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                  JUNIOR_BENCHTOP.BALANCE)
    ins_list5 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.quinone_benchtop.ALDEHYDE_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[1],
        dest_vials=REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS,
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list6 = pick_drop_rack_to(JUNIOR_BENCHTOP,
                                  REACTION_BENCHTOP.quinone_benchtop.RACK_REACTOR, JUNIOR_BENCHTOP.BALANCE,
                                  JUNIOR_BENCHTOP.SLOT_2_3_1, )
    lst = ins_list1 + ins_list2_1 + ins_list2_2 + ins_list3 + ins_list4 + ins_list5 + ins_list6
    return lst


@ins_list_path_graph
def get_ins_lst_liquid_dispense():
    # water to naoh
    ins_list7 = needle_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.quinone_benchtop.WATER_VIALS, JUNIOR_BENCHTOP.SLOT_OFF_1,
                                [REACTION_BENCHTOP.quinone_benchtop.NAOH_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2,
                                [1, ])
    # ethanol to diketone
    ins_list8 = needle_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.quinone_benchtop.ETHANOL_VIALS[-1:], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                [REACTION_BENCHTOP.quinone_benchtop.DIKETONE_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2, [1, ])
    # ethanol to aldehyde
    ins_list9 = needle_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.quinone_benchtop.ETHANOL_VIALS[:-1], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_2,
                                [1, ] * len(REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS))

    lst = ins_list7 + ins_list8 + ins_list9
    return lst


@ins_list_path_graph
def get_ins_lst_reaction():
    ins_cap = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 30 * len(REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS)},
        description="capping reactors"
    )
    ins_reaction = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait", action_parameters={"wait_time": 60 * 240},
        description="wait for 240 min"
    )
    lst = [ins_cap, ins_reaction]
    return lst



@ins_list_path_graph
def get_ins_lst_ef():  # quinone (0), grignard (1)
    ins_list1 = pick_drop_rack_to(
        JUNIOR_BENCHTOP, REACTION_BENCHTOP.grignard_benchtop.RACK_REACTANT, JUNIOR_BENCHTOP.SLOT_2_3_2, JUNIOR_BENCHTOP.BALANCE
    )
    ins_list2 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.grignard_benchtop.QUINONE_SVV, sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[10], dest_vials=[REACTION_BENCHTOP.grignard_benchtop.QUINONE_VIAL, ], amount=0.5, include_pickup_svtool=True, include_dropoff_svvial=True, include_dropoff_svtool=True
    )
    ins_list3 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.grignard_benchtop.RACK_REACTANT, JUNIOR_BENCHTOP.BALANCE,
                                  JUNIOR_BENCHTOP.SLOT_2_3_2, )
    ins_list4 = needle_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.grignard_benchtop.THF_VIALS[-1:], JUNIOR_BENCHTOP.SLOT_OFF_2,
                                [REACTION_BENCHTOP.grignard_benchtop.QUINONE_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2, [1, ])
    lst = ins_list1 + ins_list2 + ins_list3 + ins_list4
    return lst


@ins_list_path_graph
def get_ins_lst_abcd():
    # a
    ins_list1 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.grignard_benchtop.RACK_REACTOR, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                  JUNIOR_BENCHTOP.BALANCE)
    ins_list2 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.grignard_benchtop.GRIGNARD_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[11],
        dest_vials=REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS,
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list3 = pick_drop_rack_to(
        JUNIOR_BENCHTOP,
        REACTION_BENCHTOP.grignard_benchtop.RACK_REACTOR, JUNIOR_BENCHTOP.BALANCE,
        JUNIOR_BENCHTOP.SLOT_2_3_1, )

    # b
    ins_list4 = needle_dispense(
        JUNIOR_BENCHTOP,
        REACTION_BENCHTOP.grignard_benchtop.THF_VIALS[:-1], JUNIOR_BENCHTOP.SLOT_OFF_2,
        REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_1,
        [1, ] * len(REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS))
    # c
    ins_list5 = pdp_dispense(
        JUNIOR_BENCHTOP,
        REACTION_BENCHTOP.grignard_benchtop.SILYL_VIAL, JUNIOR_BENCHTOP.SLOT_2_3_2,
        REACTION_BENCHTOP.grignard_benchtop.PDP_TIPS[: len(REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS)],
        JUNIOR_BENCHTOP.SLOT_2_3_3, REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS,
        JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)

    # d
    ins_cap = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 30 * len(REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS)},
        description="capping reactors"
    )
    ins_heat = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 15 * 60},
        description="60 C wait for 15 min"
    )
    lst = ins_list1 + ins_list2 + ins_list3 + ins_list4 + ins_list5 + [ins_cap, ins_heat]
    return lst


@ins_list_path_graph
def get_ins_lst_ghi():
    # g
    ins_listg = pdp_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.grignard_benchtop.QUINONE_VIAL, JUNIOR_BENCHTOP.SLOT_2_3_2,
                             REACTION_BENCHTOP.grignard_benchtop.PDP_TIPS[len(REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS):],
                             JUNIOR_BENCHTOP.SLOT_2_3_3, REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS,
                             JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)

    # h
    ins_cap = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 30 * len(REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS)},
        description="capping reactors"
    )
    ins_heat = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 15 * 60},
        description="60 C wait for 30 min"
    )

    # i
    ins_listi = needle_dispense(
        JUNIOR_BENCHTOP,
        REACTION_BENCHTOP.grignard_benchtop.HCL_VIALS, JUNIOR_BENCHTOP.SLOT_OFF_1,
        REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_1,
        [1, ] * len(REACTION_BENCHTOP.grignard_benchtop.REACTOR_VIALS))

    lst = ins_listg + [ins_cap, ins_heat] + ins_listi
    return lst



@ins_list_path_graph
def get_ins_lst_magic():
    # TODO you can make these parallel if you already have magic...
    lst = []
    for v in REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS:
        ins = JuniorInstruction(
            device=REACTION_BENCHTOP.mage,
            action_name="set_vial_chemical_content",
            action_parameters={
                "vial": v,
                "chemical": dict(),
                "time_cost": 300,
            },
            description=f"magic empty from: {v.identifier}"
        )
        lst.append(ins)
    v = REACTION_BENCHTOP.grignard_benchtop.QUINONE_SVV
    ins = JuniorInstruction(
        device=REACTION_BENCHTOP.mage,
        action_name="set_vial_chemical_content",
        action_parameters={
            "vial": v,
            "chemical": {'Quinone': REACTION_BENCHTOP.quinone_transferred},
            "time_cost": 100,
        },
        description=f"magic add to: {v.identifier}"
    )
    lst.append(ins)
    return lst


def define_instructions():
    ins_lst_solid_dispense = get_ins_lst_solid_dispense()
    ins_lst_liquid_dispense = get_ins_lst_liquid_dispense()
    ins_lst_diketone_sln_dispense = pdp_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.quinone_benchtop.DIKETONE_VIAL,
                                                 JUNIOR_BENCHTOP.SLOT_2_3_2,
                                                 REACTION_BENCHTOP.quinone_benchtop.PDP_TIPS[: len(REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS)],
                                                 JUNIOR_BENCHTOP.SLOT_2_3_3, REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS,
                                                 JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)
    ins_lst_stir = [JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait", action_parameters={"wait_time": 300},
        description="wait for 5 min"
    ), ]
    # TODO there is a redundant `move_to` action to pick up pdp, even tho pdp is already on z2
    ins_lst_naoh_sln_dispense = pdp_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.quinone_benchtop.NAOH_VIAL, JUNIOR_BENCHTOP.SLOT_2_3_2,
                                             REACTION_BENCHTOP.quinone_benchtop.PDP_TIPS[len(REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS):],
                                             JUNIOR_BENCHTOP.SLOT_2_3_3, REACTION_BENCHTOP.quinone_benchtop.REACTOR_VIALS,
                                             JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)

    ins_lst_reaction = get_ins_lst_reaction()

    ins_lst_magic = get_ins_lst_magic()

    ins_lst_abcd = get_ins_lst_abcd()
    ins_lst_ef = get_ins_lst_ef()
    ins_lst_ghi = get_ins_lst_ghi()


    chain_ins_lol(
        [
            ins_lst_solid_dispense,
            ins_lst_liquid_dispense,
            ins_lst_diketone_sln_dispense,
            ins_lst_stir,
            ins_lst_naoh_sln_dispense,
            ins_lst_reaction,
            ins_lst_magic,
            ins_lst_abcd,
            ins_lst_ef,
            ins_lst_ghi
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
