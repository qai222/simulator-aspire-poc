import simpy

from casymda_hardware.model import *
from hardware_pydantic.junior.benchtop.tips_pn_grignard import setup_grignard_benchtop
from hardware_pydantic.junior.instruction_prototype import *

"""
making JA's molecule following 10.1021/ja0162459

# pentacenequinone
- original procedure: 10.1002/ange.19530652309
- actual procedure from: https://www.mdpi.com/1420-3049/17/4/4625
- skipping purification (black box)

1. procedure text:
Aqueous NaOH (10%, 5.96 g, 149 mmol) was slowly added to 
a solution of o-phthalaldehyde (10 g, 74.6 mmol) and 1,4-cyclohexanedione (4.18 g, 37.3 mmol) in ethanol (460 mL) 
under a N2 atmosphere. 
The solution turned from yellow to golden brown to dark brown before a yellow solid corresponding to 
pentacene-6,13-dione precipitated. After stirring the reaction mixture for four hours, 
the crude reaction mixture was filtered and washed with ethanol, water, and methanol until the washings were colorless.
 The solid residue was dried under vacuum to obtain 11.02 g (96% yield) of bright yellow pentacene-6,13-dione. 

2. benchtop setup:
fresh solution: 
- Aqueous NaOH (10%, 5.96 g, 149 mmol)
solvent:
- ethanol
- water
solid needed:
- o-phthalaldehyde (10 g, 74.6 mmol)
- 1,4-cyclohexanedione (4.18 g, 37.3 mmol)

"""

JUNIOR_BENCHTOP = create_junior_base()

REACTION_BENCHTOP = setup_grignard_benchtop(
    # TODO change to reasonable amounts
    junior_benchtop=JUNIOR_BENCHTOP,
    n_reactors=4,
    thf_init_volume=15,
    hcl_init_volume=15,
    silyl_init_volume=15,
    grignard_init_amount=100,
    quinone_init_amount=100,
)


@ins_list_path_graph
def get_ins_lst_ef():  # quinone (0), grignard (1)
    ins_list1 = pick_drop_rack_to(
        JUNIOR_BENCHTOP, REACTION_BENCHTOP.RACK_REACTANT, JUNIOR_BENCHTOP.SLOT_2_3_2, JUNIOR_BENCHTOP.BALANCE
    )
    ins_list2 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.QUINONE_SVV, sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[0], dest_vials=[REACTION_BENCHTOP.QUINONE_VIAL, ], amount=0.5, include_pickup_svtool=True, include_dropoff_svvial=True, include_dropoff_svtool=True
    )
    ins_list3 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.RACK_REACTANT, JUNIOR_BENCHTOP.BALANCE,
                                  JUNIOR_BENCHTOP.SLOT_2_3_2, )
    ins_list4 = needle_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.THF_VIALS[-1:], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                [REACTION_BENCHTOP.QUINONE_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2, [1, ])
    lst = ins_list1 + ins_list2 + ins_list3 + ins_list4
    return lst


@ins_list_path_graph
def get_ins_lst_abcd():
    # a
    ins_list1 = pick_drop_rack_to(JUNIOR_BENCHTOP, REACTION_BENCHTOP.RACK_REACTOR, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                  JUNIOR_BENCHTOP.BALANCE)
    ins_list2 = solid_dispense(
        junior_benchtop=JUNIOR_BENCHTOP,
        sv_vial=REACTION_BENCHTOP.GRIGNARD_SVV,
        sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[1],
        dest_vials=REACTION_BENCHTOP.REACTOR_VIALS,
        amount=0.5,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True)
    ins_list3 = pick_drop_rack_to(
        JUNIOR_BENCHTOP,
        REACTION_BENCHTOP.RACK_REACTOR, JUNIOR_BENCHTOP.BALANCE,
        JUNIOR_BENCHTOP.SLOT_2_3_1, )

    # b
    ins_list4 = needle_dispense(
        JUNIOR_BENCHTOP,
        REACTION_BENCHTOP.THF_VIALS[:-1], JUNIOR_BENCHTOP.SLOT_OFF_1,
        REACTION_BENCHTOP.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_1,
        [1, ] * len(REACTION_BENCHTOP.REACTOR_VIALS))
    # c
    ins_list5 = pdp_dispense(
        JUNIOR_BENCHTOP,
        REACTION_BENCHTOP.SILYL_VIAL, JUNIOR_BENCHTOP.SLOT_2_3_2,
        REACTION_BENCHTOP.PDP_TIPS[: len(REACTION_BENCHTOP.REACTOR_VIALS)],
        JUNIOR_BENCHTOP.SLOT_2_3_3, REACTION_BENCHTOP.REACTOR_VIALS,
        JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)

    # d
    ins_cap = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 30 * len(REACTION_BENCHTOP.REACTOR_VIALS)},
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
    ins_listg = pdp_dispense(JUNIOR_BENCHTOP, REACTION_BENCHTOP.QUINONE_VIAL, JUNIOR_BENCHTOP.SLOT_2_3_2,
                             REACTION_BENCHTOP.PDP_TIPS[len(REACTION_BENCHTOP.REACTOR_VIALS):],
                             JUNIOR_BENCHTOP.SLOT_2_3_3, REACTION_BENCHTOP.REACTOR_VIALS,
                             JUNIOR_BENCHTOP.SLOT_2_3_1, 0.04)

    # h
    ins_cap = JuniorInstruction(
        device=JUNIOR_BENCHTOP.SLOT_2_3_1, action_name="wait",
        action_parameters={"wait_time": 30 * len(REACTION_BENCHTOP.REACTOR_VIALS)},
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
        REACTION_BENCHTOP.HCL_VIALS, JUNIOR_BENCHTOP.SLOT_OFF_1,
        REACTION_BENCHTOP.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_1,
        [1, ] * len(REACTION_BENCHTOP.REACTOR_VIALS))

    lst = ins_listg + [ins_cap, ins_heat] + ins_listi
    return lst


def define_instructions():
    ins_lst_abcd = get_ins_lst_abcd()
    ins_lst_ef = get_ins_lst_ef()
    ins_lst_ghi = get_ins_lst_ghi()
    chain_ins_lol([ins_lst_abcd, ins_lst_ef, ins_lst_ghi])


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
