import simpy

from casymda_hardware.model import *
from hardware_pydantic.junior.benchtop.sulfonylation_parallel import setup_benchtop_for_sulfonylation
from hardware_pydantic.junior.instruction_prototype import *

JUNIOR_BENCHTOP = create_junior_base()

#
#
# def pick_drop_rack_to(rack: JuniorRack, src_slot: JuniorSlot, dest_slot: JuniorSlot) -> list[JuniorInstruction]:
#     ins1 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#             "move_to_slot": JUNIOR_BENCHTOP.VPG_SLOT,
#         },
#         description=f"move to slot: {JUNIOR_BENCHTOP.VPG_SLOT.identifier}"
#     )
#
#     ins2 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
#         action_parameters={
#             "thing": JUNIOR_BENCHTOP.VPG,
#         },
#         description=f"pick up: {JUNIOR_BENCHTOP.VPG.identifier}"
#     )
#
#     ins3 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#             "move_to_slot": src_slot,
#         },
#         description=f"move to slot: {src_slot.identifier}"
#     )
#     ins4 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
#         action_parameters={
#             "thing": rack,
#         },
#         description=f"pick up: {rack.identifier}"
#     )
#     ins5 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#             "move_to_slot": dest_slot,
#         },
#         description=f"move to slot: {dest_slot.identifier}"
#     )
#     ins6 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
#         action_parameters={
#             "dest_slot": dest_slot,
#         },
#         description=f"put down: {dest_slot.identifier}"
#     )
#
#     ins7 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#             "move_to_slot": JUNIOR_BENCHTOP.VPG_SLOT,
#         },
#         description=f"move to slot: {JUNIOR_BENCHTOP.VPG_SLOT.identifier}"
#     )
#
#     ins8 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
#         action_parameters={
#             "dest_slot": JUNIOR_BENCHTOP.VPG_SLOT,
#         },
#         description=f"put down: {JUNIOR_BENCHTOP.VPG_SLOT.identifier}"
#     )
#
#     ins_list = [ins1, ins2, ins3, ins4, ins5, ins6, ins7, ins8]
#     JuniorInstruction.path_graph(ins_list)
#     return ins_list
#
#
# def solid_dispense(
#         sv_vial: JuniorVial,
#         sv_vial_slot: JuniorSlot,
#         dest_vials: list[JuniorVial],
#         amount: float,
#         include_pickup_svtool=True,
#         include_dropoff_svvial=True,
#         include_dropoff_svtool=True,
# ):
#     ins3 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#             "move_to_slot": sv_vial_slot,
#         },
#         description=f"move to slot: {sv_vial_slot.identifier}"
#     )
#
#     ins4 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
#         action_parameters={"thing": sv_vial},
#         description=f"pick up: {sv_vial.identifier}",
#     )
#
#     ins5 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#             "move_to_slot": JUNIOR_BENCHTOP.BALANCE,
#         },
#         description=f"move to slot: {JUNIOR_BENCHTOP.BALANCE.identifier}"
#     )
#
#     if include_pickup_svtool:
#         ins1 = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#             action_parameters={
#                 "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#                 "move_to_slot": JUNIOR_BENCHTOP.SV_TOOL_SLOT,
#             },
#             description=f"move to slot: {JUNIOR_BENCHTOP.SV_TOOL_SLOT.identifier}"
#         )
#
#         ins2 = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
#             action_parameters={"thing": JUNIOR_BENCHTOP.SV_TOOL},
#             description=f"pick up: {JUNIOR_BENCHTOP.SV_TOOL.identifier}",
#         )
#         ins_list = [ins1, ins2, ins3, ins4, ins5]
#     else:
#         ins_list = [ins3, ins4, ins5]
#
#     for dest_vial in dest_vials:
#         ins6 = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_Z2, action_name="dispense_sv",
#             action_parameters={
#                 "destination_container": dest_vial,
#                 "amount": amount,
#                 # "dispense_speed": speed,
#             },
#             description=f"dispense_sv to: {dest_vial.identifier}",
#         )
#         ins_list.append(ins6)
#
#     if include_dropoff_svvial:
#         ins7 = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#             action_parameters={
#                 "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#                 "move_to_slot": sv_vial_slot,
#             },
#             description=f"move to slot: {sv_vial_slot.identifier}"
#         )
#
#         ins8 = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
#             action_parameters={
#                 "dest_slot": sv_vial_slot,
#             },
#             description=f"put down: {sv_vial_slot.identifier}"
#         )
#         ins_list.append(ins7)
#         ins_list.append(ins8)
#
#     if include_dropoff_svtool:
#         ins9 = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#             action_parameters={
#                 "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#                 "move_to_slot": JUNIOR_BENCHTOP.SV_TOOL_SLOT,
#             },
#             description=f"move to slot: {JUNIOR_BENCHTOP.SV_TOOL_SLOT.identifier}"
#         )
#
#         ins10 = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
#             action_parameters={
#                 "dest_slot": JUNIOR_BENCHTOP.SV_TOOL_SLOT,
#             },
#             description=f"put down: {JUNIOR_BENCHTOP.SV_TOOL_SLOT.identifier}"
#         )
#         ins_list.append(ins9)
#         ins_list.append(ins10)
#
#     JuniorInstruction.path_graph(ins_list)
#
#     return ins_list
#
#
# def needle_dispense(
#         src_vials: list[JuniorVial],
#         src_slot: JuniorSlot,
#         dest_vials: list[JuniorVial],
#         dest_vials_slot: JuniorSlot,
#         amounts: list[float],
# ):
#     z1_needles = [JUNIOR_LAB[f"Z1 Needle {i + 1}"] for i in range(len(amounts))]
#     ins1 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z1,
#             "move_to_slot": src_slot,
#         },
#         description=f"move to slot: {src_slot.identifier}"
#     )
#     ins2 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z1, action_name="concurrent_aspirate",
#         action_parameters={
#             "source_containers": src_vials,
#             "dispenser_containers": z1_needles,
#             "amounts": amounts,
#         },
#         description=f"concurrent aspirate from: {','.join([v.identifier for v in src_vials])}"
#     )
#     ins3 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z1,
#             "move_to_slot": dest_vials_slot,
#         },
#         description=f"move to slot: {dest_vials_slot.identifier}"
#     )
#     ins4 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z1, action_name="concurrent_dispense",
#         action_parameters={
#             "destination_containers": dest_vials,
#             "dispenser_containers": z1_needles,
#             "amounts": amounts,
#         },
#         description=f"concurrent dispense to: {','.join([v.identifier for v in dest_vials])}"
#     )
#     ins5 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z1,
#             "move_to_slot": JUNIOR_BENCHTOP.WASH_BAY,
#         },
#         description=f"move to slot: WASH BAY"
#     )
#     ins6 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z1, action_name="wash",
#         action_parameters={
#             "wash_bay": JUNIOR_BENCHTOP.WASH_BAY,
#         },
#         description="wash needles"
#     )
#     ins_list = [ins1, ins2, ins3, ins4, ins5, ins6]
#     JuniorInstruction.path_graph(ins_list)
#     return ins_list
#
#
# def pdp_dispense(
#         src_vial: JuniorVial, src_slot: JuniorSlot,
#         tips: list[JuniorPdpTip], tips_slot: JuniorSlot,
#         dest_vials: list[JuniorVial], dest_vials_slot: JuniorSlot,
#         amount: float,
#         # speed: float
# ):
#     ins1 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#         action_parameters={
#             "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#             "move_to_slot": JUNIOR_LAB['PDT SLOT 1'],
#         },
#         description=f"move to slot: PDT SLOT 1"
#     )
#
#     ins2 = JuniorInstruction(
#         device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
#         action_parameters={"thing": JUNIOR_BENCHTOP.PDP_1},
#         description=f"pick up: {JUNIOR_BENCHTOP.PDP_1.identifier}",
#     )
#
#     ins_list = [ins1, ins2]
#
#     for tip, dest_vial in zip(tips, dest_vials):
#         i_a = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#             action_parameters={
#                 "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#                 "move_to_slot": tips_slot,
#             },
#             description=f"move to slot: {tips_slot.identifier}"
#         )
#         i_b = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
#             action_parameters={"thing": tip},
#             description=f"pick up: {tip.identifier}",
#         )
#         i_c = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#             action_parameters={
#                 "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#                 "move_to_slot": src_slot,
#             },
#             description=f"move to slot: {src_slot.identifier}"
#         )
#         i_d = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_Z2, action_name="aspirate_pdp",
#             action_parameters={
#                 "source_container": src_vial,
#                 "amount": amount,
#                 # "aspirate_speed": speed,
#             },
#             description=f"aspirate_pdp from: {src_vial.identifier} amount: {amount}"
#         )
#         i_e = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#             action_parameters={
#                 "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#                 "move_to_slot": dest_vials_slot,
#             },
#             description=f"move to slot: {dest_vials_slot.identifier}"
#         )
#         i_f = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_Z2, action_name="dispense_pdp",
#             action_parameters={
#                 "destination_container": dest_vial,
#                 "amount": amount,
#                 # "dispense_speed": speed,
#             },
#             description=f"dispense_pdp to: {dest_vial.identifier}"
#         )
#         i_g = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
#             action_parameters={
#                 "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
#                 "move_to_slot": JUNIOR_LAB['DISPOSAL'],
#             },
#             description=f"move to slot: DISPOSAL"
#         )
#         i_h = JuniorInstruction(
#             device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
#             action_parameters={
#                 "dest_slot": JUNIOR_LAB['DISPOSAL'],
#             },
#             description="put down: DISPOSAL"
#         )
#         ins_list += [i_a, i_b, i_c, i_d, i_e, i_f, i_g, i_h]
#     JuniorInstruction.path_graph(ins_list)
#     return ins_list
#

SULFONYL_BENCHTOP = setup_benchtop_for_sulfonylation(
    junior_benchtop=JUNIOR_BENCHTOP,
    dcm_init_volume=10,  # in mL

    sulfonyl_init_amount=5,  # in g
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
    ins_list7 = needle_dispense(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.DCM_VIALS[:-1], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                SULFONYL_BENCHTOP.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                [1, ] * len(SULFONYL_BENCHTOP.REACTOR_VIALS))
    ins_list8 = needle_dispense(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.DCM_VIALS[-1:], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                [SULFONYL_BENCHTOP.SULFONYL_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2, [1, ])
    lst = ins_list7 + ins_list8
    return lst

@ins_list_path_graph
def get_ins_lst_reaction():
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
    ins_lst_sulfonyl_dispense = get_ins_lst_sulfonyl_dispense()
    ins_lst_amine_dispense = get_ins_lst_amine_dispense()
    ins_lst_dcm_dispense = get_ins_lst_dcm_dispense()
    ins_lst_pyridine_dispense = pdp_dispense(JUNIOR_BENCHTOP, SULFONYL_BENCHTOP.PYRIDINE_VIAL, JUNIOR_BENCHTOP.SLOT_2_3_2,
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

    diagram = JUNIOR_LAB.instruction_graph
    diagram.layout(algo="rt_circular")
    diagram.dump_file(filename=f"{name}.drawio", folder="./")
    env = simpy.Environment()
    Model(env, JUNIOR_LAB, wdir=os.path.abspath("./"), model_name=name)
    env.run()

if __name__ == '__main__':
    import os.path

    simulate(os.path.basename(__file__).rstrip(".py"))
