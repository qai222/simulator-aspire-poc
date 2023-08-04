from hardware_pydantic.junior import *

"""
following the notes of N-Sulfonylation 
"""

JUNIOR_BENCHTOP = create_junior_base()


class SulfonylationBenchtop(BaseModel):
    RACK_SOLVENT: JuniorRack
    DCM_VIALS: list[JuniorVial]
    RACK_REACTANT: JuniorRack
    SULFONYL_VIAL: JuniorVial
    PYRIDINE_VIAL: JuniorVial
    RACK_REACTOR: JuniorRack
    REACTOR_VIALS: list[JuniorVial]
    RACK_PDP_TIPS: JuniorRack
    PDP_TIPS: list[JuniorPdpTip]
    SULFONYL_SVV: JuniorVial
    AMINE_SVS: list[JuniorVial]


def setup_benchtop_for_sulfonylation(
        # DCM vials
        dcm_init_volume: float = 15,

        # solid chemical sources in sv vials
        sulfonyl_init_amount: float = 100,
        solid_amines: dict[str, float] = {"solid_amine_1": 100, "solid_amine_2": 100},

        # liquid source in HRV
        pyridine_init_volume: float = 100,

):
    n_reactors = len(solid_amines)
    n_pdp_tips = n_reactors
    n_dcm_source_vials = n_reactors + 1
    dcm_init_volumes = [dcm_init_volume, ] * n_dcm_source_vials

    # create a rack for HRVs on off-deck, fill them with DCM,
    # note Z2 arm cannot reach this deck (so no VPG and racks cannot move)
    rack_solvent, dcm_vials = JuniorRack.create_rack_with_empty_vials(
        n_vials=n_dcm_source_vials, rack_capacity=6, vial_type="HRV", rack_id="RACK_SOLVENT"
    )
    for vial, volume in zip(dcm_vials, dcm_init_volumes):
        vial.chemical_content = {"DCM": volume}
    JuniorSlot.put_rack_in_a_slot(rack_solvent, JUNIOR_BENCHTOP.SLOT_OFF_1)

    # create a rack for MRVs (reactors) on 2-3-1
    rack_reactor, reactor_vials = JuniorRack.create_rack_with_empty_vials(
        n_vials=n_reactors, rack_capacity=6, vial_type="MRV", rack_id="RACK_REACTOR"
    )
    JuniorSlot.put_rack_in_a_slot(rack_reactor, JUNIOR_BENCHTOP.SLOT_2_3_1)

    # create a rack for HRVs on 2-3-2, one HRV for RSO2Cl stock solution, another for pyridine
    rack_reactant, (sulfonyl_vial, pyridine_vial) = JuniorRack.create_rack_with_empty_vials(
        n_vials=2, rack_capacity=6, vial_type="HRV", rack_id="RACK_REACTANT"
    )
    JuniorSlot.put_rack_in_a_slot(rack_reactant, JUNIOR_BENCHTOP.SLOT_2_3_2)
    pyridine_vial.chemical_content = {"pyridine": pyridine_init_volume}

    # create a rack for PDP tips on 2-3-3
    rack_pdp_tips, pdp_tips = JuniorRack.create_rack_with_empty_tips(
        n_tips=n_pdp_tips, rack_capacity=8, rack_id="RACK_PDP_TIPS", tip_id_inherit=True
    )
    JuniorSlot.put_rack_in_a_slot(rack_pdp_tips, JUNIOR_BENCHTOP.SLOT_2_3_3)

    # SV VIALS for sulfonyl
    sulfonyl_svv = JuniorVial(
        identifier="SULFONYL_SVV", contained_by=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[0].identifier,
        chemical_content={'sulfonyl chloride': sulfonyl_init_amount},
        vial_type='SV',
    )
    JUNIOR_BENCHTOP.SV_VIAL_SLOTS[0].slot_content['SLOT'] = sulfonyl_svv.identifier

    # SV VIALS for solid amines (ex. aniline)
    amine_svs = []
    i_svv_solt = 1
    for solid_amine_name, solid_amine_amount in solid_amines.items():
        amine_svv = JuniorVial(
            identifier=f"{solid_amine_name}_SVV", contained_by=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[i_svv_solt].identifier,
            chemical_content={solid_amine_name: solid_amine_amount},
            vial_type='SV',
        )
        amine_svs.append(amine_svv)
        JUNIOR_BENCHTOP.SV_VIAL_SLOTS[i_svv_solt].slot_content['SLOT'] = amine_svv.identifier
        i_svv_solt += 1

    return SulfonylationBenchtop(
        RACK_SOLVENT=rack_solvent,
        DCM_VIALS=dcm_vials,
        RACK_REACTANT=rack_reactant,
        SULFONYL_VIAL=sulfonyl_vial,
        PYRIDINE_VIAL=pyridine_vial,
        RACK_REACTOR=rack_reactor,
        REACTOR_VIALS=reactor_vials,
        RACK_PDP_TIPS=rack_pdp_tips,
        PDP_TIPS=pdp_tips,
        SULFONYL_SVV=sulfonyl_svv,
        AMINE_SVS=amine_svs,
    )


def pick_drop_rack_to(rack: JuniorRack, src_slot: JuniorSlot, dest_slot: JuniorSlot) -> list[JuniorInstruction]:
    ins1 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
            "move_to_slot": JUNIOR_BENCHTOP.VPG_SLOT,
        },
        description=f"move to slot: {JUNIOR_BENCHTOP.VPG_SLOT.identifier}"
    )

    ins2 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
        action_parameters={
            "thing": JUNIOR_BENCHTOP.VPG,
        },
        description=f"pick up: {JUNIOR_BENCHTOP.VPG.identifier}"
    )

    ins3 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
            "move_to_slot": src_slot,
        },
        description=f"move to slot: {src_slot.identifier}"
    )
    ins4 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
        action_parameters={
            "thing": rack,
        },
        description=f"pick up: {rack.identifier}"
    )
    ins5 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
            "move_to_slot": dest_slot,
        },
        description=f"move to slot: {dest_slot.identifier}"
    )
    ins6 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
        action_parameters={
            "dest_slot": dest_slot,
        },
        description=f"put down: {dest_slot.identifier}"
    )

    ins7 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
            "move_to_slot": JUNIOR_BENCHTOP.VPG_SLOT,
        },
        description=f"move to slot: {JUNIOR_BENCHTOP.VPG_SLOT.identifier}"
    )

    ins8 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
        action_parameters={
            "dest_slot": JUNIOR_BENCHTOP.VPG_SLOT,
        },
        description=f"put down: {JUNIOR_BENCHTOP.VPG_SLOT.identifier}"
    )

    ins_list = [ins1, ins2, ins3, ins4, ins5, ins6, ins7, ins8]
    JuniorInstruction.path_graph(ins_list)
    return ins_list


def solid_dispense(
        sv_vial: JuniorVial,
        sv_vial_slot: JuniorSlot,
        dest_vials: list[JuniorVial],
        amount: float,
        include_pickup_svtool=True,
        include_dropoff_svvial=True,
        include_dropoff_svtool=True,
):
    ins3 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
            "move_to_slot": sv_vial_slot,
        },
        description=f"move to slot: {sv_vial_slot.identifier}"
    )

    ins4 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
        action_parameters={"thing": sv_vial},
        description=f"pick up: {sv_vial.identifier}",
    )

    ins5 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
            "move_to_slot": JUNIOR_BENCHTOP.BALANCE,
        },
        description=f"move to slot: {JUNIOR_BENCHTOP.BALANCE.identifier}"
    )

    if include_pickup_svtool:
        ins1 = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
                "move_to_slot": JUNIOR_BENCHTOP.SV_TOOL_SLOT,
            },
            description=f"move to slot: {JUNIOR_BENCHTOP.SV_TOOL_SLOT.identifier}"
        )

        ins2 = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
            action_parameters={"thing": JUNIOR_BENCHTOP.SV_TOOL},
            description=f"pick up: {JUNIOR_BENCHTOP.SV_TOOL.identifier}",
        )
        ins_list = [ins1, ins2, ins3, ins4, ins5]
    else:
        ins_list = [ins3, ins4, ins5]

    for dest_vial in dest_vials:
        ins6 = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_Z2, action_name="dispense_sv",
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
            device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
                "move_to_slot": sv_vial_slot,
            },
            description=f"move to slot: {sv_vial_slot.identifier}"
        )

        ins8 = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
            action_parameters={
                "dest_slot": sv_vial_slot,
            },
            description=f"put down: {sv_vial_slot.identifier}"
        )
        ins_list.append(ins7)
        ins_list.append(ins8)

    if include_dropoff_svtool:
        ins9 = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
                "move_to_slot": JUNIOR_BENCHTOP.SV_TOOL_SLOT,
            },
            description=f"move to slot: {JUNIOR_BENCHTOP.SV_TOOL_SLOT.identifier}"
        )

        ins10 = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
            action_parameters={
                "dest_slot": JUNIOR_BENCHTOP.SV_TOOL_SLOT,
            },
            description=f"put down: {JUNIOR_BENCHTOP.SV_TOOL_SLOT.identifier}"
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
        amounts: list[float],
):
    z1_needles = [JUNIOR_LAB[f"Z1 Needle {i + 1}"] for i in range(len(amounts))]
    ins1 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z1,
            "move_to_slot": src_slot,
        },
        description=f"move to slot: {src_slot.identifier}"
    )
    ins2 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z1, action_name="concurrent_aspirate",
        action_parameters={
            "source_containers": src_vials,
            "dispenser_containers": z1_needles,
            "amounts": amounts,
        },
        description=f"concurrent aspirate from: {','.join([v.identifier for v in src_vials])}"
    )
    ins3 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z1,
            "move_to_slot": dest_vials_slot,
        },
        description=f"move to slot: {dest_vials_slot.identifier}"
    )
    ins4 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z1, action_name="concurrent_dispense",
        action_parameters={
            "destination_containers": dest_vials,
            "dispenser_containers": z1_needles,
            "amounts": amounts,
        },
        description=f"concurrent dispense to: {','.join([v.identifier for v in dest_vials])}"
    )
    ins5 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z1,
            "move_to_slot": JUNIOR_BENCHTOP.WASH_BAY,
        },
        description=f"move to slot: WASH BAY"
    )
    ins6 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z1, action_name="wash",
        action_parameters={
            "wash_bay": JUNIOR_BENCHTOP.WASH_BAY,
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
        amount: float,
        # speed: float
):
    ins1 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
        action_parameters={
            "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
            "move_to_slot": JUNIOR_LAB['PDT SLOT 1'],
        },
        description=f"move to slot: PDT SLOT 1"
    )

    ins2 = JuniorInstruction(
        device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
        action_parameters={"thing": JUNIOR_BENCHTOP.PDP_1},
        description=f"pick up: {JUNIOR_BENCHTOP.PDP_1.identifier}",
    )

    ins_list = [ins1, ins2]

    for tip, dest_vial in zip(tips, dest_vials):
        i_a = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
                "move_to_slot": tips_slot,
            },
            description=f"move to slot: {tips_slot.identifier}"
        )
        i_b = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_Z2, action_name="pick_up",
            action_parameters={"thing": tip},
            description=f"pick up: {tip.identifier}",
        )
        i_c = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
                "move_to_slot": src_slot,
            },
            description=f"move to slot: {src_slot.identifier}"
        )
        i_d = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_Z2, action_name="aspirate_pdp",
            action_parameters={
                "source_container": src_vial,
                "amount": amount,
                # "aspirate_speed": speed,
            },
            description=f"aspirate_pdp from: {src_vial.identifier} amount: {amount}"
        )
        i_e = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
                "move_to_slot": dest_vials_slot,
            },
            description=f"move to slot: {dest_vials_slot.identifier}"
        )
        i_f = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_Z2, action_name="dispense_pdp",
            action_parameters={
                "destination_container": dest_vial,
                "amount": amount,
                # "dispense_speed": speed,
            },
            description=f"dispense_pdp to: {dest_vial.identifier}"
        )
        i_g = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_PLATFORM, action_name="move_to",
            action_parameters={
                "anchor_arm": JUNIOR_BENCHTOP.ARM_Z2,
                "move_to_slot": JUNIOR_LAB['DISPOSAL'],
            },
            description=f"move to slot: DISPOSAL"
        )
        i_h = JuniorInstruction(
            device=JUNIOR_BENCHTOP.ARM_Z2, action_name="put_down",
            action_parameters={
                "dest_slot": JUNIOR_LAB['DISPOSAL'],
            },
            description="put down: DISPOSAL"
        )
        ins_list += [i_a, i_b, i_c, i_d, i_e, i_f, i_g, i_h]
    JuniorInstruction.path_graph(ins_list)
    return ins_list


SULFONYL_BENCHTOP = setup_benchtop_for_sulfonylation(
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


def chain_ins_lol(lol: list[list[JuniorInstruction]]):
    for i in range(len(lol) - 1):
        former = lol[i]
        latter = lol[i + 1]
        latter[0].preceding_instructions.append(former[-1].identifier)


def ins_diverge_or_converge(ins1: JuniorInstruction, ins_lst: list[JuniorInstruction], diverge=True):
    for i in ins_lst:
        if diverge:
            i.preceding_instructions.append(ins1.identifier)
        else:
            ins1.preceding_instructions.append(i.identifier)


def get_ins_lst_sulfonyl_dispense():
    ins_list1 = pick_drop_rack_to(SULFONYL_BENCHTOP.RACK_REACTANT, JUNIOR_BENCHTOP.SLOT_2_3_2, JUNIOR_BENCHTOP.BALANCE)
    ins_list2 = solid_dispense(sv_vial=SULFONYL_BENCHTOP.SULFONYL_SVV,
                               sv_vial_slot=JUNIOR_BENCHTOP.SV_VIAL_SLOTS[0],
                               dest_vials=[SULFONYL_BENCHTOP.SULFONYL_VIAL, ],
                               amount=0.5,
                               include_pickup_svtool=True,
                               include_dropoff_svvial=True,
                               include_dropoff_svtool=True)
    ins_list3 = pick_drop_rack_to(SULFONYL_BENCHTOP.RACK_REACTANT, JUNIOR_BENCHTOP.BALANCE,
                                  JUNIOR_BENCHTOP.SLOT_2_3_2, )
    lst = ins_list1 + ins_list2 + ins_list3
    JuniorInstruction.path_graph(lst)
    return lst


def get_ins_lst_amine_dispense():
    ins_list_move_to = pick_drop_rack_to(SULFONYL_BENCHTOP.RACK_REACTOR, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                         JUNIOR_BENCHTOP.BALANCE)
    ins_list_move_back = pick_drop_rack_to(SULFONYL_BENCHTOP.RACK_REACTOR, JUNIOR_BENCHTOP.BALANCE,
                                           JUNIOR_BENCHTOP.SLOT_2_3_1, )

    dispense_lst = []
    for i in range(len(SULFONYL_BENCHTOP.AMINE_SVS)):
        amine_svv = SULFONYL_BENCHTOP.AMINE_SVS[i]
        amine_svv_slot = JUNIOR_BENCHTOP.SV_VIAL_SLOTS[i + 1]
        reactor_vial = SULFONYL_BENCHTOP.REACTOR_VIALS[i]
        ins_list_actual_dispense = solid_dispense(sv_vial=amine_svv,
                                                  sv_vial_slot=amine_svv_slot,
                                                  dest_vials=[reactor_vial, ],
                                                  amount=0.5,
                                                  include_pickup_svtool=True,
                                                  include_dropoff_svvial=True, include_dropoff_svtool=True)
        dispense_lst += ins_list_actual_dispense
    lst = ins_list_move_to + dispense_lst + ins_list_move_back
    JuniorInstruction.path_graph(lst)
    return lst


def get_ins_lst_dcm_dispense():
    ins_list7 = needle_dispense(SULFONYL_BENCHTOP.DCM_VIALS[:-1], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                SULFONYL_BENCHTOP.REACTOR_VIALS, JUNIOR_BENCHTOP.SLOT_2_3_1,
                                [1, ] * len(SULFONYL_BENCHTOP.REACTOR_VIALS))
    ins_list8 = needle_dispense(SULFONYL_BENCHTOP.DCM_VIALS[-1:], JUNIOR_BENCHTOP.SLOT_OFF_1,
                                [SULFONYL_BENCHTOP.SULFONYL_VIAL, ], JUNIOR_BENCHTOP.SLOT_2_3_2, [1, ])
    lst = ins_list7 + ins_list8
    JuniorInstruction.path_graph(lst)
    return lst


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
    JuniorInstruction.path_graph(lst)
    return lst


ins_lst_sulfonyl_dispense = get_ins_lst_sulfonyl_dispense()
ins_lst_amine_dispense = get_ins_lst_amine_dispense()
ins_lst_dcm_dispense = get_ins_lst_dcm_dispense()
ins_lst_pyridine_dispense = pdp_dispense(SULFONYL_BENCHTOP.PYRIDINE_VIAL, JUNIOR_BENCHTOP.SLOT_2_3_2,
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

diagram = JUNIOR_LAB.instruction_graph

diagram.layout(algo="rt_circular")
diagram.dump_file(filename="sim_junior_multi_single_step.drawio", folder="./")

from casymda_hardware.model import *
import simpy

env = simpy.Environment()

model = Model(env, JUNIOR_LAB, wdir=os.path.abspath("./"), model_name=f"junior_multi_single_step")

env.run()
