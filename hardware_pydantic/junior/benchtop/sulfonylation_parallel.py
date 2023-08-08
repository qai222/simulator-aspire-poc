from hardware_pydantic.junior import *

"""
following the notes of N-Sulfonylation 
"""


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
        junior_benchtop: JuniorBenchtop,
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
    JuniorSlot.put_rack_in_a_slot(rack_solvent, junior_benchtop.SLOT_OFF_1)

    # create a rack for MRVs (reactors) on 2-3-1
    rack_reactor, reactor_vials = JuniorRack.create_rack_with_empty_vials(
        n_vials=n_reactors, rack_capacity=6, vial_type="MRV", rack_id="RACK_REACTOR"
    )
    JuniorSlot.put_rack_in_a_slot(rack_reactor, junior_benchtop.SLOT_2_3_1)

    # create a rack for HRVs on 2-3-2, one HRV for RSO2Cl stock solution, another for pyridine
    rack_reactant, (sulfonyl_vial, pyridine_vial) = JuniorRack.create_rack_with_empty_vials(
        n_vials=2, rack_capacity=6, vial_type="HRV", rack_id="RACK_REACTANT"
    )
    JuniorSlot.put_rack_in_a_slot(rack_reactant, junior_benchtop.SLOT_2_3_2)
    pyridine_vial.chemical_content = {"pyridine": pyridine_init_volume}

    # create a rack for PDP tips on 2-3-3
    rack_pdp_tips, pdp_tips = JuniorRack.create_rack_with_empty_tips(
        n_tips=n_pdp_tips, rack_capacity=8, rack_id="RACK_PDP_TIPS", tip_id_inherit=True
    )
    JuniorSlot.put_rack_in_a_slot(rack_pdp_tips, junior_benchtop.SLOT_2_3_3)

    # SV VIALS for sulfonyl
    sulfonyl_svv = JuniorVial(
        identifier="SULFONYL_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[0].identifier,
        chemical_content={'sulfonyl chloride': sulfonyl_init_amount},
        vial_type='SV',
    )
    junior_benchtop.SV_VIAL_SLOTS[0].slot_content['SLOT'] = sulfonyl_svv.identifier

    # SV VIALS for solid amines (ex. aniline)
    amine_svs = []
    i_svv_solt = 1
    for solid_amine_name, solid_amine_amount in solid_amines.items():
        amine_svv = JuniorVial(
            identifier=f"{solid_amine_name}_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[i_svv_solt].identifier,
            chemical_content={solid_amine_name: solid_amine_amount},
            vial_type='SV',
        )
        amine_svs.append(amine_svv)
        junior_benchtop.SV_VIAL_SLOTS[i_svv_solt].slot_content['SLOT'] = amine_svv.identifier
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
