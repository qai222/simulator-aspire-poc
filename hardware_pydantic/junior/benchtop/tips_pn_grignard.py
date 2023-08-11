from hardware_pydantic.junior.junior_lab import *

"""
Grignard reaction from silyl acetylene and quinone to make TIPS-Pentacene, 
adapted from J.A.'s original procedure from 10.1021/ja0162459

## procedure text: 

To a flame-dried 60 ml Teflon screw-stoppered glass tube was added
4.9 ml of a 2.0 M solution of isopropyl magnesium chloride in tetrahydrofuran.
Triisopropylsilyl acetylene (1.8g, 9.87 mmol) was added via syringe, followed by an
additional 10 ml of dry tetrahydrofuran. The cap was replaced and the tube placed in a 60 °C
oil bath for 15 minutes. The tube was removed from heat and the solution was allowed to
cool. 0.5g (1.62 mmol, .16 equivalents based on Grignard reagent) of the appropriate
quinone was added to the solution, the cap was replaced and the tube placed back into the oil
bath for 30 minutes (or until there was no solid quinone remaining in the tube). The tube was
removed from heat and allowed to cool. A solution of 10% aqueous HCl saturated with SnCl2
was added carefully until the tube contents no longer bubbled on addition of the tin chloride
solution. The reaction solution turned a deep blue color. The cap was replaced and the tube
returned to the oil bath for 15 minutes. The tube was removed and allowed to cool.


## chemicals

fresh solution:
- iPrMgCl (THF) 4.9 ml, 2.0 M
solvent:
- THF 10 ml
solution:
- HCl/SnCl2
solid
- silyl acetylene (1.8g, 9.87 mmol)
- quinone 0.5g (1.62 mmol)


## automated version:

a. solid dispense grignard -> mrvs
b. z1 dispense thf -> mrvs
c. pdp add silyl -> mrvs
d. 60 C 15 min
e. solid dispense quinone -> hrv
f. z1 dispense thf -> quinone hrv
g. pdp add quinone sln -> mrvs
h. 60 C 30 min
i. z1 dispense hcl -> mrvs

implemented as a path graph rn, ef can be made independent from abcd
"""


class GrignardBenchtop(BaseModel):
    RACK_LIQUID: JuniorRack
    THF_VIALS: list[JuniorVial]
    HCL_VIALS: list[JuniorVial]

    RACK_REACTANT: JuniorRack
    QUINONE_VIAL: JuniorVial
    SILYL_VIAL: JuniorVial

    RACK_REACTOR: JuniorRack
    REACTOR_VIALS: list[JuniorVial]

    RACK_PDP_TIPS: JuniorRack
    PDP_TIPS: list[JuniorPdpTip]

    QUINONE_SVV: JuniorVial
    GRIGNARD_SVV: JuniorVial


def setup_grignard_benchtop(
        junior_benchtop: JuniorBenchtop,

        n_reactors: int = 4,

        # liquid sources
        thf_init_volume: float = 15,
        hcl_init_volume: float = 15,
        silyl_init_volume: float = 15,

        # solid chemical sources in sv vials
        grignard_init_amount: float = 100,
        quinone_init_amount: float = 100,
):
    n_pdp_tips = n_reactors * 2  # one for silyl another for quinone
    n_thf_source_vials = n_reactors + 1  # one additional for quinone solution
    n_hcl_source_vials = n_reactors

    # create a rack for HRVs on off-deck, fill them with thf,
    rack_liquid, liquid_vials = JuniorRack.create_rack_with_empty_vials(
        n_vials=n_thf_source_vials + n_hcl_source_vials, rack_capacity=12, vial_type="HRV", rack_id="RACK_LIQUID"
    )
    thf_vials = liquid_vials[:n_thf_source_vials]
    hcl_vials = liquid_vials[- n_hcl_source_vials:]
    thf_init_volumes = [thf_init_volume, ] * n_thf_source_vials
    hcl_init_volumes = [hcl_init_volume, ] * n_hcl_source_vials
    for vial, volume in zip(thf_vials, thf_init_volumes):
        vial.chemical_content = {"THF": volume}

    for vial, volume in zip(hcl_vials, hcl_init_volumes):
        vial.chemical_content = {"HCl/SnCl2": volume}
    JuniorSlot.put_rack_in_a_slot(rack_liquid, junior_benchtop.SLOT_OFF_1)

    # create a rack for MRVs (reactors) on 2-3-1
    rack_reactor, reactor_vials = JuniorRack.create_rack_with_empty_vials(
        n_vials=n_reactors, rack_capacity=6, vial_type="MRV", rack_id="RACK_REACTOR"
    )
    JuniorSlot.put_rack_in_a_slot(rack_reactor, junior_benchtop.SLOT_2_3_1)

    # create a rack for HRVs on 2-3-2, one for silyl one for quinone sln
    rack_reactant, (silyl_vial, quinone_vial) = JuniorRack.create_rack_with_empty_vials(
        n_vials=2, rack_capacity=6, vial_type="HRV", rack_id="RACK_REACTANT"
    )
    silyl_vial.chemical_content = {'silyl': silyl_init_volume}
    JuniorSlot.put_rack_in_a_slot(rack_reactant, junior_benchtop.SLOT_2_3_2)

    # create a rack for PDP tips on 2-3-3
    rack_pdp_tips, pdp_tips = JuniorRack.create_rack_with_empty_tips(
        n_tips=n_pdp_tips, rack_capacity=8, rack_id="RACK_PDP_TIPS", tip_id_inherit=True
    )
    JuniorSlot.put_rack_in_a_slot(rack_pdp_tips, junior_benchtop.SLOT_2_3_3)

    # SV VIALS for quinone, grignard
    quinone_svv = JuniorVial(
        identifier="QUINONE_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[0].identifier,
        chemical_content={'Quinone': quinone_init_amount},
        vial_type='SV',
    )
    grignard_svv = JuniorVial(
        identifier="GRIGNARD_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[1].identifier,
        chemical_content={'Grignard': grignard_init_amount},
        vial_type='SV',
    )

    junior_benchtop.SV_VIAL_SLOTS[0].slot_content['SLOT'] = quinone_svv.identifier
    junior_benchtop.SV_VIAL_SLOTS[1].slot_content['SLOT'] = grignard_svv.identifier

    return GrignardBenchtop(
        RACK_LIQUID=rack_liquid,
        THF_VIALS=thf_vials,
        HCL_VIALS=hcl_vials,
        RACK_REACTANT=rack_reactant,
        QUINONE_VIAL=quinone_vial,
        RACK_REACTOR=rack_reactor,
        REACTOR_VIALS=reactor_vials,
        RACK_PDP_TIPS=rack_pdp_tips,
        PDP_TIPS=pdp_tips,
        SILYL_VIAL=silyl_vial,
        QUINONE_SVV=quinone_svv,
        GRIGNARD_SVV=grignard_svv,
    )
