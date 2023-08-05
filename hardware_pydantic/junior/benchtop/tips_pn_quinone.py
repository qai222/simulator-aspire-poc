from hardware_pydantic.junior.junior_lab import *

"""
aldol condensation to make pentacenequinone,
originally from: 10.1002/ange.19530652309,
procedure used here from: 10.3390/molecules17044625

## procedure text:

Aqueous NaOH (10%, 5.96 g, 149 mmol) was slowly added to
a solution of o-phthalaldehyde (10 g, 74.6 mmol) and 1,4-cyclohexanedione (4.18 g, 37.3 mmol) in ethanol (460 mL)
under a N2 atmosphere.
The solution turned from yellow to golden brown to dark brown before a yellow solid corresponding to
pentacene-6,13-dione precipitated. After stirring the reaction mixture for four hours,
the crude reaction mixture was filtered and washed with ethanol, water, and methanol until the washings were colorless.
The solid residue was dried under vacuum to obtain 11.02 g (96% yield) of bright yellow pentacene-6,13-dione.


## chemicals

fresh solution:
- Aqueous NaOH (10%, 5.96 g, 149 mmol)
solvent:
- ethanol
- water
solid needed:
- o-phthalaldehyde (10 g, 74.6 mmol)
- 1,4-cyclohexanedione (4.18 g, 37.3 mmol)


## automated version:

a. solid dispense naoh -> hrv
b. solid dispense diketone -> hrv
c. solid dispense aldehyde -> mrvs
d. z1 dispense water -> naoh
e. z1 dispense ethanol -> diketone
f. z1 dispense ethanol -> aldehyde
d. pdp add diketone -> mrvs
e. stir
f. pdp add naoh -> mrvs
g. react rt 4h
"""


class QuinoneBenchtop(BaseModel):
    RACK_LIQUID: JuniorRack
    ETHANOL_VIALS: list[JuniorVial]
    WATER_VIALS: list[JuniorVial]

    RACK_REACTANT: JuniorRack
    # ALDEHYDE_VIAL: JuniorVial  # this will be loaded to reactor directly
    DIKETONE_VIAL: JuniorVial
    NAOH_VIAL: JuniorVial

    REACTOR_VIALS: list[JuniorVial]

    RACK_REACTOR: JuniorRack
    REACTOR_VIALS: list[JuniorVial]

    RACK_PDP_TIPS: JuniorRack
    PDP_TIPS: list[JuniorPdpTip]

    DIKETONE_SVV: JuniorVial
    ALDEHYDE_SVV: JuniorVial
    NAOH_SVV: JuniorVial


def setup_quinone_benchtop(
        junior_benchtop: JuniorBenchtop,
        n_reactors: int = 4,
        # liquid sources
        water_init_volume: float = 15,
        ethanol_init_volume: float = 15,

        # solid chemical sources in sv vials
        diketone_init_amount: float = 100,
        naoh_init_amount: float = 100,
        aldehyde_init_amount: float = 100,
):
    n_pdp_tips = n_reactors * 2  # one for diketone another for naoh
    n_ethanol_source_vials = n_reactors + 1  # one additional for diketone stock solution
    ethanol_init_volumes = [ethanol_init_volume, ] * n_ethanol_source_vials

    # create a rack for HRVs on off-deck, fill them with ethanol,
    rack_liquid, liquid_vials = JuniorRack.create_rack_with_empty_vials(
        n_vials=n_ethanol_source_vials + 1, rack_capacity=12, vial_type="HRV", rack_id="RACK_LIQUID"
    )
    ethanol_vials = liquid_vials[:-1]
    water_vial = liquid_vials[-1]
    for vial, volume in zip(ethanol_vials, ethanol_init_volumes):
        vial.chemical_content = {"Ethanol": volume}
    water_vial.chemical_content = {"Water": water_init_volume}

    JuniorSlot.put_rack_in_a_slot(rack_liquid, junior_benchtop.SLOT_OFF_1)

    # create a rack for MRVs (reactors) on 2-3-1
    rack_reactor, reactor_vials = JuniorRack.create_rack_with_empty_vials(
        n_vials=n_reactors, rack_capacity=12, vial_type="MRV", rack_id="RACK_REACTOR"
    )
    JuniorSlot.put_rack_in_a_slot(rack_reactor, junior_benchtop.SLOT_2_3_1)

    # create a rack for HRVs on 2-3-2, one for diketone one for naoh aq.
    rack_reactant, (diketone_vial, naoh_vial) = JuniorRack.create_rack_with_empty_vials(
        n_vials=2, rack_capacity=12, vial_type="HRV", rack_id="RACK_REACTANT"
    )
    JuniorSlot.put_rack_in_a_slot(rack_reactant, junior_benchtop.SLOT_2_3_2)

    # create a rack for PDP tips on 2-3-3
    rack_pdp_tips, pdp_tips = JuniorRack.create_rack_with_empty_tips(
        n_tips=n_pdp_tips, rack_capacity=12, rack_id="RACK_PDP_TIPS", tip_id_inherit=True
    )
    JuniorSlot.put_rack_in_a_slot(rack_pdp_tips, junior_benchtop.SLOT_2_3_3)

    # SV VIALS for diketone, aldehyde, naoh
    diketone_svv = JuniorVial(
        identifier="DIKETONE_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[0].identifier,
        chemical_content={'Diketone': diketone_init_amount},
        vial_type='SV',
    )
    aldehyde_svv = JuniorVial(
        identifier="ALDEHYDE_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[1].identifier,
        chemical_content={'Aledehyde': aldehyde_init_amount},
        vial_type='SV',
    )
    naoh_svv = JuniorVial(
        identifier="NAOH_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[2].identifier,
        chemical_content={'NaOH': naoh_init_amount},
        vial_type='SV',
    )
    junior_benchtop.SV_VIAL_SLOTS[0].slot_content['SLOT'] = diketone_svv.identifier
    junior_benchtop.SV_VIAL_SLOTS[1].slot_content['SLOT'] = aldehyde_svv.identifier
    junior_benchtop.SV_VIAL_SLOTS[2].slot_content['SLOT'] = naoh_svv.identifier

    return QuinoneBenchtop(
        RACK_LIQUID=rack_liquid,
        ETHANOL_VIALS=ethanol_vials,
        WATER_VIALS=[water_vial, ],
        RACK_REACTANT=rack_reactant,
        DIKETONE_VIAL=diketone_vial,
        NAOH_VIAL=naoh_vial,
        RACK_REACTOR=rack_reactor,
        REACTOR_VIALS=reactor_vials,
        RACK_PDP_TIPS=rack_pdp_tips,
        PDP_TIPS=pdp_tips,
        DIKETONE_SVV=diketone_svv,
        ALDEHYDE_SVV=aldehyde_svv,
        NAOH_SVV=naoh_svv,
    )
