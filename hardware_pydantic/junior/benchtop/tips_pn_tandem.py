from hardware_pydantic.junior.junior_lab import *
from hardware_pydantic.junior.benchtop.tips_pn_quinone import QuinoneBenchtop, setup_quinone_benchtop
from hardware_pydantic.junior.benchtop.tips_pn_grignard import GrignardBenchtop
"""
two-step to make tips-pn
"""

class JuniorMage(Device, JuniorLabObject):

    def action__set_vial_chemical_content(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            vial: JuniorVial,
            chemical: dict[str, float],
            time_cost: float,
    ):
        if actor_type == 'pre':
            if vial.identifier not in JUNIOR_LAB.dict_object:
                raise PreActError
        elif actor_type == 'post':
            vial.chemical_content = chemical
        elif actor_type == 'proj':
            return vial, time_cost
        else:
            raise ValueError


    def action__create_vials(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            vials: list[JuniorVial],
            in_rack: JuniorRack,
            in_rack_slot_keys: list[str],
            time_cost: float,
    ):
        if actor_type == 'pre':
            if any(v.contained_by is not None for v in vials):
                raise PreActError
            if any(v.identifier in JUNIOR_LAB.dict_object for v in vials):
                raise PreActError
            if any(in_rack.slot_content[k] is not None for k in in_rack_slot_keys):
                raise PreActError
        elif actor_type == 'post':
            for v, vk in zip(vials, in_rack_slot_keys):
                v.contained_by = in_rack.identifier
                v.contained_in_slot = vk
                in_rack.slot_content[vk] = v.identifier
        elif actor_type == 'proj':
            return vials + [in_rack, ], time_cost
        else:
            raise ValueError


    def action__annihilate_vials(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE,
            vials: list[JuniorVial],
            time_cost: float,
    ):
        if actor_type == 'pre':
            if any(v.contained_by is None for v in vials):
                raise PreActError
            if any(v.identifier not in JUNIOR_LAB.dict_object for v in vials):
                raise PreActError
            if any(not isinstance(JUNIOR_LAB[v.contained_by], JuniorRack) for v in vials):  # this does not work for SV vials
                raise PreActError
        elif actor_type == 'post':
            for v in vials:
                rack = JUNIOR_LAB[v.contained_by]
                rack.slot_content[v.contained_in_slot] = None
                rack: JuniorRack
                v.contained_in_slot = None
                v.contained_by = None
                JUNIOR_LAB.remove_object(v)
        elif actor_type == 'proj':
            return vials + [JUNIOR_LAB[v.contained_by] for v in vials], time_cost


class TipsPnBenchtop(BaseModel):
    
    mage: JuniorMage

    quinone_transferred: float

    grignard_benchtop: GrignardBenchtop

    quinone_benchtop: QuinoneBenchtop
    
    def check(self):
        assert LabContainee.get_container(self.grignard_benchtop.RACK_LIQUID, JUNIOR_LAB, JuniorSlot).identifier != LabContainee.get_container(self.quinone_benchtop.RACK_LIQUID, JUNIOR_LAB, JuniorSlot).identifier
        

def setup_tips_pn_benchtop(
        junior_benchtop: JuniorBenchtop,

        quinone_n_reactors: int = 4,
        quinone_water_init_volume: float = 15,
        quinone_ethanol_init_volume: float = 15,
        quinone_diketone_init_amount: float = 100,
        quinone_naoh_init_amount: float = 100,
        quinone_aldehyde_init_amount: float = 100,

        n_reactors: int = 4,

        # liquid sources
        thf_init_volume: float = 15,
        hcl_init_volume: float = 15,
        silyl_init_volume: float = 15,

        # solid chemical sources in sv vials
        grignard_init_amount: float = 100,
        quinone_transferred: float = 100,  # from magic
):
    quinone_benchtop = setup_quinone_benchtop(
        junior_benchtop,
        quinone_n_reactors,
    quinone_water_init_volume,
    quinone_ethanol_init_volume,
    quinone_diketone_init_amount,
    quinone_naoh_init_amount,
    quinone_aldehyde_init_amount,
    )

    n_pdp_tips = n_reactors * 2  # one for silyl another for quinone
    n_thf_source_vials = n_reactors + 1  # one additional for quinone solution
    n_hcl_source_vials = n_reactors

    # create a rack for HRVs on off-deck, fill them with thf,
    rack_liquid, liquid_vials = JuniorRack.create_rack_with_empty_vials(
        n_vials=n_thf_source_vials + n_hcl_source_vials, rack_capacity=12, vial_type="HRV", rack_id="RACK_LIQUID_GRIGNARD"
    )
    thf_vials = liquid_vials[:n_thf_source_vials]
    hcl_vials = liquid_vials[- n_hcl_source_vials:]
    thf_init_volumes = [thf_init_volume, ] * n_thf_source_vials
    hcl_init_volumes = [hcl_init_volume, ] * n_hcl_source_vials
    for vial, volume in zip(thf_vials, thf_init_volumes):
        vial.chemical_content = {"THF": volume}
    for vial, volume in zip(hcl_vials, hcl_init_volumes):
        vial.chemical_content = {"HCl/SnCl2": volume}
    JuniorSlot.put_rack_in_a_slot(rack_liquid, junior_benchtop.SLOT_OFF_2)

    # add MRVs (reactors) to 2-3-1
    rack_reactor = quinone_benchtop.RACK_REACTOR
    added_reactor_vials = []
    for k in rack_reactor.empty_slot_keys:
        vial =  JuniorVial(
            identifier=f"vial-{k} " + quinone_benchtop.RACK_REACTOR.identifier, contained_by=quinone_benchtop.RACK_REACTOR.identifier,
            contained_in_slot=k, vial_type="MRV",
        )
        quinone_benchtop.RACK_REACTOR.slot_content[k] = vial.identifier
        added_reactor_vials.append(vial)
        if len(added_reactor_vials) == n_reactors:
            break
    reactor_vials = added_reactor_vials
    
    # add HRVs on 2-3-2, one for silyl one for quinone sln
    rack_reactant = quinone_benchtop.RACK_REACTANT
    silyl_vial_key, quinone_vial_key = rack_reactant.empty_slot_keys[:2]
    silyl_vial = JuniorVial(
        identifier=f"silyl vial " + rack_reactant.identifier,
        contained_by=rack_reactant.identifier,
        contained_in_slot=silyl_vial_key, vial_type="HRV",
    )
    rack_reactant.slot_content[silyl_vial_key] = silyl_vial.identifier
    silyl_vial.chemical_content = {'silyl': silyl_init_volume}

    quinone_vial = JuniorVial(
        identifier=f"quinone vial " + rack_reactant.identifier,
        contained_by=rack_reactant.identifier,
        contained_in_slot=quinone_vial_key, vial_type="HRV",
    )
    rack_reactant.slot_content[quinone_vial_key] = quinone_vial.identifier


    # add PDP tips on 2-3-3
    rack_pdp_tips = quinone_benchtop.RACK_PDP_TIPS
    pdp_tips = []
    n_created = 0
    for k in rack_pdp_tips.empty_slot_keys:
        v = JuniorPdpTip(
            identifier=f"PdpTip-{k} " + rack_pdp_tips.identifier, contained_by=rack_pdp_tips.identifier, contained_in_slot=k,
        )
        rack_pdp_tips.slot_content[k] = v.identifier
        pdp_tips.append(v)
        n_created += 1
        if n_created == n_pdp_tips:
            break

    # SV VIALS for quinone [10], grignard [11]
    quinone_svv = JuniorVial(
        identifier="QUINONE_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[10].identifier,
        # chemical_content={'Quinone': quinone_init_amount},  # added by mage
        vial_type='SV',
    )
    grignard_svv = JuniorVial(
        identifier="GRIGNARD_SVV", contained_by=junior_benchtop.SV_VIAL_SLOTS[11].identifier,
        chemical_content={'Grignard': grignard_init_amount},
        vial_type='SV',
    )
    junior_benchtop.SV_VIAL_SLOTS[10].slot_content['SLOT'] = quinone_svv.identifier
    junior_benchtop.SV_VIAL_SLOTS[11].slot_content['SLOT'] = grignard_svv.identifier
    
    mage = JuniorMage()
    
    grignard_benchtop = GrignardBenchtop(
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

    return TipsPnBenchtop(grignard_benchtop=grignard_benchtop, quinone_benchtop=quinone_benchtop, mage=mage, quinone_transferred=quinone_transferred)



