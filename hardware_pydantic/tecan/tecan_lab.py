from hardware_pydantic.tecan.tecan_devices import *
from hardware_pydantic.tecan.tecan_objects import TecanHotel

"""Initialization of the TECAN lab."""


def create_tecan_base():
    """ create a base for the TECAN lab.

    Returns
    -------
    The TECAN lab.

    """
    assert len(TECAN_LAB.dict_object) == 0, "TECAN BASE HAS ALREADY BEEN INIT!!!"

    liquid_tank_1 = TecanLiquidTank(
        identifier="TANK-1",
        layout=TecanLayout.from_relative_layout(layout_x=TECAN_LAYOUT_SLOT_SIZE_X_SMALL)
    )

    liquid_tank_2 = TecanLiquidTank(
        identifier="TANK-2",
        layout=TecanLayout.from_relative_layout("above",
                                                liquid_tank_1.layout,
                                                TECAN_LAYOUT_SLOT_SIZE_X_SMALL)
    )

    liquid_tank_3 = TecanLiquidTank(
        identifier="TANK-3",
        layout=TecanLayout.from_relative_layout("above",
                                                liquid_tank_2.layout,
                                                TECAN_LAYOUT_SLOT_SIZE_X_SMALL)
    )

    wash_bay = TecanWashBay(
        identifier="WASH-BAY",
        layout=TecanLayout.from_relative_layout("right_to",
                                                liquid_tank_1.layout,
                                                TECAN_LAYOUT_SLOT_SIZE_X_SMALL,
                                                TECAN_LAYOUT_SLOT_SIZE_Y * 3)
    )

    dispensing_slots = []

    dispensing_slots_per_row = 4

    for i in range(12):
        irow = i // dispensing_slots_per_row
        icol = i % dispensing_slots_per_row
        if i == 0:
            d_slot = TecanSlot(
                identifier=f"D-SLOT-{i + 1}",
                layout=TecanLayout.from_relative_layout('right_to',
                                                        wash_bay.layout,
                                                        TECAN_LAYOUT_SLOT_SIZE_X_SMALL * 2,
                                                        TECAN_LAYOUT_SLOT_SIZE_Y_SMALL),
            )
        else:
            last_slot = dispensing_slots[i - 1]
            if irow == (i - 1) // dispensing_slots_per_row:
                relation = "right_to"
                relative = last_slot
            else:
                relation = "above"
                relative = dispensing_slots[i - dispensing_slots_per_row]
            d_slot = TecanSlot(
                identifier=f"D-SLOT-{i + 1}",
                layout=TecanLayout.from_relative_layout(relation,
                                                        relative.layout,
                                                        TECAN_LAYOUT_SLOT_SIZE_X_SMALL * 2,
                                                        TECAN_LAYOUT_SLOT_SIZE_Y_SMALL),
            )
        dispensing_slots.append(d_slot)

    heater_slot_1 = TecanSlot(
        identifier="HEATER-1",
        layout=TecanLayout.from_relative_layout("right_to", dispensing_slots[3].layout)
    )

    heater_slot_2 = TecanSlot(
        identifier="HEATER-2",
        layout=TecanLayout.from_relative_layout("above", heater_slot_1.layout)
    )

    hotel = TecanHotel.from_capacity(
        can_contain=[TecanPlate.__name__, ],
        capacity=16,
        container_id="HOTEL",
        layout=TecanLayout.from_relative_layout("right_to", heater_slot_2.layout)
    )

    arm_1 = TecanArm1(
        identifier='ARM-1',
        slot_content={
            str(i + 1): TecanArm1Needle(identifier=f"ARM-1 Needle {i + 1}",
                                        contained_by='ARM-1',
                                        contained_in_slot=str(i + 1),
                                        material="STEEL").identifier
            for i in range(8)
        },
    )

    arm_z2 = TecanArm2(identifier='ARM-2', )

    return TECAN_LAB
