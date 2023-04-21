from copy import deepcopy

from deepdiff import DeepDiff

from schema import *


def create_vial(identifier: str, capacity: float = 50, unit: str = 'mL'):
    vial = Artifact(
        identifier=identifier,
        type="vial",
        state={
            Quality(name='capacity', value=capacity, related_to=None, unit=unit, ),
        },
    )
    return vial


def create_heater(identifier: str):
    heater = Artifact(
        identifier=identifier,
        type="heater",
        state={
            Quality(name='switch', value='off'),
            Quality(name='temperature_set_to', value=25, unit='C'),
        },
    )
    return heater


def create_rack(identifier: str):
    rack = Artifact(
        identifier=identifier,
        type="rack",
        state={
            Quality(name='capacity', value=16),
        },
    )
    return rack


def state_delta(state1: set[Quality], state2: set[Quality]):
    return DeepDiff(state1, state2, view='tree')


vial_0 = create_vial('vial_0')
vial_1 = create_vial('vial_1')
heater_0 = create_heater('heater_0')
rack_0 = create_rack('rack_0')

system = System(artifacts={vial_0, vial_1, heater_0, rack_0})
state0 = deepcopy(system.state)
print(state0)

# fill vial_0
vial_0.modify_quality(
    name='chemical DMSO', new_value=10, related_to=('flask A', 'filled_from'), unit='mL'
)
state1 = deepcopy(system.state)
print("FILL VIAL_0")
print(state_delta(state0, state1).pretty())

# fill vial_1 from vial_0
vial_0.modify_quality(
    name='chemical DMSO', new_value=0, related_to=('vial_1', 'filled_to'), unit='mL'
)
vial_1.modify_quality(
    name='chemical DMSO', new_value=10, related_to=('vial_0', 'filled_from'), unit='mL'
)
state2 = deepcopy(system.state)
print("FILL VIAL_1 from VIAL_0")
print(state_delta(state1, state2).pretty())

# place vial_1 in rack_0
vial_1.modify_quality(
    name='position', new_value='8A', related_to=('rack_0', 'inside')
)
state3 = deepcopy(system.state)
print("MOVE VIAL_1 TO RACK_0")
print(state_delta(state2, state3).pretty())