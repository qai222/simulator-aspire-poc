from schema import ActionCreateArtifact, ActionTransitArtifact

import hardware
from hardware import LAB
from schema import *

HARDWARE_UNITS = []
for name, values in vars(hardware).items():
    if isinstance(values, Artifact):
        HARDWARE_UNITS.append(values)

INIT_ACTIONS = dict()

for art in HARDWARE_UNITS:
    act = ActionCreateArtifact(
        description=f"create: {art.type} -- {art.identifier}",
        duration=0.0,
        creation=art,
        lab=LAB
    )
    INIT_ACTIONS[act.identifier] = act

for act in INIT_ACTIONS.values():
    act.execute()

# print(LAB)
for i, art in LAB.artifacts.items():
    if art.type == "MATTER":
        print(art)



# # liquid transfer action
# def create_liquid_transfer_action(source:Artifact, destination: Artifact, matter: Artifact, amount:float):
#     # source must have enough matter to be transferred
#     art_preactor_0 = ArtifactCondition(
#         artifact=matter, target_qi=hardware.qi_chemical_boundary, condition="EQ", condition_parameter=source.identifier
#     )
#     art_preactor_1 = ArtifactCondition(
#         artifact=matter, target_qi=hardware.qi_chemical_amount, condition="GT", condition_parameter=amount
#     )  # TODO unit check?
#
#     # destination must have enough space
#     art_preactor_2 = ArtifactCondition(
#         artifact=destination, target_qi=hardware.qi_volume_capacity, condition="EQ", condition_parameter=source.identifier
#     )
#     art_preactor_3 = ArtifactCondition(
#         artifact=matter, target_qi=hardware.qi_chemical_amount, condition="GT", condition_parameter=amount
#     )  # TODO unit check?
#
#
#
#     matter
#
#     preactor_art = ArtifactCondition(artifact=source, target_qi=hardware.qi)
#     act = ActionTransitArtifact()
#
#
#     source_art = LAB.artifacts[source]
#     destination_art = LAB.artifacts[destination]
#
#
#
#
#
#
#
#
#
