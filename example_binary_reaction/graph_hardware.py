"""
suspend this so we can work on oo code
"""

# from __future__ import annotations
#
# from graph_schema import *
# from graph_schema.instances import *
# """
# what do we have in the lab?
# - two storage containers from which chemical A and B were transferred
# """
#
# quality_can_hold = Quality(name="can hold")
#
#
# class ArtifactFactory:
#
#
#     @staticmethod
#     def build_container(capacity_volume: float, can_hold_what: list[Individual] | None = None):
#         art = Artifact()
#         HasQuality(subject_individual=art, object_individual=quality_can_hold)
#         if can_hold is not None:
#
#
#
#
#
# storage_a = Artifact()
# storage_b = Artifact()
# chemical_a = Artifact()
# chemical_b = Artifact()
# rack_0 = Artifact()
# transferor_0 = Artifact()
#
#
#
# storage_a = Artifact()
# capacity_volume = Quality(name="capacity_volume", value=5000)
# HasQuality(subject_individual=storage_a, object_individual=artifact_type_container)
# HasQuality(subject_individual=storage_a, object_individual=capacity_volume)
#
# storage_b = Artifact()
# capacity_volume = Quality(name="capacity_volume", value=4000)
# HasQuality(subject_individual=storage_b, object_individual=artifact_type_container)
# HasQuality(subject_individual=storage_b, object_individual=capacity_volume)
#
# # a rack holding empty vials
# rack_0 = Artifact()
# HasQuality(subject_individual=rack_0, object_individual=artifact_type_container)
# # 50 slots
# capacity_slot = Quality(name="capacity_slot", value=50)
# HasQuality(subject_individual=rack_0, object_individual=capacity_slot)
#
# q = Quality(name="artifact type", value="")
#
# # two storage containers from which chemical A and B were transferred
# storage_a = Artifact()
# storage_b = Artifact()
# # set their capacities
# qi_volume_capacity = QualityIdentifier(name="volume_capacity")
# storage_a[qi_volume_capacity] = Quality(identifier=qi_volume_capacity, value=500, unit="mL")
# storage_b[qi_volume_capacity] = Quality(identifier=qi_volume_capacity, value=800, unit="mL")
#
# # a rack holding empty vials
# rack_0 = Artifact(type="RACK")
# # 50 slots
# qi_slot_capacity = QualityIdentifier(name="slot_capacity")
# rack_0[qi_slot_capacity] = Quality(identifier=qi_slot_capacity, value=50)
#
# # 50 vials (can hold 30 mL) initially on rack_0
# vials = []
# qi_vial_position = QualityIdentifier(name="position", relative=rack_0.identifier, relation="on_rack")
# for i in range(50):
#     vial = Artifact(type="VIAL")
#     vial[qi_volume_capacity] = Quality(identifier=qi_volume_capacity, value=30, unit="mL")
#     vial[qi_vial_position] = Quality(identifier=qi_vial_position, value=i)
#     vials.append(vial)
#
# # one robot arm that can access both storage containers and rack_0
# transferor_0 = Artifact(type="TRANSFEROR")
# qi_robot_access = QualityIdentifier(name="robot_access")
# transferor_0[qi_robot_access] = Quality(identifier=qi_robot_access,
#                                         value=[rack_0.identifier, storage_a.identifier, storage_b.identifier])
# # it can transfer A and B, but at most 10 mL at a time
# qi_transfer_capacity = QualityIdentifier(name="transfer_capacity")
# transferor_0[qi_transfer_capacity] = Quality(identifier=qi_transfer_capacity, value=10, unit="mL")
# # initially it sits at rack_0
# qi_robot_position = QualityIdentifier(name="robot_position")
# transferor_0[qi_robot_position] = Quality(identifier=qi_robot_position, value=rack_0.identifier)
#
# # two heaters
# heater_0 = Artifact(type="HEATER")
# heater_1 = Artifact(type="HEATER")
# # set their heating/cooling rates
# qi_heating_rate = QualityIdentifier(name="heating_rate")
# qi_cooling_rate = QualityIdentifier(name="cooling_rate")
# heater_0[qi_heating_rate] = Quality(identifier=qi_heating_rate, value=0.5, unit="C/s")
# heater_1[qi_heating_rate] = Quality(identifier=qi_heating_rate, value=1, unit="C/s")
# heater_0[qi_cooling_rate] = Quality(identifier=qi_cooling_rate, value=0.05, unit="C/s")
# heater_1[qi_cooling_rate] = Quality(identifier=qi_cooling_rate, value=0.1, unit="C/s")
#
# # another robot arm can access rack_0 and both heaters
# transferor_1 = Artifact(type="TRANSFEROR")
# transferor_1[qi_robot_access] = Quality(identifier=qi_robot_access,
#                                         value=[rack_0.identifier, heater_0.identifier, heater_1.identifier])
# # it can only transfer at most 1 vial at a time
# transferor_1[qi_transfer_capacity] = Quality(identifier=qi_transfer_capacity, value=1)
#
# # another rack holding finished vials
# rack_1 = Artifact(type="RACK")
# # 50 slots
# rack_1[qi_slot_capacity] = Quality(identifier=qi_slot_capacity, value=50)
#
# # another robot arm can access rack_1 and both heaters
# transferor_2 = Artifact(type="TRANSFEROR")
# transferor_2[qi_robot_access] = Quality(identifier=qi_robot_access,
#                                         value=[rack_1.identifier, heater_0.identifier, heater_1.identifier])
# # it can only transfer at most 1 vial at a time
# transferor_2[qi_transfer_capacity] = Quality(identifier=qi_transfer_capacity, value=1)
#
# # chemicals
# qi_chemical_amount = QualityIdentifier(name="chemical_amount")
# qi_chemical_composition = QualityIdentifier(name="chemical_composition")
# qi_chemical_boundary = QualityIdentifier(name="chemical_boundary")
# init_chemical_A = Artifact(type="MATTER")
# init_chemical_A[qi_chemical_amount] = Quality(identifier=qi_chemical_amount, value=5000, unit="mL")
# init_chemical_A[qi_chemical_composition] = Quality(identifier=qi_chemical_composition, value="A")
# init_chemical_A[qi_chemical_boundary] = Quality(identifier=qi_chemical_boundary, value=storage_a.identifier)
# init_chemical_B = Artifact(type="MATTER")
# init_chemical_B[qi_chemical_amount] = Quality(identifier=qi_chemical_amount, value=8000, unit="mL")
# init_chemical_B[qi_chemical_composition] = Quality(identifier=qi_chemical_composition, value="B")
# init_chemical_B[qi_chemical_boundary] = Quality(identifier=qi_chemical_boundary, value=storage_b.identifier)
#
