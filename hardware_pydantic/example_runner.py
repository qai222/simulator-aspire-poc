from hardware_pydantic import *

"""
an example lab
                      -- heater 1 --
rack 1 -- transferor 1              transferor 2 -- rack 2
                      -- heater 2 --
"""
instructions = set()

rack1 = Rack(identifier='rack1')
rack2 = Rack(identifier='rack2')
transferor1 = VialTransferor(identifier='crane1')
transferor2 = VialTransferor(identifier='crane2')
heater1 = Heater(identifier='heater1')
heater2 = Heater(identifier='heater2')
lab = Lab()
for d in [rack1, rack2, transferor1, transferor2, heater1, heater2]:
    lab.add_object(d)

# spawn vial in rack1
for ilabel, label in enumerate(rack1.content):
    v = Vial(identifier=f"vial{ilabel + 1}", position=label, position_relative=rack1.identifier)
    rack1.content[label] = v.identifier
    lab.add_object(v)

# move vial1 from rack1 to heater1
ins_move_rack1_to_heater1 = Instruction(
    actor_device=transferor1,
    device_action_method_name="action__transfer",
    device_action_method_parameters={
        "from_obj": rack1,
        "to_obj": heater1,
        "transferee": lab.dict_object['vial1'],
        "to_position": None,
    },
)

# move vial1 from heater1 to rack2
ins_move_heater1_to_rack2 = Instruction(
    actor_device=transferor1,
    device_action_method_name="action__transfer",
    device_action_method_parameters={
        "from_obj": heater1,
        "to_obj": rack2,
        "transferee": lab.dict_object['vial1'],
        "to_position": 'A2',
    },
    dependent_instructions=[ins_move_rack1_to_heater1.identifier, ]
)

ins_set_heater1_to_200 = Instruction(
    actor_device=heater1,
    device_action_method_name="action__set_point",
    device_action_method_parameters={
        "set_point": 200,
    },
    dependent_instructions=[ins_move_rack1_to_heater1.identifier, ]
)

ins_heater1_heating_to_set_point = Instruction(
    actor_device=heater1,
    device_action_method_name="action__heat_process",
    device_action_method_parameters={},
    dependent_instructions=[ins_set_heater1_to_200.identifier, ]
)

instructions.add(ins_move_heater1_to_rack2)
instructions.add(ins_move_rack1_to_heater1)
instructions.add(ins_set_heater1_to_200)
instructions.add(ins_heater1_heating_to_set_point)

lab.act_by_instruction(ins_move_rack1_to_heater1)
lab.act_by_instruction(ins_move_heater1_to_rack2)
lab.act_by_instruction(ins_set_heater1_to_200)
lab.act_by_instruction(ins_heater1_heating_to_set_point)
