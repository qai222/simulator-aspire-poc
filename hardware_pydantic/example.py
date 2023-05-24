from hardware_pydantic import *

"""
an example lab
                      -- heater 1 --
rack 1 -- transferor 1              transferor 2 -- rack 2
                      -- heater 2 --
"""
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
    identifier="ins_move_rack1_to_heater1",
    actor_device=transferor1,
    action_name="transfer",
    action_parameters={
        "from_obj": rack1,
        "to_obj": heater1,
        "transferee": lab.dict_object['vial1'],
        "to_position": None,
    },
)

# move vial1 from heater1 to rack2
ins_move_heater1_to_rack2 = Instruction(
    identifier="ins_move_heater1_to_rack2",
    actor_device=transferor2,
    action_name="transfer",
    action_parameters={
        "from_obj": heater1,
        "to_obj": rack2,
        "transferee": lab.dict_object['vial1'],
        "to_position": 'A2',
    },
    preceding_instructions=[ins_move_rack1_to_heater1.identifier, ]
)

ins_set_heater1_to_200 = Instruction(
    identifier="ins_set_heater1_to_200",
    actor_device=heater1,
    action_name="set_point",
    action_parameters={
        "set_point": 200,
    },
    preceding_instructions=[ins_move_rack1_to_heater1.identifier, ]
)

ins_heater1_heating_to_set_point = Instruction(
    identifier="ins_heater1_heating_to_set_point",
    actor_device=heater1,
    action_name="heat_process",
    action_parameters={},
    preceding_instructions=[ins_set_heater1_to_200.identifier, ]
)

lab.add_instruction(ins_heater1_heating_to_set_point)
lab.add_instruction(ins_set_heater1_to_200)
lab.add_instruction(ins_move_rack1_to_heater1)
lab.add_instruction(ins_move_heater1_to_rack2)

if __name__ == '__main__':
    lab.act_by_instruction(ins_move_rack1_to_heater1)
    lab.act_by_instruction(ins_move_heater1_to_rack2)
    lab.act_by_instruction(ins_set_heater1_to_200)
    lab.act_by_instruction(ins_heater1_heating_to_set_point)
    """
    2023-05-22 15:06:16.420 | WARNING  | hardware_pydantic.base:function_caller:25 - >> ACTION COMMITTED *action__transfer* of *VialTransferor*: crane1
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: from_obj
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: identifier='rack1' content={'A1': 'vial1', 'A2': 'vial2', 'B1': 'vial3', 'B2': 'vial4'}
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: to_obj
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: identifier='heater1' set_point=25 reading=25 content=None
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: transferee
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: identifier='vial1' position='A1' position_relative='rack1' content={}
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: to_position
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: None
    2023-05-22 15:06:16.421 | WARNING  | hardware_pydantic.base:function_caller:25 - >> ACTION COMMITTED *action__transfer* of *VialTransferor*: crane2
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: from_obj
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: identifier='heater1' set_point=25 reading=25 content=None
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: to_obj
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: identifier='rack2' content={'A1': None, 'A2': None, 'B1': None, 'B2': None}
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: transferee
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: identifier='vial1' position='A1' position_relative='rack1' content={}
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: to_position
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: A2
    2023-05-22 15:06:16.421 | WARNING  | hardware_pydantic.base:function_caller:25 - >> ACTION COMMITTED *action__set_point* of *Heater*: heater1
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:27 - action parameter name: set_point
    2023-05-22 15:06:16.421 | INFO     | hardware_pydantic.base:function_caller:28 - action parameter value: 200
    2023-05-22 15:06:16.421 | WARNING  | hardware_pydantic.base:function_caller:25 - >> ACTION COMMITTED *action__heat_process* of *Heater*: heater1
    """