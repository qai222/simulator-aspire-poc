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
    device=transferor1,
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
    device=transferor2,
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
    device=heater1,
    action_name="set_point",
    action_parameters={
        "set_point": 200,
    },
    preceding_instructions=[ins_move_rack1_to_heater1.identifier, ]
)

ins_heater1_heating_to_set_point = Instruction(
    identifier="ins_heater1_heating_to_set_point",
    device=heater1,
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
    ins_move_rack1_to_heater1  ready at: 0
    projected processing time: InstructionJob: ins_move_rack1_to_heater1 == 5
    involved objects: [Heater(identifier='heater1', set_point=25, set_point_max=400, reading=25, content=None), Vial(identifier='vial1', position='A1', position_relative='rack1', content={})]
    FINISHED STEP
    InstructionJob: ins_move_rack1_to_heater1 at: 5
    CURRENT LAB:
    rack1: {'identifier': 'rack1', 'content': {'A1': None, 'A2': 'vial2', 'B1': 'vial3', 'B2': 'vial4'}}
    rack2: {'identifier': 'rack2', 'content': {'A1': None, 'A2': None, 'B1': None, 'B2': None}}
    crane1: {'identifier': 'crane1'}
    crane2: {'identifier': 'crane2'}
    heater1: {'identifier': 'heater1', 'set_point': 25, 'set_point_max': 400, 'reading': 25, 'content': 'vial1'}
    heater2: {'identifier': 'heater2', 'set_point': 25, 'set_point_max': 400, 'reading': 25, 'content': None}
    vial1: {'identifier': 'vial1', 'position': None, 'position_relative': 'heater1', 'content': {}}
    vial2: {'identifier': 'vial2', 'position': 'A2', 'position_relative': 'rack1', 'content': {}}
    vial3: {'identifier': 'vial3', 'position': 'B1', 'position_relative': 'rack1', 'content': {}}
    vial4: {'identifier': 'vial4', 'position': 'B2', 'position_relative': 'rack1', 'content': {}}
    ============
    ins_set_heater1_to_200  ready at: 5
    ins_move_heater1_to_rack2  ready at: 5
    projected processing time: InstructionJob: ins_set_heater1_to_200 == 1e-05
    involved objects: []
    projected processing time: InstructionJob: ins_move_heater1_to_rack2 == 5
    involved objects: [Rack(identifier='rack2', content={'A1': None, 'A2': None, 'B1': None, 'B2': None}), Vial(identifier='vial1', position=None, position_relative='heater1', content={})]
    FINISHED STEP
    InstructionJob: ins_set_heater1_to_200 at: 5.00001
    CURRENT LAB:
    rack1: {'identifier': 'rack1', 'content': {'A1': None, 'A2': 'vial2', 'B1': 'vial3', 'B2': 'vial4'}}
    rack2: {'identifier': 'rack2', 'content': {'A1': None, 'A2': None, 'B1': None, 'B2': None}}
    crane1: {'identifier': 'crane1'}
    crane2: {'identifier': 'crane2'}
    heater1: {'identifier': 'heater1', 'set_point': 200, 'set_point_max': 400, 'reading': 25, 'content': 'vial1'}
    heater2: {'identifier': 'heater2', 'set_point': 25, 'set_point_max': 400, 'reading': 25, 'content': None}
    vial1: {'identifier': 'vial1', 'position': None, 'position_relative': 'heater1', 'content': {}}
    vial2: {'identifier': 'vial2', 'position': 'A2', 'position_relative': 'rack1', 'content': {}}
    vial3: {'identifier': 'vial3', 'position': 'B1', 'position_relative': 'rack1', 'content': {}}
    vial4: {'identifier': 'vial4', 'position': 'B2', 'position_relative': 'rack1', 'content': {}}
    ============
    ins_heater1_heating_to_set_point  ready at: 5.00001
    projected processing time: InstructionJob: ins_heater1_heating_to_set_point == 17.5
    involved objects: []
    FINISHED STEP
    InstructionJob: ins_move_heater1_to_rack2 at: 10
    CURRENT LAB:
    rack1: {'identifier': 'rack1', 'content': {'A1': None, 'A2': 'vial2', 'B1': 'vial3', 'B2': 'vial4'}}
    rack2: {'identifier': 'rack2', 'content': {'A1': None, 'A2': 'vial1', 'B1': None, 'B2': None}}
    crane1: {'identifier': 'crane1'}
    crane2: {'identifier': 'crane2'}
    heater1: {'identifier': 'heater1', 'set_point': 200, 'set_point_max': 400, 'reading': 25, 'content': None}
    heater2: {'identifier': 'heater2', 'set_point': 25, 'set_point_max': 400, 'reading': 25, 'content': None}
    vial1: {'identifier': 'vial1', 'position': 'A2', 'position_relative': 'rack2', 'content': {}}
    vial2: {'identifier': 'vial2', 'position': 'A2', 'position_relative': 'rack1', 'content': {}}
    vial3: {'identifier': 'vial3', 'position': 'B1', 'position_relative': 'rack1', 'content': {}}
    vial4: {'identifier': 'vial4', 'position': 'B2', 'position_relative': 'rack1', 'content': {}}
    ============
    FINISHED STEP
    InstructionJob: ins_heater1_heating_to_set_point at: 22.50001
    CURRENT LAB:
    rack1: {'identifier': 'rack1', 'content': {'A1': None, 'A2': 'vial2', 'B1': 'vial3', 'B2': 'vial4'}}
    rack2: {'identifier': 'rack2', 'content': {'A1': None, 'A2': 'vial1', 'B1': None, 'B2': None}}
    crane1: {'identifier': 'crane1'}
    crane2: {'identifier': 'crane2'}
    heater1: {'identifier': 'heater1', 'set_point': 200, 'set_point_max': 400, 'reading': 200, 'content': None}
    heater2: {'identifier': 'heater2', 'set_point': 25, 'set_point_max': 400, 'reading': 25, 'content': None}
    vial1: {'identifier': 'vial1', 'position': 'A2', 'position_relative': 'rack2', 'content': {}}
    vial2: {'identifier': 'vial2', 'position': 'A2', 'position_relative': 'rack1', 'content': {}}
    vial3: {'identifier': 'vial3', 'position': 'B1', 'position_relative': 'rack1', 'content': {}}
    vial4: {'identifier': 'vial4', 'position': 'B2', 'position_relative': 'rack1', 'content': {}}
    ============
    """
