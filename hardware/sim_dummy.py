from hardware.devices import *

lab = Lab()

heater_0 = Heater()
heater_1 = Heater()
vial_0 = Vial()
vial_1 = Vial()

for d in [heater_0, heater_1, vial_0, vial_1]:
    lab.add_device(d)

i = Instruction(
    heater_0,
    device_action_method_parameters={"set_point": 12},
    device_action_method_name="action__set_point",
)
j = Instruction(
    heater_1,
    device_action_method_name="action__heat_object",
)

print(heater_0.state)
i.send()
cost = heater_0.act_by_instruction(i)
print(cost)
print(heater_0.state)
cost = heater_0.act_by_instruction(j)
print(cost)
print(heater_0.state)
