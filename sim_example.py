from casymda_hardware.model import *
from hardware_pydantic.example import *
import simpy

env = simpy.Environment()

model = Model(env, lab)

env.run()