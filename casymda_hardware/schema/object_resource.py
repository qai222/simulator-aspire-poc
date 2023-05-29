from simpy import Resource, Environment

from hardware_pydantic.lab_objects import LabObject


class LabObjectResource:

    def __init__(self, lab_object: LabObject, resource: Resource, env: Environment):
        self.lab_object = lab_object
        self.resource = resource
        self.env = env

    @classmethod
    def from_lab_object(cls, o: LabObject, env: Environment):
        return cls(o, Resource(env, capacity=1), env)
