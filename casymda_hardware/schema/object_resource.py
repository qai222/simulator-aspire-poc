from typing import Any
from simpy import PriorityResource, Resource, Environment

from hardware_pydantic.lab_objects import LabObject


class LabObjectResource:
    def __init__(
        self,
        lab_object: LabObject,
        resource: Any[Resource, PriorityResource],
        env: Environment,
    ):
        """The lab object resource is a wrapper around a simpy resource.

        Parameters
        ----------
        lab_object : LabObject
            the lab object to be wrapped.
        resource : Resource
            The simpy resource to be wrapped.
        env : Environment
            The `simpy` environment.

        """
        self.lab_object = lab_object
        self.resource = resource
        self.env = env

    @classmethod
    def from_lab_object(
        cls, o: LabObject, env: Environment, use_priority: bool = False
    ):
        """Create a lab object resource from a lab object.

        Parameters
        ----------
        o : LabObject
            The lab object to be wrapped.
        env : Environment
            The `simpy` environment.
        use_priority : bool, optional
            Whether to use a priority resource. Default=False.

        Returns
        -------
        LabObjectResource
            The lab object resource.

        """
        if use_priority:
            return cls(o, PriorityResource(env, capacity=1), env)
        else:
            return cls(o, Resource(env, capacity=1), env)
