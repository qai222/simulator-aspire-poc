from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field


def str_uuid() -> str:
    return str(uuid4())


def resolve_function(func):
    if isinstance(func, staticmethod):
        return func.__func__
    return func


def action_method_logging(func):
    """ logging decorator for `action_method` of a `Device` """

    def function_caller(self: Device, *args, **kwargs):
        _func = resolve_function(func)
        logger.warning(f">> ACTION COMMITTED *{func.__name__}* of *{self.__class__.__name__}*: {self.identifier}")
        for k, v in kwargs.items():
            logger.info(f"action parameter name: {k}")
            logger.info(f"action parameter value: {v}")
        _func(self, *args, **kwargs)

    return function_caller


class Individual(BaseModel):
    """ a thing with an identifier """

    identifier: str = Field(default_factory=str_uuid)

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other: Individual):
        return self.identifier == other.identifier


class LabObject(Individual):
    """
    a physical object in a lab that
    1. is not a chemical designed to participate in reactions
    2. has constant, physically defined 3D boundary
    3. has a (non empty) set of measurable qualities that need to be tracked, this set is the state of this `LabObject`
        #TODO can we have a pydantic model history tracker? similar to https://pypi.org/project/pydantic-changedetect/
    """

    @property
    def state(self) -> dict:
        return self.__dict__

    def validate_state(self, state: dict) -> bool:
        pass

    def validate_current_state(self) -> bool:
        return self.validate_state(self.state)


class Device(LabObject):
    """
    a `LabObject` that

    1. can receive instructions
    2. can change its state and other lab objects' states using its action methods,
        it cannot change another device's state!

    an action method's name follows "action__{action_name}"

    a problem in simulation:
    an action should change the states of involved objects after the required time has passed
    this means
        - the time cost for finishing this action should be first computed at runtime
        - then, after the time has passed, change the states of objects through the action method
    we call the former a `projection`, an action method should always be accompanied by a projection method
    # TODO a decorator to check this pairing
    """

    @property
    def action_names(self):
        """ a sorted list of the names of all defined action methods """
        return sorted({k[8:] for k in dir(self) if k.startswith("action__")})

    @staticmethod
    def get_projection_method_name(action_name: str):
        return f"projection__{action_name}"

    @staticmethod
    def get_action_method_name(action_name: str):
        return f"action__{action_name}"

    def project(self, action_name: str = "dummy", action_parameters: dict[str, Any] = None) -> float:
        """ projection method uses the same `action_parameters` """
        if action_parameters is None:
            action_parameters = dict()
        assert action_name in self.action_names
        projection_method_name = Device.get_projection_method_name(action_name)
        return getattr(self, projection_method_name)(**action_parameters)

    def act(self, action_name: str = "dummy", action_parameters: dict[str, Any] = None) -> None:
        """
        perform the action defined by `action_method_name`,
        note an action method will always return the time cost
        """
        if action_parameters is None:
            action_parameters = dict()
        assert action_name in self.action_names
        action_method_name = Device.get_action_method_name(action_name)
        getattr(self, action_method_name)(**action_parameters)

    def act_by_instruction(self, i: Instruction) -> None:
        """ perform action with an instruction """
        assert i.actor_device == self
        self.act(action_name=i.action_name, action_parameters=i.action_parameters, )

    def project_by_instruction(self, i: Instruction) -> float:
        assert i.actor_device == self
        return self.project(action_name=i.action_name, action_parameters=i.action_parameters, )

    @action_method_logging
    def action__dummy(self, **kwargs):
        pass

    def project__dummy(self, **kwargs) -> float:
        return 1e-5


class Instruction(Individual):
    """
    an instruction sent to a device for an action

    instruction:
    - an instruction is sent to and received by one and only one `Device` instance (the `actor`) instantly
    - an instruction requests one and only one `action_method` from the `Device` instance
    - an instruction contains static parameters that the `action_method` needs
    - an instruction can involve zero, one or more `LabObject` instances
    - an instruction cannot involve any `Device` instance except the `actor`

    action:
    - an action is a physical process performed following an instruction
    - an action
        - starts when
            - the actor is available, and
            - the action is at the top of the queue of that actor
        - ends when
            - the duration, returned by the action method of the actor, has passed
    """
    actor_device: Device
    action_parameters: dict = dict()
    action_name: str = "dummy"
    description: str = ""

    preceding_type: Literal["ALL", "ANY"] = "ALL"
    preceding_instructions: list[str] = []


class Lab(BaseModel):
    dict_instruction: dict[str, Instruction] = dict()
    dict_object: dict[str, LabObject] = dict()

    def act_by_instruction(self, i: Instruction):
        actor = self.dict_object[i.actor_device.identifier]  # make sure we are working on the same device
        assert isinstance(actor, Device)
        return actor.act_by_instruction(i)

    def add_instruction(self, i: Instruction):
        assert i.identifier not in self.dict_instruction
        self.dict_instruction[i.identifier] = i

    def remove_instruction(self, i: Instruction | str):
        if isinstance(i, str):
            assert i in self.dict_instruction
            self.dict_instruction.pop(i)
        else:
            assert i.identifier in self.dict_instruction
            self.dict_instruction.pop(i.identifier)

    def add_object(self, d: LabObject):
        assert d.identifier not in self.dict_object
        self.dict_object[d.identifier] = d

    def remove_object(self, d: LabObject | str):
        if isinstance(d, str):
            assert d in self.dict_object
            self.dict_object.pop(d)
        else:
            assert d.identifier in self.dict_object
            self.dict_object.pop(d.identifier)

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        return {d.identifier: d.state for d in self.dict_object.values()}

    def __repr__(self):
        return "\n".join([f"{obj.identifier}: {obj.state}" for obj in self.dict_object.values()])

    def __str__(self):
        return self.__repr__()

    @property
    def instruction_graph(self):
        # TODO implement
        return
