from __future__ import annotations

from copy import deepcopy
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
        logger.warning(f">> RUNNING *{func.__name__}* of *{self.__class__.__name__}*: {self.identifier}")
        for k, v in kwargs.items():
            logger.info(f"action parameter name: {k}")
            logger.info(f"action parameter value: {v}")
        duration = _func(self, *args, **kwargs)
        logger.warning(f">> FINISHED with duration: {duration}")
        return duration

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
        return deepcopy(self.__dict__)

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

    an action method must start with "action__"
    """

    @property
    def action_method_names(self):
        """ a sorted list of the names of all defined action methods """
        return sorted({k for k in dir(self) if k.startswith("action__")})

    def act(self, action_method_name: str = "action__dummy", action_parameters: dict[str, Any] = None) -> float:
        """
        perform the action defined by `action_method_name`,
        note an action method will always return the time cost
        """
        if action_parameters is None:
            action_parameters = dict()
        assert action_method_name in self.action_method_names
        return getattr(self, action_method_name)(**action_parameters)

    def act_by_instruction(self, i: Instruction) -> float:
        """ perform action with an instruction """
        assert i.actor_device == self
        return self.act(
            action_method_name=i.action_method,
            action_parameters=i.action_parameters,
        )

    @action_method_logging
    def action__dummy(self, **kwargs) -> float:
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
    action_method: str = "action__dummy"
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

    @property
    def instruction_graph(self):
        # TODO implement
        return

