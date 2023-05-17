from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from loguru import logger
from monty.json import MSONable


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
        logger.info(f"running *{func.__name__}* of *{self.__class__.__name__}*: {self.identifier}")
        return _func(self, *args, **kwargs)

    return function_caller


class Individual(MSONable):
    """ a thing with an identifier """

    def __init__(self, identifier: str = str_uuid(), ):
        self.identifier = identifier

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
    """

    @property
    def state(self) -> dict:
        return {k: v for k, v in self.as_dict().items() if not k.startswith("@")}


class Device(LabObject):
    """
    a `LabObject` that

    1. can receive instructions
    2. can change its state and other lab objects' states, it cannot change another device's state!

    an action method must start with "action__"
    """

    _default_action_duration = 1e-5

    def __init__(self, identifier: str = str_uuid(), occupied: bool = False):
        super().__init__(identifier=identifier)
        self.occupied = occupied

    def validate_state(self, state: dict) -> bool: pass

    def validate_current_state(self) -> bool:
        return self.validate_state(self.state)

    @property
    def action_method_names(self):
        return sorted({k for k in dir(self) if k.startswith("action__")})

    def act(self, action_method_name: str = "action__dummy", action_parameters: dict[str, Any] = None) -> float:
        if action_parameters is None:
            action_parameters = dict()
        assert action_method_name in self.action_method_names
        return getattr(self, action_method_name)(**action_parameters)

    def act_by_instruction(self, i: Instruction) -> float:
        return self.act(action_method_name=i.device_action_method_name,
                        action_parameters=i.device_action_method_parameters)

    @action_method_logging
    def action__dummy(self, **kwargs) -> float:
        return Device._default_action_duration

    def action_occupy(self):
        self.occupied = True

    def action_free_occupied(self):
        self.occupied = False


class Instruction(Individual):
    """
    an instruction sent to a device for an action

    instruction:
    - an instruction is sent to and received by one and only one `Device` instance (the `actor`) instantly
    - an instruction requests one and only one `action_method` from the `Device` instance
    - an instruction contains static parameters that the `action_method` needs
    - an instruction can involve zero, one or more `LabObject` instances
    - an instruction cannot involve any `Device` instance except the `actor`
    - an instruction can be either 'INIT' or 'SENT'
    - an instruction can only be transitioned from 'INIT' to 'SENT'
    - this transition can only happen when
        - all of `dependent_instructions` are `SENT` or;
        - any of `dependent_instructions` are `SENT`

    action:
    - an action is a physical process performed following an instruction
    - an action
        - starts when
            - the actor is available, and
            - the action is at the top of the queue of that actor
        - ends when
            - the duration, returned by the action method, has passed
    """

    def __init__(self, actor: Device,
                 device_action_method_parameters: dict[str, Any] = None,
                 device_action_method_name: str = "action__dummy", identifier: str = str_uuid(),
                 state: Literal['INIT', 'SENT'] = 'INIT', created_by: str = "ADMIN",
                 dependent_instructions: list[Instruction] = None, dependency_type: str = Literal["ANY", "ALL"],
                 description: str = "a dummy action"):
        super().__init__(identifier)
        if device_action_method_parameters is None:
            device_action_method_parameters = dict()
        self.device_action_method_parameters = device_action_method_parameters
        self.device_action_method_name = device_action_method_name
        self.actor = actor
        self.description = description
        self.dependency_type = dependency_type
        if dependent_instructions is None:
            dependent_instructions = []
        self.dependent_instructions = dependent_instructions
        self.created_by = created_by
        self.state = state

    def send(self):
        assert self.state == "INIT"
        self.state = "SENT"


class Lab(Individual):
    def __init__(self, identifier: str = str_uuid(), devices: tuple[Device] = (), ):
        super().__init__(identifier=identifier)
        self.devices = devices

    def add_device(self, d: Device):
        self.devices = tuple(list(self.devices) + [d, ])

    def remove_device(self, d: Device | str):
        if isinstance(d, str):
            d2r = [dd for dd in self.devices if dd == d][0]
        else:
            d2r = d
        self.devices = [dd for dd in self.devices if dd != d2r]

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        return {d.identifier: d.state for d in self.devices}


if __name__ == '__main__':
    device = Device()
    i = Instruction(actor=device)
    device.act(i.device_action_method_name, i.device_action_method_parameters)
