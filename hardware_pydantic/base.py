from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .utils import str_uuid


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
        # TODO can we have a pydantic model history tracker? similar to https://pypi.org/project/pydantic-changedetect/
        # TODO mutable fields vs immutable fields?
    """

    @property
    def state(self) -> dict:
        return self.__dict__

    def validate_state(self, state: dict) -> bool:
        pass

    def validate_current_state(self) -> bool:
        return self.validate_state(self.state)


class PreActError(Exception): pass


class PostActError(Exception): pass


class Device(LabObject):
    """
    a `LabObject` that
    1. can receive instructions
    2. can change its state and other lab objects' states using its action methods,
    3. cannot change another device's state

    # TODO a decorator to check pairing of pre__ and post__
    """

    @property
    def action_names(self) -> list[str]:
        """ a sorted list of the names of all defined actions """
        prefix = "pre__"
        names = sorted({k[len(prefix):] for k in dir(self) if k.startswith(prefix)})
        return names

    @staticmethod
    def get_pre_actor_name(action_name: str) -> str:
        return f"pre__{action_name}"

    @staticmethod
    def get_post_actor_name(action_name: str) -> str:
        return f"post__{action_name}"

    def act(
            self,
            action_name: str = "dummy",
            action_parameters: dict[str, Any] = None,
            is_pre=True,
    ):
        assert action_name in self.action_names, f"{action_name} not in {self.action_names}"
        if action_parameters is None:
            action_parameters = dict()
        if is_pre:
            method_name = Device.get_pre_actor_name(action_name)
        else:
            method_name = Device.get_post_actor_name(action_name)
        return getattr(self, method_name)(**action_parameters)

    def pre__dummy(self, **kwargs) -> tuple[list[LabObject], float]:
        """
        1. return a list of all involved objects, except self
        2. check the current states of involved objects, raise PreActorError if not met
        3. return the projected time cost
        """
        return [], 0

    def post__dummy(self, **kwargs) -> None:
        """
        1. make state transitions for involved objects, raise PostActorError if illegal transition
        """
        return

    def act_by_instruction(self, i: Instruction, is_pre: bool):
        """ perform action with an instruction """
        assert i.device == self
        return self.act(action_name=i.action_name, action_parameters=i.action_parameters, is_pre=is_pre)


class Instruction(Individual):
    """
    an instruction sent to a device for an action

    instruction:
    - an instruction is sent to and received by one and only one `Device` instance (the `actor`) instantly
    - an instruction requests one and only one `action_name` from the `Device` instance
    - an instruction contains static parameters that the `action_method` needs
    - an instruction can involve zero or more `LabObject` instances
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
    device: Device
    action_parameters: dict = dict()
    action_name: str = "dummy"
    description: str = ""

    preceding_type: Literal["ALL", "ANY"] = "ALL"
    preceding_instructions: list[str] = []


class Lab(BaseModel):
    dict_instruction: dict[str, Instruction] = dict()
    dict_object: dict[str, LabObject | Device] = dict()

    def act_by_instruction(self, i: Instruction, is_pre: bool = True):
        actor = self.dict_object[i.device.identifier]  # make sure we are working on the same device
        assert isinstance(actor, Device)
        return actor.act_by_instruction(i, is_pre=is_pre)

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
