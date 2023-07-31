from __future__ import annotations

from typing import Any, Literal, Type

from N2G import drawio_diagram  # only used for drawing instruction DAG
from pydantic import BaseModel, Field

from simulator_aspire_poc.hardware_pydantic.utils import str_uuid

DEVICE_ACTION_METHOD_PREFIX = "action__"
DEVICE_ACTION_METHOD_ACTOR_TYPE = Literal['pre', 'post', 'proj']


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
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("layout"):
                continue
            d[k] = v
        return d

    def validate_state(self, state: dict) -> bool:
        pass

    def validate_current_state(self) -> bool:
        return self.validate_state(self.state)


class PreActError(Exception):
    pass


class PostActError(Exception):
    pass


class Device(LabObject):
    """
    a `LabObject` that
    1. can receive instructions
    2. can change its state and other lab objects' states using its action methods,
    3. cannot change another device's state # TODO does this actually matter?
    """

    @property
    def action_names(self) -> list[str]:
        """ a sorted list of the names of all defined actions """
        names = sorted(
            {k[len(DEVICE_ACTION_METHOD_PREFIX):] for k in dir(self) if k.startswith(DEVICE_ACTION_METHOD_PREFIX)}
        )
        return names

    def action__dummy(
            self,
            actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE = 'pre',
            **kwargs
    ) -> tuple[list[LabObject], float] | None:
        """
        ACTION: dummy
        DESCRIPTION: wait for certain amount of time
        PARAMS:
        """
        assert 'actor_type' not in kwargs
        if actor_type == 'pre':
            # the pre actor method of an action should
            # - check the current states of involved objects, raise PreActorError if not met
            return
        elif actor_type == 'post':
            # the post actor method of an action should
            # - make state transitions for involved objects, raise PostActorError if illegal transition
            return
        elif actor_type == 'proj':
            # the projection method of an action should
            # - return a list of all involved objects, except self
            # - return the projected time cost
            return [], 0
        else:
            raise ValueError

    def act(
            self,
            action_name: str = "dummy",
            actor_type: Literal['pre', 'post', 'proj'] = 'pre',
            action_parameters: dict[str, Any] = None,
    ):
        assert action_name in self.action_names, f"{action_name} not in {self.action_names}"
        if action_parameters is None:
            action_parameters = dict()
        method_name = DEVICE_ACTION_METHOD_PREFIX + action_name
        return getattr(self, method_name)(actor_type=actor_type, **action_parameters)

    def act_by_instruction(self, i: Instruction, actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE):
        """ perform action with an instruction """
        assert i.device == self
        return self.act(action_name=i.action_name, action_parameters=i.action_parameters, actor_type=actor_type)


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
    # TODO this has no effect as it is not passed to casymda

    preceding_instructions: list[str] = []


class Lab(BaseModel):
    dict_instruction: dict[str, Instruction] = dict()
    dict_object: dict[str, LabObject | Device] = dict()

    def __getitem__(self, identifier: str):
        return self.dict_object[identifier]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def act_by_instruction(self, i: Instruction, actor_type: DEVICE_ACTION_METHOD_ACTOR_TYPE):
        actor = self.dict_object[i.device.identifier]  # make sure we are working on the same device
        assert isinstance(actor, Device)
        return actor.act_by_instruction(i, actor_type=actor_type)

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

    def add_object(self, d: LabObject | Device):
        assert d.identifier not in self.dict_object
        self.dict_object[d.identifier] = d

    def remove_object(self, d: LabObject | Device | str):
        if isinstance(d, str):
            assert d in self.dict_object
            self.dict_object.pop(d)
        else:
            assert d.identifier in self.dict_object
            self.dict_object.pop(d.identifier)

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        return {d.identifier: d.state for d in self.dict_object.values()}

    def dict_object_by_class(self, object_class: Type):
        return {k: v for k, v in self.dict_object.items() if v.__class__ == object_class}

    def __repr__(self):
        return "\n".join([f"{obj.identifier}: {obj.state}" for obj in self.dict_object.values()])

    def __str__(self):
        return self.__repr__()

    @property
    def instruction_graph(self) -> drawio_diagram:
        diagram = drawio_diagram()
        diagram.add_diagram("Page-1")

        for k, ins in self.dict_instruction.items():
            diagram.add_node(id=f"{ins.identifier}\n{ins.description}")

        for ins in self.dict_instruction.values():
            for dep in ins.preceding_instructions:
                pre_ins = self.dict_instruction[dep]
                this_ins_node = f"{ins.identifier}\n{ins.description}"
                pre_ins_node = f"{pre_ins.identifier}\n{pre_ins.description}"
                diagram.add_link(pre_ins_node, this_ins_node, style="endArrow=classic")
        return diagram
