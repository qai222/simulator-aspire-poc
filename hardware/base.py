from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def str_uuid() -> str:
    return str(uuid4())


class LabObject(BaseModel, ABC):
    """
    a physical object in a lab that
    1. is not a chemical designed to participate reactions
    2. has physically defined 3D boundary
    3. has a (non empty) set of measurable qualities that need to be tracked
    """
    identifier: str = Field(default_factory=str_uuid)

    # @abstractmethod
    # def validate_state(self, state: dict[str, Any]) -> bool:
    #     pass
    #
    # def validate_current_state(self) -> bool:
    #     return self.validate_state(self.__dict__)

    def __hash__(self):
        return hash(self.identifier)


class Device(LabObject, ABC):
    """
    a `LabOject` that
    1. can receive commands
    2. can change its state and other lab objects' states, it cannot change another device's state!
    """

    is_occupied: bool = False

    @staticmethod
    def change_state(other: LabObject, state: dict[str, Any]):
        # assert other.validate_state(state)
        other.__dict__.update(state)

    # @abstractmethod
    def wait(self, time: float):
        """ simpy action """
        pass


class Instruction(BaseModel, ABC):
    """ an instruction sent to a device for an action """
    identifier: str = Field(default_factory=str_uuid)

    state: Literal['INIT', 'RUN', 'FIN'] = 'INIT'

    created_by: str = "ADMIN"

    dep_action: list[Instruction] = []

    dep_state: dict[Device, dict[str, Any]] = dict()

    # subject: Device

    description: str = ""

    action_method_name: str

    # @abstractmethod
    def is_runnable(self) -> bool: pass  # check dep here

