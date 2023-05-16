from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from loguru import logger
from pydantic import Field, BaseModel

from ..abstraction import Artifact, QualityIdentifier
from ..utils import str_uuid


class ArtifactCondition(BaseModel):
    artifact: Artifact

    target_qi: QualityIdentifier

    condition: Literal["GT", "LT", "EQ"] = "EQ"  # TODO this is so ugly...

    condition_parameter: str | float

    @property
    def satisfy(self) -> bool:
        if self.condition == "GT":
            return self.artifact[self.target_qi].value > self.condition_parameter
        if self.condition == "LT":
            return self.artifact[self.target_qi].value < self.condition_parameter
        if self.condition == "EQ":
            return self.artifact[self.target_qi].value == self.condition_parameter


class ActionCondition(BaseModel):
    # anchor_action_identifier: str
    anchor_action: Action  # well this is circular type hint...

    precede_anchor: bool = False

    succeed_anchor: bool = False

    @property
    def satisfy(self) -> bool:
        satisfy = True
        if self.precede_anchor and self.anchor_action.action_state != "INITIALIZED":
            satisfy = False
        if self.succeed_anchor and self.anchor_action.action_state != "EXECUTED":
            satisfy = False
        return satisfy

    class Config:
       undefined_types_warning=False

    # TODO max delay?


class Action(BaseModel, ABC):
    # meta
    identifier: str = Field(default_factory=str_uuid)

    description: str = ""

    action_state: Literal["RUNNING", "INITIALIZED", "EXECUTED"] = "INITIALIZED"

    # pre actors
    artifact_conditions_pre: list[ArtifactCondition] = []

    action_conditions_pre: list[ActionCondition] = []

    # post actors
    duration: float | None = None  # None means its dynamically terminated

    artifact_conditions_post: list[ArtifactCondition] = []

    action_conditions_post: list[ActionCondition] = []

    # history
    time_start: datetime | None = None
    time_end: datetime | None = None

    def pre_check(self):
        # error out if not initialized
        logger.info(f"executing action: {self.identifier}")
        logger.info(f"this action is to: {self.description}")
        if self.action_state != "INITIALIZED":
            msg = f"this action cannot be executed due to its state of: {self.action_state}"
            logger.error(msg)
            raise RuntimeError(msg)

        # check pre conditions
        for c in self.action_conditions_pre + self.artifact_conditions_pre:
            if not c.satisfy:
                msg = f"precondition for this action is not satisfied: {c}"
                logger.error(msg)
                raise RuntimeError(msg)

        # otherwise execute normally
        self.action_state = "RUNNING"
        logger.info(f"this action is now RUNNING")

    def post_check(self):
        # error out if already executed
        logger.info(f"finalizing action: {self.identifier}")
        logger.info(f"this action is to: {self.description}")
        if self.action_state == "EXECUTED":
            msg = f"this action cannot be finalized due to its state of: {self.action_state}"
            logger.error(msg)
            raise RuntimeError(msg)

        # check post conditions
        # TODO shouldn't this be computed at runtime?
        for c in self.action_conditions_post + self.artifact_conditions_post:
            if not c.satisfy:
                msg = f"post condition for this action is not satisfied: {c}"
                logger.error(msg)
                raise RuntimeError(msg)

        # otherwise finalize normally
        self.action_state = "EXECUTED"
        logger.info(f"this action is now EXECUTED")

    @abstractmethod
    def execute(self):
        pass


"""
# or use decorator?
def action_execution(func):
    @wraps(func)
    def execute_check(action: Action):
        # do nothing if not initialized
        logger.info(f"executing action: {action.identifier}")
        logger.info(f"this action is to: {action.description}")
        if action.action_state != "INITIALIZED":
            logger.info(f"this action is SKIPPED due to its state of: {action.action_state}")
            return

        # check pre conditions
        for c in action.action_conditions_pre + action.artifact_conditions_pre:
            if not c.satisfy:
                msg = f"precondition for this action is not satisfied: {c}"
                logger.error(msg)
                raise RuntimeError(msg)

        # otherwise execute normally
        action.action_state = "RUNNING"
        logger.info(f"this action is now RUNNING")
        func(action)
        return

    return execute_check


def action_finalization(func):
    @wraps(func)
    def finalize_check(action: Action):
        # do nothing if already executed
        logger.info(f"finalizing action: {action.identifier}")
        logger.info(f"this action is to: {action.description}")
        if action.action_state == "EXECUTED":
            logger.info(f"this action has already been EXECUTED!")
            return

        # check pre conditions
        for c in action.action_conditions_post + action.artifact_conditions_post:
            if not c.satisfy:
                msg = f"post condition for this action is not satisfied: {c}"
                logger.error(msg)
                raise RuntimeError(msg)

        # otherwise finalize normally
        action.action_state = "EXECUTED"
        logger.info(f"this action is now EXECUTED")
        func(action)
        return

    return finalize_check
"""
