from collections import OrderedDict

from ..abstraction import Artifact, Quality, QualityIdentifier
from .action_base import Action, ABC, abstractmethod


class ActionTransitArtifact(Action, ABC):
    target: Artifact

    @abstractmethod
    def get_new_state(self) -> OrderedDict[QualityIdentifier, Quality]:
        pass

    def execute(self):
        new_state = self.get_new_state()
        self.target.transition(new_state=new_state, transition_action_identifier=self.identifier)
