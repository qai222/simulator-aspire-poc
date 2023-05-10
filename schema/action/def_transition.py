from schema.action.base import Action, ABC, abstractmethod
from schema.artifact import OrderedDict
from schema.lab import Artifact
from schema.quality import Quality, QualityIdentifier


class ActionTransitArtifact(Action, ABC):
    target: Artifact

    @abstractmethod
    def get_new_state(self) -> OrderedDict[QualityIdentifier, Quality]:
        pass

    def execute(self):
        new_state = self.get_new_state()
        self.target.transition(new_state=new_state, transition_action_identifier=self.identifier)
