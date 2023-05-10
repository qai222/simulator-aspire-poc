from .action_base import Action
from ..abstraction import Lab, Artifact, Quality, QualityIdentifier


class ActionCreateArtifact(Action):
    creation: Artifact

    lab: Lab

    def execute(self):
        qi = QualityIdentifier(name="CreatedBy")
        self.creation[qi] = Quality(identifier=qi, value=self.identifier)
        self.lab.artifacts[self.creation.identifier] = self.creation


class ActionAnnihilateArtifact(Action):
    creation: Artifact

    lab: Lab

    def execute(self):
        qi = QualityIdentifier(name="DestroyedBy")
        self.creation[qi] = Quality(identifier=qi, value=self.identifier)
        self.lab.artifacts.pop(self.creation.identifier)
