from ..abstraction import Lab, Artifact, Quality, QualityIdentifier
from .action_base import Action


class ActionCreateArtifact(Action):
    creation: Artifact

    lab: Lab

    def execute(self):
        qi = QualityIdentifier(name="CreatedBy")
        self.creation[qi] = Quality(identifier=qi, value=self.identifier)
        self.lab.artifacts.add(self.creation)


class ActionAnnihilateArtifact(Action):
    creation: Artifact

    lab: Lab

    def execute(self):
        qi = QualityIdentifier(name="DestroyedBy")
        self.creation[qi] = Quality(identifier=qi, value=self.identifier)
        self.lab.artifacts.remove(self.creation)
