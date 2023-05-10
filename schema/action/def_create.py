from schema.action.base import Action
from schema.lab import Lab, Artifact
from schema.quality import Quality, QualityIdentifier


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
