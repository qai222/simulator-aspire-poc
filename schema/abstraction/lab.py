from .artifact import Artifact, BaseModel


class Lab(BaseModel):
    artifacts: set[Artifact] = set()

    @property
    def state(self):
        s = dict()
        for art in self.artifacts:
            s[art.identifier] = art.state
        return s

    def __str__(self):
        return "=== SYSTEM ===\n" + \
            "\n".join([a.__str__() for a in self.artifacts]) + \
            "\n=== SYSTEM END ==="
