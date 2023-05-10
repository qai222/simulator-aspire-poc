from .artifact import Artifact, BaseModel


class Lab(BaseModel):  # TODO should just use a dict
    artifacts: dict[str, Artifact] = dict()

    def __str__(self):
        return "=== SYSTEM ===\n" + \
            "\n".join([a.__str__() for a in self.artifacts.values()]) + \
            "\n=== SYSTEM END ==="
