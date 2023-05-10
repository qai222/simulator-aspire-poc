from __future__ import annotations

from pydantic import BaseModel


class QualityIdentifier(BaseModel):
    name: str
    relative: str | None = None  # relative identifier
    relation: str | None = None  # e.g. `belongs to`

    def __hash__(self):
        return hash(self.as_tuple())

    def as_tuple(self) -> tuple[str, str | None, str | None]:
        return self.name, self.relative, self.relation

    @classmethod
    def from_tuple(cls, t: tuple[str, str | None, str | None]):
        return cls(name=t[0], relative=t[1], relation=t[2])


class Quality(BaseModel):
    """
    [quality](http://purl.obolibrary.org/obo/BFO_0000019)
    """

    identifier: QualityIdentifier
    value: bool | int | float | str
    unit: str | None = None

    def __hash__(self):
        return hash(self.identifier)

    @property
    def is_relational(self) -> bool:
        return self.identifier.relation is not None
