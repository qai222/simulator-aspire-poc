from __future__ import annotations

from hardware_pydantic.base import *


class Vial(LabObject):
    position: str | float | None = None
    position_relative: str | None = None
    content: dict[str, float] = dict()

    @property
    def content_sum(self) -> float:
        if len(self.content) == 0:
            return 0
        return sum(self.content.values())

    def add_content(self, content: dict[str, float]):
        for k, v in content.items():
            if k not in self.content:
                self.content[k] = content[k]
            else:
                self.content[k] += content[k]

    def remove_content(self, amount: float) -> dict[str, float]:
        # TODO by default the content is homogeneous liquid
        pct = amount / self.content_sum
        removed = dict()
        for k in self.content:
            removed[k] = self.content[k] * pct
            self.content[k] -= removed[k]
        return removed


class Rack(LabObject):
    position: str | None = None
    content: dict[str, str | None] = {"A1": None, "A2": None, "B1": None, "B2": None}
    """ {<string position>: <vial identifier> | None} """

    @property
    def capacity(self):
        return len(self.content)

    @property
    def vials(self) -> set[str]:
        return set([v for v in self.content.values() if v is not None])

    @classmethod
    def from_capacity(cls, capacity: int = 16, rack_id: str = None):
        content = {str(i): None for i in range(capacity)}
        if rack_id is None:
            return cls(content=content)
        else:
            return cls(content=content, identifier=rack_id)
