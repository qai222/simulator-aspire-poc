from __future__ import annotations

from hardware.base import *


class Vial(LabObject):

    def __init__(self,
                 identifier: str = str_uuid(),
                 content: dict[str, float] = None,
                 position: str | float = None,
                 position_relative: str = None,
                 max_capacity: float = 5,
                 ):
        super().__init__(identifier=identifier)
        self.position_relative = position_relative
        self.position = position
        if content is None:
            content = dict()
        self.content = content
        self.max_capacity = max_capacity

    def validate_state(self, state: dict[str, Any]) -> bool:
        pass

    @property
    def content_sum(self) -> float:
        return sum(self.content.values())

    def add_content(self, content: dict[str, float]):
        for k, v in content.items():
            if k not in self.content:
                self.content = content[k]
            else:
                self.content += content[k]

    def remove_content(self, amount: float) -> dict[str, float]:
        # TODO by default the content is homogeneous liquid
        pct = amount / self.content_sum
        removed = dict()
        for k in self.content:
            removed[k] = self.content[k] * pct
            self.content[k] -= removed[k]
        return removed
