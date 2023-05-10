from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schema.quality import Quality, QualityIdentifier
from schema.utils import str_uuid

"""
in addition to using `artifact` for devices, other classes are available for other components of the system
[material](http://purl.allotrope.org/ontologies/material#AFM_0000275)
e.g. [portion of material](http://purl.obolibrary.org/obo/CHMO_0000993)
"""

# TODO import this from a config file
ArtifactTypes = Literal["VIAL", "HEATER", "RACK", "ARM"]


class Artifact(BaseModel):
    """
    [artifact](http://purl.allotrope.org/ontologies/equipment#AFE_0002099)
    """

    identifier: str = Field(default_factory=str_uuid)
    """ 
    unique string identifier for the instance 
    """

    type: ArtifactTypes = "VIAL"
    """
    type of the artifact, e.g. `VIAL`
    """

    state: dict[QualityIdentifier, Quality] = dict()

    state_history: dict[str, dict[QualityIdentifier, Quality]] = dict()  # keys should be action identifiers

    def __getitem__(self, key: QualityIdentifier | str):
        if isinstance(key, str):
            key = QualityIdentifier(name=key)
        return self.state[key]

    @property
    def internal_qualities(self):
        """
        # TODO should `existence` be an internal quality?

        a set of qualities describing the internal state of the instance
        [system configuration](http://purl.allotrope.org/ontologies/quality#AFQ_0000036)

        QA: I could also use [realizable entity](http://purl.obolibrary.org/obo/BFO_0000017)
            but since for now there is no update from the real system, the objects of our "actions"
            can only be configurations.
        """
        return {q for q in self.state.values() if not q.is_relational}

    @property
    def relational_qualities(self):
        """
        a set of qualities describing the relational quality of the instance
        [relational quality](http://purl.obolibrary.org/obo/BFO_0000145)

        QA: it can be unclear if who should be the relative(s) to define a quality.

        There is an important assumption that such a quality can be fully defined by the relation between
        the owner of this state to **one** external source. If this assumption is not true then validations are needed to
        check the consistency of these definitions.
        """
        return {q for q in self.state.values() if q.is_relational}

    @property
    def relatives(self) -> set[str]:
        return {q.identifier.relative for q in self.state.values()}

    def __hash__(self):
        return hash(self.identifier)

    def __str__(self):
        qualities = '\n\t'.join([s.json() for s in self.state.values()])
        return f"{self.type} -- {self.identifier}\n\t{qualities}"

    # def modify_state(self, quality_identifier: QualityIdentifier, new_quality: Quality | None):
    #
    #     assert quality_identifier in self.quality_dict
    #     assert new_quality.identifier == quality_identifier
    #
    #     old_q = self.quality_dict[quality_identifier]
    #     self.state.remove(old_q)
    #     if new_quality is None:
    #         # remove the quality
    #         logger.warning(f"removing quality: {old_q}")
    #         return
    #     else:
    #         self.state.add(new_quality)
