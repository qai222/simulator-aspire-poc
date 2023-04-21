from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

"""
in addition to using `artifact` for devices, other classes are available for other components of the system
[material](http://purl.allotrope.org/ontologies/material#AFM_0000275)
e.g. [portion of material](http://purl.obolibrary.org/obo/CHMO_0000993)
"""


class Quality(BaseModel):
    """
    [quality](http://purl.obolibrary.org/obo/BFO_0000019)
    """

    name: str
    value: str | int | float | bool | None
    related_to: tuple[str, str] | None = None  # (<relative identifier>, <relation type>)
    unit: str = None

    def __hash__(self):
        return hash((self.name, self.value, self.related_to, self.unit))

    @property
    def is_relational(self) -> bool:
        return self.related_to is not None


class Artifact(BaseModel):
    """
    [artifact](http://purl.allotrope.org/ontologies/equipment#AFE_0002099)
    """

    identifier: str
    """ 
    unique string identifier for the instance 
    """

    type: str
    """
    type of the artifact, e.g. `vial`
    """

    state: set[Quality]

    @property
    def quality_dict(self):
        return {(q.name, q.related_to): q for q in self.state}

    def modify_quality(self, name: str, new_value=None, related_to: tuple[str, str] = None, unit: str = None,
                       remove=False):
        # TODO this by default includes state history
        q = Quality(name=name, value=new_value, related_to=related_to, unit=unit)
        qk = (name, related_to)
        if qk not in self.quality_dict:
            if remove:
                logger.warning(f"removing non existing quality: {qk}")
                return
        else:
            old_q = self.quality_dict[qk]
            self.state.remove(old_q)
        self.state.add(q)

    @property
    def internal_qualities(self):
        """
        a set of qualities describing the internal state of the instance
        [system configuration](http://purl.allotrope.org/ontologies/quality#AFQ_0000036)

        QA: I could also use [realizable entity](http://purl.obolibrary.org/obo/BFO_0000017)
            but since for now there is no update from the real system, the objects of our "actions"
            can only be configurations.
        """
        return {q for q in self.state if not q.is_relational}

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
        return {q for q in self.state if q.is_relational}

    @property
    def relatives(self) -> set[str]:
        return {q.related_to[0] for q in self.state}

    def __hash__(self):
        return hash(self.identifier)

    def __str__(self):
        qualities = '\n\t'.join([s.json() for s in self.state])
        return f"{self.type} -- {self.identifier}\n\t{qualities}"


class System(BaseModel):
    artifacts: set[Artifact]

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
