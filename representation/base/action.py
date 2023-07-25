from __future__ import annotations

from .hardware import *


class Action(Individual):
    description: str = ""

    precedents: list[Action]

    # TODO action inputs
    # TODO action outputs
    # inputs: list[HomogenousMixture]
    # outputs: list[HomogenousMixture]
    inputs: list[Compound | ChemicalIdentifier]
    outputs: list[Compound | ChemicalIdentifier]

    # TODO pre conditions
    # TODO post conditions (necessary?)

    # TODO action effects:
    #  the effect of an action can be
    #  1. change data properties of individuals
    #  2. create/annihilate individuals
    #  3. create/annihilate object properties

    # TODO there could be a list of possible assignments
    #  ex. a liquid transfer can be assigned to [s, t, lh1] or [s, t, lh2]
    uses_hardware_unit: list[HardwareUnit]
