from __future__ import annotations

from .abstract import Individual
from pydantic import Field


class ChemicalIdentifier(Individual):
    value: str

    type: str = "NAME"


class Compound(Individual):
    chemical_identifier: ChemicalIdentifier = Field(..., exclude=True)

    amount: float | None = None

    amount_unit: str | None = None


class HeterogeneousMixture:
    # TODO implement
    pass


class HomogenousMixture(Individual):
    from_compounds: list[Compound] = Field(..., exclude=True)

    phase: str = "SOLUTION"

    amount: float | None = None

    amount_unit: float | None = None


class HardwareClass(Individual):
    name: str

    description: str = ""


class HardwareUnit(Individual):
    hardware_class: HardwareClass = Field(..., exclude=True)

    made_of: list[ChemicalIdentifier] = Field(..., exclude=True)

    # initial_position: tuple[float, float] = (0.0, 0.0)
