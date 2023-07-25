from __future__ import annotations

from .abstract import Individual


class ChemicalIdentifier(Individual):
    value: str

    type: str = "NAME"


class Compound(Individual):
    chemical_identifier: ChemicalIdentifier

    amount: float | None = None

    amount_unit: str | None = None


class HeterogeneousMixture:
    # TODO implement
    pass


class HomogenousMixture(Individual):
    from_compounds: list[Compound]

    phase: str = "SOLUTION"

    amount: float | None = None

    amount_unit: float | None = None


class HardwareClass(Individual):
    name: str

    description: str = ""


class HardwareUnit(Individual):
    hardware_class: HardwareClass

    made_of: list[ChemicalIdentifier]

    # initial_position: tuple[float, float] = (0.0, 0.0)
