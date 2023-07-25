from __future__ import annotations

from .abstract import Individual


class HardwareClass(Individual):
    name: str

    description: str = ""


class HardwareMaterial(Individual):
    name: str = "GLASS"

    description: str = ""


class HardwareUnit(Individual):
    hardware_class: HardwareClass

    hardware_material: HardwareMaterial

    initial_position: tuple[float, float] = (0, 0)



