from __future__ import annotations

from typing import Literal

from hardware_pydantic.base import Lab, LabObject, Instruction, BaseModel

JUNIOR_LAYOUT_SLOT_SIZE_X = 80
JUNIOR_LAYOUT_SLOT_SIZE_Y = 120
JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL = 40
JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL = 20
JUNIOR_LAB = Lab()
JUNIOR_VIAL_TYPE = Literal["HRV", "MRV", "SV"]


class JuniorLabObject(LabObject):
    """Base class for all Junior lab objects."""
    def model_post_init(self, *args) -> None:
        JUNIOR_LAB.add_object(self)


class JuniorInstruction(Instruction):
    """Base class for all Junior instructions."""
    def model_post_init(self, *args) -> None:
        JUNIOR_LAB.add_instruction(self)

    @staticmethod
    def path_graph(ins_list: list[JuniorInstruction]):
        for i in range(1, len(ins_list)):
            ins = ins_list[i]
            ins.preceding_instructions.append(ins_list[i-1].identifier)


class JuniorLayout(BaseModel):
    """A region appears in layout.

    Parameters
    ----------
    layout_position : tuple[float, float], optional
        The left bot conor of the layout box. Default is None.
    layout_x : float, optional
        The x length. Default is JUNIOR_LAYOUT_SLOT_SIZE_X.
    layout_y : float, optional
        The y length, Default is JUNIOR_LAYOUT_SLOT_SIZE_Y.

    """
    layout_position: tuple[float, float] | None = None
    layout_x: float = JUNIOR_LAYOUT_SLOT_SIZE_X
    layout_y: float = JUNIOR_LAYOUT_SLOT_SIZE_Y

    @classmethod
    def from_relative_layout(
            cls,
            layout_relation: Literal["above", "right_to"] = None,
            layout_relative: JuniorLayout = None,
            layout_x: float = JUNIOR_LAYOUT_SLOT_SIZE_X,
            layout_y: float = JUNIOR_LAYOUT_SLOT_SIZE_Y,
    ):
        """Create a layout from a relative layout.

        Parameters
        ----------
        layout_relation : Literal["above", "right_to"], optional
            The relation between the new layout and the relative layout. Default is None.
        layout_relative : JuniorLayout, optional
            The relative layout. Default is None.
        layout_x : float, optional
            The x length. Default is JUNIOR_LAYOUT_SLOT_SIZE_X.
        layout_y : float, optional
            The y length, Default is JUNIOR_LAYOUT_SLOT_SIZE_Y.

        Returns
        -------
        layout : JuniorLayout
            The layout.

        """
        if layout_relative is None:
            abs_layout_position = (0, 0)
        else:
            if layout_relation == "above":
                abs_layout_position = (
                    layout_relative.layout_position[0],
                    layout_relative.layout_position[1] + layout_relative.layout_y + 20
                )
            elif layout_relation == "right_to":
                abs_layout_position = (
                    layout_relative.layout_position[0] + layout_relative.layout_x + 20,
                    layout_relative.layout_position[1],
                )
            else:
                raise ValueError
        return cls(layout_position=abs_layout_position, layout_x=layout_x, layout_y=layout_y, )
