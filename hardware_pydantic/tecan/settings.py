from __future__ import annotations

from typing import Literal

from hardware_pydantic.base import Lab, LabObject, Instruction, BaseModel

TECAN_LAYOUT_SLOT_SIZE_X = 80
TECAN_LAYOUT_SLOT_SIZE_Y = 120
TECAN_LAYOUT_SLOT_SIZE_X_SMALL = 40
TECAN_LAYOUT_SLOT_SIZE_Y_SMALL = 20
TECAN_LAB = Lab()
TECAN_VIAL_TYPE = Literal["HRV", "MRV", "SV"]


class TecanLabObject(LabObject):
    """Base class for all Tecan lab objects."""
    def model_post_init(self, *args) -> None:
        """Add the object to the Tecan lab."""
        TECAN_LAB.add_object(self)


class TecanInstruction(Instruction):
    """Base class for all Tecan instructions."""
    def model_post_init(self, *args) -> None:
        """Add the instruction to the Tecan lab."""
        TECAN_LAB.add_instruction(self)

    @staticmethod
    def path_graph(ins_list: list[TecanInstruction]):
        """Add preceding instructions to each instruction."""
        for i in range(1, len(ins_list)):
            ins = ins_list[i]
            pre_ins_id = ins_list[i-1].identifier
            if pre_ins_id not in ins.preceding_instructions:
                ins.preceding_instructions.append(pre_ins_id)


class TecanLayout(BaseModel):
    """A region appears in layout.

    Parameters
    ----------
    layout_position : tuple[float, float] | None
        left bot conor of the layout box. Default is None.
    layout_x : float
        The x length. Default is TECAN_LAYOUT_SLOT_SIZE_X.
    layout_y : float
        The y length. Default is TECAN_LAYOUT_SLOT_SIZE_Y.

    """

    layout_position: tuple[float, float] | None = None
    layout_x: float = TECAN_LAYOUT_SLOT_SIZE_X
    layout_y: float = TECAN_LAYOUT_SLOT_SIZE_Y

    @classmethod
    def from_relative_layout(
            cls,
            layout_relation: Literal["above", "right_to"] = None,
            layout_relative: TecanLayout = None,
            layout_x: float = TECAN_LAYOUT_SLOT_SIZE_X,
            layout_y: float = TECAN_LAYOUT_SLOT_SIZE_Y,
    ):
        """Create a TecanLayout from a relative layout.

        Parameters
        ----------
        layout_relation : Literal["above", "right_to"], optional
            The relation between the new layout and the relative layout. Default is None.
        layout_relative : TecanLayout, optional
            The relative layout. Default is None.
        layout_x : float, optional
            The x length. Default is TECAN_LAYOUT_SLOT_SIZE_X.
        layout_y : float, optional
            The y length. Default is TECAN_LAYOUT_SLOT_SIZE_Y.

        Returns
        -------
        TecanLayout
            The TecanLayout object.

        Raises
        ------
        ValueError
            If the layout_relation is not "above" or "right_to".

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
