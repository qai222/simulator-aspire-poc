from __future__ import annotations

from typing import Literal, ClassVar, Type

from hardware_pydantic.base import Lab, LabObject, Instruction, JuniorOntology
from twa.data_model.base_ontology import BaseClass, BaseOntology, ObjectProperty, DatatypeProperty

JUNIOR_LAYOUT_SLOT_SIZE_X = 80
JUNIOR_LAYOUT_SLOT_SIZE_Y = 120
JUNIOR_LAYOUT_SLOT_SIZE_X_SMALL = 40
JUNIOR_LAYOUT_SLOT_SIZE_Y_SMALL = 20
JUNIOR_LAB = Lab()
JUNIOR_VIAL_TYPE = Literal["HRV", "MRV", "SV"]


class JuniorLabObject(LabObject):
    """Base class for all Junior lab objects."""
    def model_post_init(self, __context) -> None:
        # NOTE that the super().model_post_init() must be called to finalise the object creation
        # and have the object registred in the knowledge graph
        super().model_post_init(__context)
        JUNIOR_LAB.add_object(self)


class JuniorInstruction(Instruction):
    """Base class for all Junior instructions."""
    def model_post_init(self, *args) -> None:
        # NOTE that the super().model_post_init() must be called to finalise the object creation
        # and have the object registred in the knowledge graph
        super().model_post_init(*args)
        JUNIOR_LAB.add_instruction(self)

    @staticmethod
    def path_graph(ins_list: list[JuniorInstruction]):
        for i in range(1, len(ins_list)):
            ins = ins_list[i]
            ins.preceding_instructions.add(ins_list[i-1].identifier)

class Layout_x(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology

class Layout_y(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology

class Absolute_layout_position_x(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology

class Absolute_layout_position_y(DatatypeProperty):
    rdfs_isDefinedBy = JuniorOntology


class JuniorLayout(BaseClass):
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
    rdfs_isDefinedBy = JuniorOntology
    absolute_layout_position_x: Absolute_layout_position_x[float]
    absolute_layout_position_y: Absolute_layout_position_y[float]
    layout_x: Layout_x[float]
    layout_y: Layout_y[float]

    def __init__(self, **data):
        if 'layout_x' not in data:
            data['layout_x'] = JUNIOR_LAYOUT_SLOT_SIZE_X
        else:
            data['layout_x'] = float(data['layout_x'])
        if 'layout_y' not in data:
            data['layout_y'] = JUNIOR_LAYOUT_SLOT_SIZE_Y
        else:
            data['layout_y'] = float(data['layout_y'])
        super().__init__(**data)

    @property
    def layout_position(self):
        return tuple([
            list(self.absolute_layout_position_x)[0],
            list(self.absolute_layout_position_y)[0]])

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
            absolute_layout_position_x = 0.0
            absolute_layout_position_y = 0.0
        else:
            if layout_relation == "above":
                absolute_layout_position_x = layout_relative.layout_position[0]
                absolute_layout_position_y = layout_relative.layout_position[1] + list(layout_relative.layout_y)[0] + 20
            elif layout_relation == "right_to":
                absolute_layout_position_x = layout_relative.layout_position[0] + list(layout_relative.layout_x)[0] + 20
                absolute_layout_position_y = layout_relative.layout_position[1]
            else:
                raise ValueError
        return cls(
            absolute_layout_position_x=absolute_layout_position_x,
            absolute_layout_position_y=absolute_layout_position_y,
            layout_x=layout_x,
            layout_y=layout_y,
        )

class Layout(ObjectProperty):
    rdfs_isDefinedBy = JuniorOntology
