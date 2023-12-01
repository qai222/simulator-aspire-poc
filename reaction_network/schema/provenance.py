from __future__ import annotations

from typing import Any

from pydantic import BaseModel, create_model


def get_provenance_model(pydantic_model: type[BaseModel], provenance_model_name: str):
    """
    given a base pydantic model, this function creates a new pydantic model with an additional string field named

    <field_name>__provenance

    for each field in the base model
    """
    old_model_keys = sorted(pydantic_model.model_fields.keys())
    provenance_fields = {k + '__provenance': (Any, None) for k in old_model_keys}

    new_model = create_model(
        provenance_model_name, **provenance_fields,
        __base__=pydantic_model,
    )
    return new_model
