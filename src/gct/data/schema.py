"""Typed row contracts for the columnar experiment dataset."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Split = Literal["train", "validation", "test"]


class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    base_world_id: str
    split: Split
    arm: str
    coordinate_condition: str
    world_variant: Literal["primary", "renamed"]
    world_version: str
    fluid: str
    pressure: float
    concentration: float
    irrelevant_q: float | None
    observable_json: str
    oracle_target: float
    prompt: str
    prompt_hash: str
    renderer_variant: str
    transform_id: str
    transform_family: Literal["identity", "nuisance", "substantive", "control"]
    transform_name: str
    transform_parameters_json: str
    oracle_identity: bool
    inverse_transform_id: str | None
    source_sample_id: str | None
    composition_json: str
    square_id: str | None
    cycle_id: str | None
    char_count: int
    secondary_entity_holdout: bool


DATASET_COLUMNS = list(DatasetRow.model_fields)
