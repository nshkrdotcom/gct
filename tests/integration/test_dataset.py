from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gct.config import ExperimentConfig
from gct.data.generate import (
    build_dataset,
    generate_rows,
    logical_dataset_hash,
    validate_dataset_frame,
    validate_dataset_path,
)


@pytest.mark.integration
def test_dataset_is_deterministic_and_valid(ci_config: ExperimentConfig) -> None:
    first = generate_rows(ci_config)
    second = generate_rows(ci_config)
    assert logical_dataset_hash(first) == logical_dataset_hash(second)
    frame = pd.DataFrame([row.model_dump(mode="python") for row in first])
    result = validate_dataset_frame(frame, ci_config)
    assert result["valid"] is True
    assert result["unobservable_pairs"] > 0


@pytest.mark.integration
def test_no_split_or_hidden_coordinate_leakage(ci_config: ExperimentConfig) -> None:
    frame = pd.DataFrame([row.model_dump(mode="python") for row in generate_rows(ci_config)])
    assert frame.groupby("base_world_id")["split"].nunique().max() == 1
    unobservable = frame[
        (frame.coordinate_condition == "unobservable_coordinate")
        & (frame.transform_name == "pressure_shift")
    ]
    by_id = frame.set_index("sample_id")
    for row in unobservable.itertuples():
        source = by_id.loc[row.source_sample_id]
        assert source.prompt == row.prompt
        assert source.prompt_hash == row.prompt_hash
        assert "sensor_reading" not in json.loads(row.observable_json)
        assert source.oracle_target != row.oracle_target


@pytest.mark.integration
def test_parquet_round_trip_and_hash_validation(
    ci_config: ExperimentConfig, tmp_path: Path
) -> None:
    local = ci_config.model_copy(
        update={"project": ci_config.project.model_copy(update={"run_root": Path("runs")})}
    )
    run_dir = build_dataset(local, tmp_path)
    result = validate_dataset_path(run_dir)
    assert result["valid"] is True
    assert (run_dir / "dataset" / "samples.parquet").is_file()
