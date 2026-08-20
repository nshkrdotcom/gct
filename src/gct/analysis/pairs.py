"""Activation/dataset alignment and paired-edge selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gct.storage.activations import load_activation_layer


@dataclass(frozen=True, slots=True)
class PairedData:
    metadata: pd.DataFrame
    source: np.ndarray
    target: np.ndarray


def load_layer_dataset(run_dir: Path, layer: int) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = pd.read_parquet(run_dir / "dataset" / "samples.parquet")
    index, tensor = load_activation_layer(run_dir, layer)
    if dataset["sample_id"].astype(str).tolist() != index["sample_id"].astype(str).tolist():
        raise RuntimeError("dataset and activation index are not identically ordered")
    merged = dataset.copy()
    merged["token_count"] = index["token_count"].to_numpy()
    merged["activation_norm"] = np.linalg.norm(tensor.float().numpy(), axis=1)
    return merged, tensor.float().numpy()


def pair_group_name(row: pd.Series) -> str:
    world_variant = str(row["world_variant"])
    condition = str(row["coordinate_condition"])
    transform = str(row["transform_name"])
    return f"{world_variant}|{condition}|{transform}"


def paired_data(
    frame: pd.DataFrame,
    activations: np.ndarray,
    *,
    group: str | None = None,
    split: str | None = None,
) -> PairedData:
    edges = frame[frame["source_sample_id"].notna()].copy()
    edges["operator_group"] = edges.apply(pair_group_name, axis=1)
    if group is not None:
        edges = edges[edges["operator_group"] == group]
    if split is not None:
        edges = edges[edges["split"] == split]
    position = {str(sample_id): index for index, sample_id in enumerate(frame["sample_id"])}
    target_positions = [position[str(sample)] for sample in edges["sample_id"]]
    source_positions = [position[str(sample)] for sample in edges["source_sample_id"]]
    return PairedData(
        edges.reset_index(drop=True),
        activations[source_positions].astype(np.float32),
        activations[target_positions].astype(np.float32),
    )


def parsed_deltas(metadata: pd.DataFrame, key: str = "delta") -> np.ndarray:
    return np.array(
        [float(json.loads(value)[key]) for value in metadata["transform_parameters_json"]],
        dtype=np.float32,
    )
