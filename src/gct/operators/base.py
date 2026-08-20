"""Common transport interface and safe tensor serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


class TransportModel(Protocol):
    model_type: str

    def fit(self, source: np.ndarray, target: np.ndarray) -> Self: ...

    def predict(self, source: np.ndarray) -> np.ndarray: ...

    def capacity(self) -> dict[str, int | float]: ...


def ridge_coefficients(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape[0] <= x.shape[1]:
        dual = np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y)
        return (x.T @ dual).astype(np.float32)
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y).astype(np.float32)


def save_operator(
    path: Path, model_type: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()}
    if not tensors:
        tensors = {"_empty": torch.zeros(1, dtype=torch.float32)}
    string_metadata = {
        "model_type": model_type,
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }
    save_file(tensors, path, metadata=string_metadata)


def load_operator_payload(path: Path) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        model_type = metadata.get("model_type")
        if model_type is None:
            raise ValueError(f"operator artifact lacks model_type: {path}")
        for key in handle.keys():
            if key != "_empty":
                arrays[key] = handle.get_tensor(key).numpy()
        raw_metadata = json.loads(metadata.get("metadata_json", "{}"))
    return model_type, arrays, raw_metadata
