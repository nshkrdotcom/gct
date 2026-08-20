"""Identity and mean-shift baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from gct.operators.base import load_operator_payload, save_operator


@dataclass(slots=True)
class IdentityTransport:
    model_type: str = "identity"

    def fit(self, source: np.ndarray, target: np.ndarray) -> Self:
        if np.asarray(source).shape != np.asarray(target).shape:
            raise ValueError("source and target shapes must match")
        return self

    def predict(self, source: np.ndarray) -> np.ndarray:
        return np.asarray(source, dtype=np.float32).copy()

    def capacity(self) -> dict[str, int | float]:
        return {"effective_parameters": 0}

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        save_operator(path, self.model_type, {}, {**(metadata or {}), **self.capacity()})

    @classmethod
    def load(cls, path: Path) -> IdentityTransport:
        model_type, _, _ = load_operator_payload(path)
        if model_type != "identity":
            raise ValueError("artifact is not an identity transport")
        return cls()


@dataclass(slots=True)
class MeanShiftTransport:
    shift: np.ndarray | None = None
    model_type: str = "mean_shift"

    def fit(self, source: np.ndarray, target: np.ndarray) -> Self:
        x = np.asarray(source, dtype=np.float32)
        y = np.asarray(target, dtype=np.float32)
        if x.shape != y.shape:
            raise ValueError("source and target shapes must match")
        self.shift = (y - x).mean(axis=0)
        return self

    def predict(self, source: np.ndarray) -> np.ndarray:
        if self.shift is None:
            raise RuntimeError("mean-shift transport is not fitted")
        result = np.asarray(source, dtype=np.float32) + self.shift
        return np.asarray(result, dtype=np.float32)

    def capacity(self) -> dict[str, int | float]:
        return {"effective_parameters": 0 if self.shift is None else int(self.shift.size)}

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        if self.shift is None:
            raise RuntimeError("cannot save an unfitted model")
        save_operator(
            path, self.model_type, {"shift": self.shift}, {**(metadata or {}), **self.capacity()}
        )

    @classmethod
    def load(cls, path: Path) -> MeanShiftTransport:
        model_type, arrays, _ = load_operator_payload(path)
        if model_type != "mean_shift":
            raise ValueError("artifact is not a mean-shift transport")
        return cls(arrays["shift"])
