"""Train-only per-dimension standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    fit_split: str = "train"

    @classmethod
    def fit(cls, values: np.ndarray) -> Standardizer:
        matrix = np.asarray(values, dtype=np.float32)
        return cls(matrix.mean(axis=0), np.maximum(matrix.std(axis=0), 1e-6))

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        return np.asarray(result, dtype=np.float32)
