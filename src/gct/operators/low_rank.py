"""Low-rank residual transport without a materialized d-by-d product."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from gct.operators.base import load_operator_payload, ridge_coefficients, save_operator


@dataclass(slots=True)
class LowRankResidualTransport:
    rank: int
    alpha: float = 1.0
    a: np.ndarray | None = None
    b: np.ndarray | None = None
    bias: np.ndarray | None = None
    model_type: str = "low_rank_residual"

    def fit(self, source: np.ndarray, target: np.ndarray) -> Self:
        x = np.asarray(source, dtype=np.float32)
        y = np.asarray(target, dtype=np.float32)
        if x.shape != y.shape:
            raise ValueError("source and target shapes must match")
        residual = y - x
        x_mean = x.mean(axis=0)
        residual_mean = residual.mean(axis=0)
        x_centered = x - x_mean
        residual_centered = residual - residual_mean
        max_rank = min(self.rank, residual_centered.shape[0], residual_centered.shape[1])
        if max_rank < 1:
            raise ValueError("low-rank model requires at least one paired row")
        _, _, vt = np.linalg.svd(residual_centered.astype(np.float64), full_matrices=False)
        self.a = vt[:max_rank].T.astype(np.float32)
        scores = residual_centered @ self.a
        coefficient = ridge_coefficients(x_centered, scores, self.alpha)
        self.b = coefficient.T
        self.bias = residual_mean - (x_mean @ self.b.T) @ self.a.T
        return self

    def predict(self, source: np.ndarray) -> np.ndarray:
        if self.a is None or self.b is None or self.bias is None:
            raise RuntimeError("low-rank transport is not fitted")
        z = np.asarray(source, dtype=np.float32)
        # This is deliberately factored: no d-by-d A@B matrix is constructed.
        result = z + (z @ self.b.T) @ self.a.T + self.bias
        return np.asarray(result, dtype=np.float32)

    def capacity(self) -> dict[str, int | float]:
        effective = 0
        actual_rank = 0
        if self.a is not None and self.b is not None and self.bias is not None:
            effective = self.a.size + self.b.size + self.bias.size
            actual_rank = self.a.shape[1]
        return {
            "effective_parameters": int(effective),
            "requested_rank": self.rank,
            "actual_rank": actual_rank,
        }

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        if self.a is None or self.b is None or self.bias is None:
            raise RuntimeError("cannot save an unfitted model")
        save_operator(
            path,
            self.model_type,
            {"a": self.a, "b": self.b, "bias": self.bias},
            {**(metadata or {}), "alpha": self.alpha, **self.capacity()},
        )

    @classmethod
    def load(cls, path: Path) -> LowRankResidualTransport:
        model_type, arrays, metadata = load_operator_payload(path)
        if model_type != "low_rank_residual":
            raise ValueError("artifact is not low-rank residual transport")
        model = cls(int(metadata["requested_rank"]), float(metadata["alpha"]))
        model.a = arrays["a"]
        model.b = arrays["b"]
        model.bias = arrays["bias"]
        return model
