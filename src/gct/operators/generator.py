"""Continuous matrix-generator transport in a train-fit PCA space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from scipy.linalg import expm

from gct.operators.base import load_operator_payload, ridge_coefficients, save_operator
from gct.preprocessing.pca import PCASpace


@dataclass(slots=True)
class ContinuousGeneratorTransport:
    n_components: int
    regularization: float = 0.001
    pca: PCASpace | None = None
    generator: np.ndarray | None = None
    model_type: str = "continuous_generator"

    def fit_with_deltas(
        self, source: np.ndarray, target: np.ndarray, deltas: np.ndarray
    ) -> ContinuousGeneratorTransport:
        x = np.asarray(source, dtype=np.float32)
        y = np.asarray(target, dtype=np.float32)
        delta = np.asarray(deltas, dtype=np.float32).reshape(-1)
        if x.shape != y.shape or len(delta) != len(x):
            raise ValueError("source, target, and delta rows must align")
        if np.any(np.isclose(delta, 0)):
            raise ValueError("generator fitting requires nonzero deltas")
        self.pca = PCASpace.fit(np.vstack([x, y]), self.n_components)
        xr = self.pca.transform(x)
        yr = self.pca.transform(y)
        derivative = (yr - xr) / delta[:, None]
        self.generator = ridge_coefficients(xr, derivative, self.regularization)
        return self

    def fit(self, source: np.ndarray, target: np.ndarray) -> Self:
        raise TypeError("continuous generator requires fit_with_deltas")

    def predict_delta(self, source: np.ndarray, delta: float | np.ndarray) -> np.ndarray:
        if self.pca is None or self.generator is None:
            raise RuntimeError("continuous generator is not fitted")
        xr = self.pca.transform(source)
        values = np.asarray(delta, dtype=np.float32)
        if values.ndim == 0:
            predicted = xr @ expm(float(values) * self.generator)
        else:
            if len(values) != len(xr):
                raise ValueError("one delta is required per source row")
            predicted = np.vstack(
                [
                    row @ expm(float(amount) * self.generator)
                    for row, amount in zip(xr, values, strict=True)
                ]
            )
        return self.pca.inverse_transform(predicted)

    def predict(self, source: np.ndarray) -> np.ndarray:
        raise TypeError("continuous generator prediction requires an explicit delta")

    def reduced_route(self, source: np.ndarray, deltas: tuple[float, ...]) -> np.ndarray:
        if self.pca is None or self.generator is None:
            raise RuntimeError("continuous generator is not fitted")
        reduced = self.pca.transform(source)
        for delta in deltas:
            reduced = reduced @ expm(delta * self.generator)
        return reduced

    def capacity(self) -> dict[str, int | float]:
        effective = 0 if self.generator is None else self.generator.size
        return {"effective_parameters": int(effective), "pca_dimension": self.n_components}

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        if self.pca is None or self.generator is None:
            raise RuntimeError("cannot save an unfitted model")
        arrays = {
            "pca_mean": self.pca.mean,
            "pca_components": self.pca.components,
            "pca_variance": self.pca.explained_variance,
            "generator": self.generator,
        }
        save_operator(
            path,
            self.model_type,
            arrays,
            {
                **(metadata or {}),
                "regularization": self.regularization,
                **self.capacity(),
            },
        )

    @classmethod
    def load(cls, path: Path) -> ContinuousGeneratorTransport:
        model_type, arrays, metadata = load_operator_payload(path)
        if model_type != "continuous_generator":
            raise ValueError("artifact is not a continuous generator")
        model = cls(int(metadata["pca_dimension"]), float(metadata["regularization"]))
        model.pca = PCASpace(arrays["pca_mean"], arrays["pca_components"], arrays["pca_variance"])
        model.generator = arrays["generator"]
        return model
