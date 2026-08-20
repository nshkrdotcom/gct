"""Regularized affine transport in a train-fit PCA space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from gct.operators.base import load_operator_payload, ridge_coefficients, save_operator
from gct.preprocessing.pca import PCASpace


@dataclass(slots=True)
class AffineRidgeTransport:
    n_components: int
    alpha: float = 1.0
    pca: PCASpace | None = None
    coefficient: np.ndarray | None = None
    intercept: np.ndarray | None = None
    model_type: str = "affine_ridge"

    def fit(self, source: np.ndarray, target: np.ndarray) -> Self:
        x = np.asarray(source, dtype=np.float32)
        y = np.asarray(target, dtype=np.float32)
        if x.shape != y.shape:
            raise ValueError("source and target shapes must match")
        self.pca = PCASpace.fit(np.vstack([x, y]), self.n_components)
        xr = self.pca.transform(x)
        yr = self.pca.transform(y)
        x_mean = xr.mean(axis=0)
        y_mean = yr.mean(axis=0)
        self.coefficient = ridge_coefficients(xr - x_mean, yr - y_mean, self.alpha)
        self.intercept = y_mean - x_mean @ self.coefficient
        return self

    def predict(self, source: np.ndarray) -> np.ndarray:
        if self.pca is None or self.coefficient is None or self.intercept is None:
            raise RuntimeError("affine transport is not fitted")
        reduced = self.pca.transform(source) @ self.coefficient + self.intercept
        return self.pca.inverse_transform(reduced)

    def capacity(self) -> dict[str, int | float]:
        effective = 0
        if self.coefficient is not None and self.intercept is not None:
            effective = self.coefficient.size + self.intercept.size
        return {"effective_parameters": int(effective), "pca_dimension": self.n_components}

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        if self.pca is None or self.coefficient is None or self.intercept is None:
            raise RuntimeError("cannot save an unfitted model")
        arrays = {
            "pca_mean": self.pca.mean,
            "pca_components": self.pca.components,
            "pca_variance": self.pca.explained_variance,
            "coefficient": self.coefficient,
            "intercept": self.intercept,
        }
        save_operator(
            path,
            self.model_type,
            arrays,
            {**(metadata or {}), "alpha": self.alpha, **self.capacity()},
        )

    @classmethod
    def load(cls, path: Path) -> AffineRidgeTransport:
        model_type, arrays, metadata = load_operator_payload(path)
        if model_type != "affine_ridge":
            raise ValueError("artifact is not affine ridge")
        model = cls(int(metadata["pca_dimension"]), float(metadata["alpha"]))
        model.pca = PCASpace(arrays["pca_mean"], arrays["pca_components"], arrays["pca_variance"])
        model.coefficient = arrays["coefficient"]
        model.intercept = arrays["intercept"]
        return model
