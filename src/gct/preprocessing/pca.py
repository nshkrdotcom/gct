"""SVD/PCA projection and whitening without a full covariance inverse."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True, slots=True)
class PCASpace:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    fit_split: str = "train"

    @classmethod
    def fit(cls, values: np.ndarray, n_components: int) -> PCASpace:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) < 2:
            raise ValueError("PCA requires a 2D matrix with at least two training rows")
        dimension = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
        if dimension < 1:
            raise ValueError("PCA retained dimension must be positive")
        solver = "randomized" if dimension < min(matrix.shape) else "full"
        estimator = PCA(
            n_components=dimension,
            svd_solver=solver,
            random_state=0 if solver == "randomized" else None,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in divide", category=RuntimeWarning
            )
            estimator.fit(matrix)
        return cls(
            estimator.mean_.astype(np.float32),
            estimator.components_.astype(np.float32),
            estimator.explained_variance_.astype(np.float32),
        )

    def transform(self, values: np.ndarray, *, whiten: bool = False) -> np.ndarray:
        reduced = (np.asarray(values, dtype=np.float32) - self.mean) @ self.components.T
        if whiten:
            reduced = reduced / np.sqrt(np.maximum(self.explained_variance, 1e-8))
        return np.asarray(reduced, dtype=np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32) @ self.components + self.mean
        return np.asarray(result, dtype=np.float32)
