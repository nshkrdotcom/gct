"""Raw cosine, train-standardized L2, and train-PCA-whitened L2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gct.preprocessing.pca import PCASpace
from gct.preprocessing.scaling import Standardizer


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float32)
    y = np.asarray(right, dtype=np.float32)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    similarity = np.divide(
        np.sum(x * y, axis=1),
        denominator,
        out=np.zeros(len(x), dtype=np.float32),
        where=denominator > 1e-12,
    )
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def root_mean_square_l2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2, axis=1))


@dataclass(frozen=True, slots=True)
class MetricSpace:
    standardizer: Standardizer
    pca: PCASpace
    fit_split: str = "train"

    @classmethod
    def fit(cls, train_values: np.ndarray, pca_dimension: int) -> MetricSpace:
        return cls(Standardizer.fit(train_values), PCASpace.fit(train_values, pca_dimension))

    def distances(self, predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "cosine": cosine_distance(predicted, target),
            "standardized_l2": root_mean_square_l2(
                self.standardizer.transform(predicted), self.standardizer.transform(target)
            ),
            "whitened_l2": root_mean_square_l2(
                self.pca.transform(predicted, whiten=True),
                self.pca.transform(target, whiten=True),
            ),
        }
