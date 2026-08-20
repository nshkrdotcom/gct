"""Permutation-test summaries."""

from __future__ import annotations

import numpy as np


def empirical_upper_p(observed: float, null_values: np.ndarray) -> float:
    null = np.asarray(null_values, dtype=np.float64)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))
