"""Transport-defect normalization against identity."""

from __future__ import annotations

import numpy as np


def normalized_improvement(defect: np.ndarray, identity_defect: np.ndarray) -> np.ndarray:
    model = np.asarray(defect, dtype=np.float64)
    identity = np.asarray(identity_defect, dtype=np.float64)
    result = np.full_like(model, np.nan)
    usable = identity > 1e-12
    result[usable] = 1.0 - model[usable] / identity[usable]
    return result


def normalized_defect(defect: np.ndarray, identity_defect: np.ndarray) -> np.ndarray:
    model = np.asarray(defect, dtype=np.float64)
    identity = np.asarray(identity_defect, dtype=np.float64)
    result = np.full_like(model, np.nan)
    usable = identity > 1e-12
    result[usable] = model[usable] / identity[usable]
    return result
