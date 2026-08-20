"""Cycle-consistency defect proxy."""

from __future__ import annotations

import numpy as np

from gct.metrics.distances import MetricSpace


def cycle_defect(
    start: np.ndarray, transported_endpoint: np.ndarray, space: MetricSpace
) -> dict[str, np.ndarray]:
    return space.distances(transported_endpoint, start)
