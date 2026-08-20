"""Representation-dependent matching/descent proxy."""

from __future__ import annotations

import numpy as np

from gct.metrics.distances import MetricSpace


def matching_proxy(
    local_left: np.ndarray, local_right: np.ndarray, space: MetricSpace
) -> dict[str, np.ndarray]:
    return space.distances(local_left, local_right)
