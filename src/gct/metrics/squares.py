"""Commuting-square defect proxy (not a categorical pullback claim)."""

from __future__ import annotations

import numpy as np

from gct.metrics.distances import MetricSpace


def commuting_square_defect(
    route_one: np.ndarray, route_two: np.ndarray, space: MetricSpace
) -> dict[str, np.ndarray]:
    return space.distances(route_one, route_two)
