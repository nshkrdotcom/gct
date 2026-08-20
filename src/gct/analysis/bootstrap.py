"""Grouped bootstrap confidence intervals by base world."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    replicates: int
    groups: int
    draws: np.ndarray


def grouped_bootstrap_mean(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    clean = frame[[group_column, value_column]].dropna()
    groups = clean[group_column].astype(str).unique()
    if len(groups) < 2:
        raise ValueError("grouped bootstrap requires at least two groups")
    values = {
        group: clean.loc[clean[group_column].astype(str) == group, value_column].to_numpy(float)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pooled = np.concatenate([values[group] for group in sampled])
        draws[replicate] = float(np.mean(pooled))
    return BootstrapResult(
        estimate=float(clean[value_column].mean()),
        lower=float(np.percentile(draws, 2.5)),
        upper=float(np.percentile(draws, 97.5)),
        replicates=replicates,
        groups=len(groups),
        draws=draws,
    )


def grouped_bootstrap_statistic(
    frame: pd.DataFrame,
    value_columns: list[str],
    group_column: str,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """Bootstrap arbitrary row statistics while resampling whole groups."""
    clean = frame[[group_column, *value_columns]].dropna()
    groups = clean[group_column].astype(str).unique()
    if len(groups) < 2:
        raise ValueError("grouped bootstrap requires at least two groups")
    pieces = {
        group: clean.loc[clean[group_column].astype(str) == group, value_columns]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pooled = pd.concat([pieces[group] for group in sampled], ignore_index=True)
        draws[replicate] = statistic(pooled)
    return BootstrapResult(
        estimate=statistic(clean[value_columns]),
        lower=float(np.nanpercentile(draws, 2.5)),
        upper=float(np.nanpercentile(draws, 97.5)),
        replicates=replicates,
        groups=len(groups),
        draws=draws,
    )
