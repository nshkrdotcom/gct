"""Small table helpers for paired held-out comparisons."""

from __future__ import annotations

import pandas as pd


def paired_condition_difference(
    frame: pd.DataFrame,
    *,
    left_condition: str,
    right_condition: str,
    value: str,
    output_name: str,
) -> pd.DataFrame:
    subset = frame[frame["coordinate_condition"].isin([left_condition, right_condition])]
    grouped = (
        subset.groupby(["base_world_id", "coordinate_condition"], observed=True)[value]
        .mean()
        .unstack("coordinate_condition")
        .dropna()
    )
    if left_condition not in grouped or right_condition not in grouped:
        return pd.DataFrame(columns=["base_world_id", output_name])
    grouped[output_name] = grouped[left_condition] - grouped[right_condition]
    return grouped.reset_index()[["base_world_id", output_name]]
