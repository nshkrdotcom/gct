"""Grouped split integrity checks."""

from __future__ import annotations

import pandas as pd


def assert_grouped_splits(frame: pd.DataFrame) -> None:
    memberships = frame.groupby("base_world_id", observed=True)["split"].nunique()
    leaked = memberships[memberships != 1]
    if not leaked.empty:
        raise ValueError(f"base worlds cross splits: {leaked.index.tolist()[:10]}")
    split_groups = {
        split: set(group["base_world_id"].astype(str))
        for split, group in frame.groupby("split", observed=True)
    }
    expected = {"train", "validation", "test"}
    if set(split_groups) != expected:
        raise ValueError(f"dataset splits must be exactly {sorted(expected)}")
    for left in expected:
        for right in expected:
            if left < right and split_groups[left] & split_groups[right]:
                raise ValueError(f"group leakage between {left} and {right}")
