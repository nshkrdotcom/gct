"""Stable-ID joins and paired base-world statistics for cross-family analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from gct.analysis.bootstrap import BootstrapResult, grouped_bootstrap_mean

STABLE_METADATA_COLUMNS = (
    "base_world_id",
    "split",
    "arm",
    "world_variant",
    "transform_name",
    "renderer_variant",
)

ENDPOINT_EFFECT_DIRECTIONS = {
    "H1": "negative_supports",
    "H2": "positive_supports",
    "H3": "positive_supports",
    "H4": "positive_supports",
    "H5": "positive_supports",
    "H6": "negative_control",
    "H7": "positive_supports",
    "H8": "joint_gate",
}


def stable_id_join(
    baseline: pd.DataFrame,
    replication: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    required = {"sample_id", *STABLE_METADATA_COLUMNS, *value_columns}
    for label, frame in (("baseline", baseline), ("replication", replication)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} frame lacks stable join columns: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{label} frame contains duplicate stable IDs")
    baseline_ids = set(baseline["sample_id"].astype(str))
    replication_ids = set(replication["sample_id"].astype(str))
    if baseline_ids != replication_ids:
        raise ValueError("baseline and replication stable ID sets differ")
    columns = ["sample_id", *STABLE_METADATA_COLUMNS, *value_columns]
    joined = baseline[columns].merge(
        replication[columns],
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_replication"),
        sort=True,
    )
    for column in STABLE_METADATA_COLUMNS:
        baseline_column = f"{column}_baseline"
        replication_column = f"{column}_replication"
        if not joined[baseline_column].astype(str).equals(joined[replication_column].astype(str)):
            raise ValueError(f"stable metadata changed between models: {column}")
        joined[column] = joined.pop(baseline_column)
        joined = joined.drop(columns=[replication_column])
    return joined.sort_values("sample_id").reset_index(drop=True)


def paired_model_difference(
    frame: pd.DataFrame,
    baseline_column: str,
    replication_column: str,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    required = {"base_world_id", baseline_column, replication_column}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"paired comparison lacks columns: {sorted(required - set(frame.columns))}"
        )
    effects = frame[["base_world_id", baseline_column, replication_column]].copy()
    effects["replication_minus_baseline"] = effects[replication_column].astype(float) - effects[
        baseline_column
    ].astype(float)
    return grouped_bootstrap_mean(
        effects,
        "replication_minus_baseline",
        "base_world_id",
        replicates,
        seed,
    )


def paired_grouped_mean_difference(
    baseline: pd.DataFrame,
    replication: pd.DataFrame,
    baseline_column: str,
    replication_column: str | None,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """Difference of pooled means while pairing whole, possibly multi-row groups."""
    replication_column = replication_column or baseline_column
    left = baseline[["base_world_id", baseline_column]].dropna().copy()
    right = replication[["base_world_id", replication_column]].dropna().copy()
    left["base_world_id"] = left["base_world_id"].astype(str)
    right["base_world_id"] = right["base_world_id"].astype(str)
    groups = np.asarray(sorted(left["base_world_id"].unique()), dtype=object)
    if len(groups) < 2:
        raise ValueError("paired grouped bootstrap requires at least two base worlds")
    if set(groups) != set(right["base_world_id"].unique()):
        raise ValueError("baseline and replication base-world sets differ")
    left_values = {
        group: left.loc[left["base_world_id"] == group, baseline_column].to_numpy(float)
        for group in groups
    }
    right_values = {
        group: right.loc[right["base_world_id"] == group, replication_column].to_numpy(float)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        left_pooled = np.concatenate([left_values[str(group)] for group in sampled])
        right_pooled = np.concatenate([right_values[str(group)] for group in sampled])
        draws[replicate] = float(right_pooled.mean() - left_pooled.mean())
    estimate = float(right[replication_column].astype(float).mean()) - float(
        left[baseline_column].astype(float).mean()
    )
    return BootstrapResult(
        estimate=estimate,
        lower=float(np.percentile(draws, 2.5)),
        upper=float(np.percentile(draws, 97.5)),
        replicates=replicates,
        groups=len(groups),
        draws=draws,
    )


def join_group_effects(
    baseline: pd.DataFrame,
    replication: pd.DataFrame,
    baseline_column: str,
    replication_column: str | None = None,
) -> pd.DataFrame:
    """Join additive endpoint effects by base world with strict set validation."""
    replication_column = replication_column or baseline_column
    for label, frame, column in (
        ("baseline", baseline, baseline_column),
        ("replication", replication, replication_column),
    ):
        missing = {"base_world_id", column} - set(frame.columns)
        if missing:
            raise ValueError(f"{label} group effect lacks columns: {sorted(missing)}")
        if frame["base_world_id"].duplicated().any():
            raise ValueError(f"{label} group effect contains duplicate base worlds")
    baseline_ids = set(baseline["base_world_id"].astype(str))
    replication_ids = set(replication["base_world_id"].astype(str))
    if baseline_ids != replication_ids:
        raise ValueError("baseline and replication base-world sets differ")
    left = baseline[["base_world_id", baseline_column]].rename(
        columns={baseline_column: "effect_baseline"}
    )
    right = replication[["base_world_id", replication_column]].rename(
        columns={replication_column: "effect_replication"}
    )
    joined = left.merge(right, on="base_world_id", validate="one_to_one", sort=True)
    joined["replication_minus_baseline"] = joined["effect_replication"].astype(float) - joined[
        "effect_baseline"
    ].astype(float)
    return joined.sort_values("base_world_id").reset_index(drop=True)


def primary_endpoint_effect(
    name: str, record: dict[str, Any]
) -> tuple[float | None, list[float] | None]:
    """Return the frozen primary scalar and CI used in the cross-model table."""
    if name in {"H1", "H2", "H3", "H7"}:
        return float(record["estimate"]), [float(value) for value in record["ci_95"]]
    if name == "H4":
        grouped = record["grouped_absolute_prediction_error_gain"]
        return float(grouped["estimate"]), [float(value) for value in grouped["ci_95"]]
    if name in {"H5", "H6"}:
        return float(record["test_r2"]), [float(value) for value in record["test_r2_ci_95"]]
    if name == "H8":
        return None, None
    raise ValueError(f"unknown hypothesis: {name}")


def validate_endpoint_comparisons(comparisons: Sequence[dict[str, object]]) -> None:
    names = [str(record.get("hypothesis")) for record in comparisons]
    expected = [f"H{index}" for index in range(1, 9)]
    if sorted(names) != expected or len(names) != len(set(names)):
        missing = sorted(set(expected) - set(names))
        raise ValueError(f"cross-model report must contain H1-H8 exactly once; missing {missing}")
    h6 = next(record for record in comparisons if record["hypothesis"] == "H6")
    if (
        h6.get("baseline_status") != "control_pass"
        or h6.get("replication_status") != "control_pass"
    ):
        raise ValueError(
            "H6 negative control must pass in both runs before cross-model interpretation"
        )
