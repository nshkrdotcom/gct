from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gct.analysis.cross_model import (
    ENDPOINT_EFFECT_DIRECTIONS,
    join_group_effects,
    paired_grouped_mean_difference,
    paired_model_difference,
    primary_endpoint_effect,
    stable_id_join,
    validate_endpoint_comparisons,
)


def _frame(order: list[int], values: list[float]) -> pd.DataFrame:
    source = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "base_world_id": ["g0", "g0", "g1"],
            "split": ["test", "test", "test"],
            "arm": ["explicit_coordinate"] * 3,
            "world_variant": ["primary"] * 3,
            "transform_name": ["pressure_shift"] * 3,
            "renderer_variant": ["prose"] * 3,
            "value": values,
        }
    )
    return source.iloc[order].reset_index(drop=True)


def test_cross_model_join_uses_stable_ids_not_row_order() -> None:
    baseline = _frame([0, 1, 2], [1.0, 2.0, 3.0])
    replication = _frame([2, 0, 1], [20.0, 30.0, 10.0])
    joined = stable_id_join(baseline, replication, ["value"])
    assert joined["sample_id"].tolist() == ["s0", "s1", "s2"]
    assert joined["value_baseline"].tolist() == [1.0, 2.0, 3.0]
    assert joined["value_replication"].tolist() == [20.0, 30.0, 10.0]


def test_cross_model_join_rejects_duplicate_or_changed_stable_metadata() -> None:
    baseline = _frame([0, 1, 2], [1.0, 2.0, 3.0])
    duplicate = pd.concat([baseline, baseline.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate stable IDs"):
        stable_id_join(baseline, duplicate, ["value"])
    changed = baseline.copy()
    changed.loc[0, "base_world_id"] = "wrong"
    with pytest.raises(ValueError, match="stable metadata"):
        stable_id_join(baseline, changed, ["value"])


def test_paired_bootstrap_resamples_whole_base_worlds() -> None:
    frame = pd.DataFrame(
        {
            "base_world_id": ["g0", "g0", "g1"],
            "baseline": [0.0, 0.0, 0.0],
            "replication": [1.0, 3.0, 10.0],
        }
    )
    result = paired_model_difference(frame, "baseline", "replication", 8, seed=17)
    rng = np.random.default_rng(17)
    expected = []
    pieces = {"g0": np.array([1.0, 3.0]), "g1": np.array([10.0])}
    for _ in range(8):
        groups = rng.choice(np.array(["g0", "g1"], dtype=object), size=2, replace=True)
        expected.append(float(np.concatenate([pieces[str(group)] for group in groups]).mean()))
    assert np.array_equal(result.draws, np.asarray(expected))
    assert result.groups == 2


def test_paired_grouped_difference_preserves_multirow_group_pairing() -> None:
    baseline = pd.DataFrame({"base_world_id": ["g0", "g0", "g1"], "value": [0.0, 0.0, 0.0]})
    replication = pd.DataFrame({"base_world_id": ["g0", "g0", "g1"], "value": [1.0, 3.0, 10.0]})
    result = paired_grouped_mean_difference(
        baseline, replication, "value", "value", replicates=8, seed=17
    )
    rng = np.random.default_rng(17)
    expected = []
    pieces = {"g0": np.array([1.0, 3.0]), "g1": np.array([10.0])}
    for _ in range(8):
        groups = rng.choice(np.array(["g0", "g1"], dtype=object), size=2, replace=True)
        expected.append(float(np.concatenate([pieces[str(group)] for group in groups]).mean()))
    assert result.estimate == pytest.approx(14 / 3)
    assert np.array_equal(result.draws, np.asarray(expected))


def test_group_effect_join_rejects_missing_worlds() -> None:
    baseline = pd.DataFrame({"base_world_id": ["g0", "g1"], "value": [1.0, 2.0]})
    replication = pd.DataFrame({"base_world_id": ["g0"], "value": [3.0]})
    with pytest.raises(ValueError, match="base-world sets differ"):
        join_group_effects(baseline, replication, "value")


def test_primary_endpoint_effect_uses_frozen_sign_conventions() -> None:
    assert primary_endpoint_effect("H1", {"estimate": 1.0, "ci_95": [0.5, 1.5]}) == (
        1.0,
        [0.5, 1.5],
    )
    assert primary_endpoint_effect(
        "H4",
        {"grouped_absolute_prediction_error_gain": {"estimate": -0.2, "ci_95": [-0.4, 0.1]}},
    ) == (-0.2, [-0.4, 0.1])
    assert primary_endpoint_effect("H8", {"status": "not_supported"}) == (None, None)


def test_cross_model_report_requires_all_endpoints_and_negative_control() -> None:
    comparisons = [
        {
            "hypothesis": f"H{index}",
            "baseline_status": "not_supported" if index != 6 else "control_pass",
            "replication_status": "not_supported" if index != 6 else "control_pass",
        }
        for index in range(1, 9)
    ]
    validate_endpoint_comparisons(comparisons)
    with pytest.raises(ValueError, match="H6"):
        validate_endpoint_comparisons([row for row in comparisons if row["hypothesis"] != "H6"])
    failed_control = [dict(row) for row in comparisons]
    failed_control[5]["replication_status"] = "not_supported"
    with pytest.raises(ValueError, match="negative control"):
        validate_endpoint_comparisons(failed_control)


def test_endpoint_effect_sign_conventions_match_v1() -> None:
    assert ENDPOINT_EFFECT_DIRECTIONS["H1"] == "negative_supports"
    assert ENDPOINT_EFFECT_DIRECTIONS["H2"] == "positive_supports"
    assert ENDPOINT_EFFECT_DIRECTIONS["H3"] == "positive_supports"
    assert ENDPOINT_EFFECT_DIRECTIONS["H4"] == "positive_supports"
    assert ENDPOINT_EFFECT_DIRECTIONS["H7"] == "positive_supports"
