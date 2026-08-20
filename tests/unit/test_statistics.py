from __future__ import annotations

import numpy as np
import pandas as pd

from gct.analysis.bootstrap import grouped_bootstrap_mean, grouped_bootstrap_statistic
from gct.analysis.multiple_testing import benjamini_hochberg
from gct.analysis.tables import paired_condition_difference


def test_grouped_bootstrap_is_deterministic() -> None:
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1.0, 1.0, 3.0, 3.0]})
    first = grouped_bootstrap_mean(frame, "value", "group", 100, 4)
    second = grouped_bootstrap_mean(frame, "value", "group", 100, 4)
    assert first.estimate == 2.0
    assert np.array_equal(first.draws, second.draws)


def test_benjamini_hochberg_known_values() -> None:
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_grouped_bootstrap_arbitrary_statistic_is_deterministic() -> None:
    frame = pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "truth": [1.0, 2.0, 3.0, 4.0], "pred": [1, 1, 3, 3]}
    )

    def statistic(data: pd.DataFrame) -> float:
        return float(np.mean(np.abs(data["truth"] - data["pred"])))

    first = grouped_bootstrap_statistic(frame, ["truth", "pred"], "group", statistic, 50, 9)
    second = grouped_bootstrap_statistic(frame, ["truth", "pred"], "group", statistic, 50, 9)
    assert first.estimate == 0.5
    assert np.array_equal(first.draws, second.draws)


def test_missing_paired_condition_is_explicitly_empty() -> None:
    frame = pd.DataFrame(
        {
            "base_world_id": ["a"],
            "coordinate_condition": ["explicit_coordinate"],
            "value": [1.0],
        }
    )
    result = paired_condition_difference(
        frame,
        left_condition="inferable_unnamed_coordinate",
        right_condition="explicit_coordinate",
        value="value",
        output_name="gain",
    )
    assert result.empty
    assert list(result) == ["base_world_id", "gain"]
