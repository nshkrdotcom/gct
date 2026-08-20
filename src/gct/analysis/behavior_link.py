"""Held-out behavior prediction with trivial-confound baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gct.probes.hidden_coordinate import regression_metrics


def evaluate_behavior_link(
    transport: pd.DataFrame, behavior: pd.DataFrame
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    defect = transport[
        (transport["model_type"] == "low_rank_residual") & (transport["metric"] == "whitened_l2")
    ][["sample_id", "raw_defect"]]
    merged = behavior.merge(defect, on="sample_id", how="inner").dropna(
        subset=["absolute_oracle_error"]
    )
    confounds = [
        "character_count",
        "character_count_difference",
        "token_count",
        "token_count_difference",
        "activation_norm",
        "oracle_delta_magnitude",
    ]
    feature_sets = {
        "confounds_only": confounds,
        "defect_only": ["raw_defect"],
        "confounds_plus_defect": [*confounds, "raw_defect"],
    }
    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]
    if len(train) < 3 or len(test) < 2:
        return [], pd.DataFrame()
    output: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for name, features in feature_sets.items():
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[features].to_numpy(float), train["absolute_oracle_error"].to_numpy(float))
        predictions = model.predict(test[features].to_numpy(float))
        r2, mae = regression_metrics(test["absolute_oracle_error"].to_numpy(float), predictions)
        output.append(
            {
                "outcome": "absolute_oracle_error",
                "feature_set": name,
                "fit_split": "train",
                "evaluation_split": "test",
                "r2": r2,
                "mae": mae,
                "train_rows": len(train),
                "test_rows": len(test),
                "features": features,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "sample_id": test["sample_id"].astype(str).to_numpy(),
                    "base_world_id": test["base_world_id"].astype(str).to_numpy(),
                    "feature_set": name,
                    "observed_absolute_oracle_error": test["absolute_oracle_error"].to_numpy(float),
                    "predicted_absolute_oracle_error": np.asarray(predictions, dtype=float),
                    "absolute_prediction_error": np.abs(
                        test["absolute_oracle_error"].to_numpy(float) - predictions
                    ),
                }
            )
        )
    return output, pd.concat(prediction_rows, ignore_index=True)
