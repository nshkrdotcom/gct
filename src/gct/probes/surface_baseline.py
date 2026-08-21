"""Exploratory surface-feature baseline for the hidden-coordinate probe.

This analysis is EXPLORATORY and NON-CONFIRMATORY. It is not part of the frozen
`preregistration` mapping, it changes no H1-H8 status, and it never writes into a
confirmatory namespace.

Motivation. In the inferable arm the prompt prints the calibration reading
`R = 40 + 7 ln(P)` verbatim, and `P = exp((R - 40) / 7)` is an exact inverse. A
held-out residual R2 for `P` therefore has no interpretation until one knows what a
trivial reader of the prompt's own numerals achieves on the same rows. This module
supplies that missing baseline: it fits `P` from the numeric literals rendered into
the prompt, using the same estimator family, the same grouped train/validation/test
discipline, the same grouped permutation null, and the same grouped bootstrap as
`gct.probes.hidden_coordinate`, and it loads no activations and no model.

The comparison is deliberately linear on both sides. The residual probe is a ridge
fit over PCA components; this baseline is a ridge fit over parsed literals. Because
`R` inverts to `P` exactly, a nonlinear reader would do strictly better, so the
number reported here is a lower bound on trivial surface recoverability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gct.analysis.bootstrap import grouped_bootstrap_statistic
from gct.analysis.pairs import pair_group_name
from gct.config import load_config
from gct.operators.base import ridge_coefficients
from gct.operators.fit import group_slug
from gct.probes.hidden_coordinate import _group_permute, regression_metrics
from gct.storage.manifests import artifact_record, write_json_atomic

#: Rendered context labels mapped to one canonical feature name. The primary and
#: renamed worlds use one-to-one field symbols for the same quantities, so both
#: spellings collapse to the same feature. Parsing fails closed on anything else.
CONTEXT_FIELD_NAMES: dict[str, str] = {
    "Fluid": "entity",
    "Entity": "entity",
    "Concentration M": "concentration_literal",
    "Composition Y": "concentration_literal",
    "Pressure P": "explicit_coordinate_literal",
    "Control X": "explicit_coordinate_literal",
    "Calibration reading R": "calibration_reading_literal",
    "Proxy G": "calibration_reading_literal",
    "Nuisance Q": "nuisance_literal",
    "Nuisance W": "nuisance_literal",
}

#: Numeric features in a fixed order. Only the ones an arm actually renders are used.
NUMERIC_FEATURE_ORDER: tuple[str, ...] = (
    "explicit_coordinate_literal",
    "calibration_reading_literal",
    "concentration_literal",
    "nuisance_literal",
)

#: Predeclared ridge penalties. One is selected on validation rows only.
ALPHA_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

PRESSURE_SHIFT = "pressure_shift"


def parse_context_fields(prompt: str) -> dict[str, str]:
    """Extract the per-sample observable block's rendered fields.

    The prompt's `Context:` block is the only place a per-sample numeral appears; the
    law block above it is constant within an arm. Unknown labels raise rather than
    being dropped, so a future renderer change cannot silently empty this baseline.
    """
    lines = prompt.splitlines()
    try:
        start = lines.index("Context:")
    except ValueError:
        raise ValueError("prompt has no context block") from None
    body = lines[start + 1].strip().rstrip(".")
    if not body:
        raise ValueError("prompt has an empty context block")
    fields: dict[str, str] = {}
    for item in body.split(";"):
        label, separator, value = item.strip().partition(":")
        if not separator:
            raise ValueError(f"unrecognized context entry without a label: {item.strip()!r}")
        canonical = CONTEXT_FIELD_NAMES.get(label.strip())
        if canonical is None:
            raise ValueError(f"unrecognized context field label: {label.strip()!r}")
        fields[canonical] = value.strip()
    return fields


def rendered_numeric_names(prompts: list[str]) -> tuple[str, ...]:
    """Numeric features rendered by every prompt in the arm."""
    present: set[str] | None = None
    for prompt in prompts:
        keys = set(parse_context_fields(prompt)) & set(NUMERIC_FEATURE_ORDER)
        present = keys if present is None else (present & keys)
    return tuple(name for name in NUMERIC_FEATURE_ORDER if name in (present or set()))


def build_feature_matrix(
    prompts: list[str] | tuple[str, ...],
    *,
    numeric_names: tuple[str, ...],
    entity_values: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Numeric literals plus a train-vocabulary entity one-hot.

    `entity_values` comes from training rows only. A held-out entity (Cyrene in the
    full protocol) therefore leaves every indicator at zero rather than introducing a
    column that training never saw.
    """
    columns = tuple(numeric_names) + tuple(f"entity={value}" for value in entity_values)
    matrix = np.zeros((len(prompts), len(columns)), dtype=np.float64)
    for row, prompt in enumerate(prompts):
        fields = parse_context_fields(prompt)
        for index, name in enumerate(numeric_names):
            matrix[row, index] = float(fields[name])
        entity = fields.get("entity")
        for offset, value in enumerate(entity_values):
            if entity == value:
                matrix[row, len(numeric_names) + offset] = 1.0
    return matrix, columns


@dataclass(slots=True)
class SurfaceProbe:
    """Ridge over parsed literals, standardized on train rows only.

    The centering, the `ridge_coefficients` call, and the intercept reconstruction are
    the same as `gct.probes.hidden_coordinate.ResidualProbe`. The differences are the
    input matrix and the absence of a PCA step, which has no meaning over a handful of
    columns; standardization keeps one ridge penalty comparable across features whose
    natural scales differ by orders of magnitude.
    """

    alpha: float
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coefficient: np.ndarray | None = None
    intercept: float | None = None

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("surface probe is not fitted")
        centered = np.asarray(features, dtype=np.float64) - self.mean
        return np.asarray(centered / self.scale, dtype=np.float64)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> SurfaceProbe:
        raw = np.asarray(features, dtype=np.float64)
        self.mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        x = self._standardize(raw)
        y = np.asarray(labels, dtype=np.float64)
        y_mean = float(y.mean())
        self.coefficient = ridge_coefficients(x, (y - y_mean)[:, None], self.alpha)[:, 0]
        self.intercept = y_mean
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficient is None or self.intercept is None:
            raise RuntimeError("surface probe is not fitted")
        return np.asarray(
            self._standardize(features) @ self.coefficient + self.intercept, dtype=np.float32
        )


def _select_on_validation(
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, SurfaceProbe]:
    candidates: list[tuple[float, float, SurfaceProbe]] = []
    for alpha in ALPHA_GRID:
        probe = SurfaceProbe(alpha=alpha).fit(train_features, y_train)
        val_r2, _ = regression_metrics(y_val, probe.predict(val_features))
        candidates.append((val_r2, alpha, probe))
    val_r2, alpha, probe = max(
        candidates, key=lambda item: (np.nan_to_num(item[0], nan=-np.inf), -item[1])
    )
    return alpha, probe


def _pressure_shift_edges(frame: pd.DataFrame) -> pd.DataFrame:
    edges = frame[frame["source_sample_id"].notna()].copy()
    edges["operator_group"] = edges.apply(pair_group_name, axis=1)
    return edges[edges["transform_name"] == PRESSURE_SHIFT]


def fit_surface_baseline(run_dir: Path) -> Path:
    """Fit the exploratory surface baseline for one completed run.

    Reads `dataset/samples.parquet` and `config.yaml` only. No activations, no model.
    """
    run_dir = run_dir.resolve()
    config = load_config(run_dir / "config.yaml")
    frame = pd.read_parquet(run_dir / "dataset" / "samples.parquet")
    edges = _pressure_shift_edges(frame)
    if edges.empty:
        raise ValueError("run contains no pressure-shift pairs")
    output_dir = run_dir / "exploratory" / "surface_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.project.seed + 901)
    rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(sorted(edges["operator_group"].unique())):
        arm = edges[edges["operator_group"] == group]
        splits = {name: arm[arm["split"] == name] for name in ("train", "validation", "test")}
        if any(len(part) == 0 for part in splits.values()):
            raise ValueError(f"operator group {group} lacks a complete grouped split")
        train, validation, test = splits["train"], splits["validation"], splits["test"]
        numeric_names = rendered_numeric_names(train["prompt"].astype(str).tolist())
        entity_values = tuple(
            sorted(
                {
                    str(parse_context_fields(prompt).get("entity"))
                    for prompt in train["prompt"].astype(str)
                }
            )
        )
        matrices = {
            name: build_feature_matrix(
                part["prompt"].astype(str).tolist(),
                numeric_names=numeric_names,
                entity_values=entity_values,
            )[0]
            for name, part in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        }
        _, columns = build_feature_matrix(
            train["prompt"].astype(str).tolist()[:1],
            numeric_names=numeric_names,
            entity_values=entity_values,
        )
        y_train = train["pressure"].to_numpy(dtype=np.float32)
        y_val = validation["pressure"].to_numpy(dtype=np.float32)
        y_test = test["pressure"].to_numpy(dtype=np.float32)
        alpha, probe = _select_on_validation(
            matrices["train"], y_train, matrices["validation"], y_val
        )
        prediction = probe.predict(matrices["test"])
        test_r2, test_mae = regression_metrics(y_test, prediction)
        bootstrap_frame = pd.DataFrame(
            {
                "base_world_id": test["base_world_id"].astype(str).to_numpy(),
                "label": y_test,
                "prediction": prediction,
            }
        )
        r2_bootstrap = grouped_bootstrap_statistic(
            bootstrap_frame,
            ["label", "prediction"],
            "base_world_id",
            lambda values: regression_metrics(
                values["label"].to_numpy(float), values["prediction"].to_numpy(float)
            )[0],
            config.statistics.bootstrap_replicates,
            config.project.seed + 950 + group_index * 2,
        )
        mae_bootstrap = grouped_bootstrap_statistic(
            bootstrap_frame,
            ["label", "prediction"],
            "base_world_id",
            lambda values: regression_metrics(
                values["label"].to_numpy(float), values["prediction"].to_numpy(float)
            )[1],
            config.statistics.bootstrap_replicates,
            config.project.seed + 951 + group_index * 2,
        )
        train_groups = train["base_world_id"].astype(str).to_numpy()
        val_groups = validation["base_world_id"].astype(str).to_numpy()
        group_null: list[float] = []
        for replicate in range(config.statistics.permutation_replicates):
            shuffled_train = _group_permute(y_train, train_groups, rng)
            shuffled_val = _group_permute(y_val, val_groups, rng)
            _, null_probe = _select_on_validation(
                matrices["train"], shuffled_train, matrices["validation"], shuffled_val
            )
            null_r2, null_mae = regression_metrics(y_test, null_probe.predict(matrices["test"]))
            group_null.append(null_r2)
            null_rows.append(
                {
                    "operator_group": group,
                    "replicate": replicate,
                    "test_r2": null_r2,
                    "test_mae": null_mae,
                }
            )
        p_value = (1 + sum(value >= test_r2 for value in group_null)) / (1 + len(group_null))
        rows.append(
            {
                "operator_group": group,
                "group_slug": group_slug(group),
                "world_variant": group.split("|")[0],
                "coordinate_condition": group.split("|")[1],
                "feature_columns": ",".join(columns),
                "selected_alpha": alpha,
                "test_r2": test_r2,
                "test_mae": test_mae,
                "test_r2_ci_95": [r2_bootstrap.lower, r2_bootstrap.upper],
                "test_mae_ci_95": [mae_bootstrap.lower, mae_bootstrap.upper],
                "permutation_p_value": p_value,
                "null_r2_95th_percentile": float(np.nanpercentile(group_null, 95)),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "analysis_role": "exploratory_non_confirmatory",
            }
        )
    results = pd.DataFrame(rows)
    nulls = pd.DataFrame(null_rows)
    results_path = output_dir / "results.parquet"
    null_path = output_dir / "permutation_nulls.parquet"
    results.to_parquet(results_path, index=False, compression="zstd")
    nulls.to_parquet(null_path, index=False, compression="zstd")
    manifest = {
        "schema_version": "gct-exploratory-surface-baseline-v1",
        "status": "complete",
        "analysis_role": "exploratory_non_confirmatory",
        "confirmatory": False,
        "preregistered": False,
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "reads_activations": False,
        "reads_model": False,
        "estimator": "ridge over parsed prompt literals, standardized on train rows",
        "alpha_grid": list(ALPHA_GRID),
        "alpha_selected_on": "validation",
        "label": "pressure",
        "group_key": config.statistics.group_key,
        "bootstrap_replicates": config.statistics.bootstrap_replicates,
        "permutation_replicates": config.statistics.permutation_replicates,
        "context_field_labels": dict(sorted(CONTEXT_FIELD_NAMES.items())),
        "numeric_feature_order": list(NUMERIC_FEATURE_ORDER),
        "results": artifact_record(results_path, run_dir, "surface_baseline_results"),
        "permutation_nulls": artifact_record(null_path, run_dir, "surface_baseline_nulls"),
        "interpretation_note": (
            "Exploratory and non-confirmatory. It changes no preregistered endpoint status "
            "and is not a replacement for the residual probe. Because R = 40 + 7 ln(P) is "
            "exactly invertible, this linear read of the prompt's own numerals is a lower "
            "bound on trivial surface recoverability of P."
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return output_dir
