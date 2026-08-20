"""Train/validation-only residual probe with grouped permutation nulls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gct.analysis.bootstrap import grouped_bootstrap_statistic
from gct.analysis.pairs import load_layer_dataset, paired_data
from gct.config import ExperimentConfig
from gct.operators.base import ridge_coefficients
from gct.operators.fit import group_slug
from gct.operators.low_rank import LowRankResidualTransport
from gct.operators.registry import load_transport
from gct.preprocessing.pca import PCASpace
from gct.provenance import update_run_manifest
from gct.storage.hashes import file_hash
from gct.storage.manifests import artifact_record, read_json, write_json_atomic


@dataclass(slots=True)
class ResidualProbe:
    pca: PCASpace
    alpha: float
    coefficient: np.ndarray | None = None
    intercept: float | None = None

    def fit(self, residuals: np.ndarray, labels: np.ndarray) -> ResidualProbe:
        x = self.pca.transform(residuals)
        y = np.asarray(labels, dtype=np.float32)
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        self.coefficient = ridge_coefficients(x - x_mean, (y - y_mean)[:, None], self.alpha)[:, 0]
        self.intercept = y_mean - float(x_mean @ self.coefficient)
        return self

    def predict(self, residuals: np.ndarray) -> np.ndarray:
        if self.coefficient is None or self.intercept is None:
            raise RuntimeError("residual probe is not fitted")
        result = self.pca.transform(residuals) @ self.coefficient + self.intercept
        return np.asarray(result, dtype=np.float32)


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    y = np.asarray(labels, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = (
        float("nan") if denominator <= 1e-12 else 1.0 - float(np.sum((y - pred) ** 2)) / denominator
    )
    mae = float(np.mean(np.abs(y - pred)))
    return r2, mae


def _slice_pca(pca: PCASpace, dimension: int) -> PCASpace:
    actual = min(dimension, pca.components.shape[0])
    return PCASpace(pca.mean, pca.components[:actual], pca.explained_variance[:actual])


def _group_permute(labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    group_values = {group: float(np.mean(labels[groups == group])) for group in unique}
    shuffled = rng.permutation([group_values[group] for group in unique])
    mapping = {group: value for group, value in zip(unique, shuffled, strict=True)}
    return np.array([mapping[group] for group in groups], dtype=np.float32)


def _residual_split(
    frame: pd.DataFrame,
    values: np.ndarray,
    group: str,
    split: str,
    transport: LowRankResidualTransport,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    pairs = paired_data(frame, values, group=group, split=split)
    residuals = pairs.target - transport.predict(pairs.source)
    labels = pairs.metadata["pressure"].to_numpy(dtype=np.float32)
    return pairs.metadata, residuals, labels


def fit_hidden_coordinate_probes(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    operator_manifest = read_json(run_dir / "operators" / "manifest.json")
    selection = read_json(run_dir / "operators" / "selection_frozen.json")
    layer = int(selection["primary_layer"])
    frame, values = load_layer_dataset(run_dir, layer)
    pressure_records = [
        record
        for record in operator_manifest["operators"]
        if record["model_type"] == "low_rank_residual"
        and str(record["operator_group"]).endswith("|pressure_shift")
    ]
    output_dir = run_dir / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.project.seed + 701)
    alpha = 1.0
    for group_index, record in enumerate(pressure_records):
        group = str(record["operator_group"])
        transport = load_transport(run_dir / str(record["path"]))
        if not isinstance(transport, LowRankResidualTransport):
            raise TypeError("pressure residual probe requires the selected low-rank transport")
        train_meta, train_residual, y_train = _residual_split(
            frame, values, group, "train", transport
        )
        val_meta, val_residual, y_val = _residual_split(
            frame, values, group, "validation", transport
        )
        test_meta, test_residual, y_test = _residual_split(frame, values, group, "test", transport)
        max_dimension = min(
            max(config.preprocessing.pca_dims), len(train_residual) - 1, train_residual.shape[1]
        )
        if max_dimension < 1:
            continue
        full_pca = PCASpace.fit(train_residual, max_dimension)
        dimensions = sorted(
            {min(value, full_pca.components.shape[0]) for value in config.preprocessing.pca_dims}
        )
        candidates: list[tuple[float, int, ResidualProbe]] = []
        for dimension in dimensions:
            probe = ResidualProbe(_slice_pca(full_pca, dimension), alpha).fit(
                train_residual, y_train
            )
            val_r2, _ = regression_metrics(y_val, probe.predict(val_residual))
            candidates.append((val_r2, dimension, probe))
            for split, residuals, labels in (
                ("train", train_residual, y_train),
                ("validation", val_residual, y_val),
                ("test", test_residual, y_test),
            ):
                r2, mae = regression_metrics(labels, probe.predict(residuals))
                curve_rows.append(
                    {
                        "operator_group": group,
                        "world_variant": group.split("|")[0],
                        "coordinate_condition": group.split("|")[1],
                        "split": split,
                        "pca_dimension": dimension,
                        "r2": r2,
                        "mae": mae,
                        "primary_selected_dimension": False,
                        "analysis_role": (
                            "predeclared_exploratory_dimension_curve"
                            if split == "test"
                            else "selection_support"
                        ),
                    }
                )
        _, selected_dimension, selected_probe = max(
            candidates, key=lambda item: (np.nan_to_num(item[0], nan=-np.inf), -item[1])
        )
        test_prediction = selected_probe.predict(test_residual)
        observed_r2, observed_mae = regression_metrics(y_test, test_prediction)
        bootstrap_frame = pd.DataFrame(
            {
                "base_world_id": test_meta["base_world_id"].astype(str).to_numpy(),
                "label": y_test,
                "prediction": test_prediction,
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
            config.project.seed + 800 + group_index * 2,
        )
        mae_bootstrap = grouped_bootstrap_statistic(
            bootstrap_frame,
            ["label", "prediction"],
            "base_world_id",
            lambda values: regression_metrics(
                values["label"].to_numpy(float), values["prediction"].to_numpy(float)
            )[1],
            config.statistics.bootstrap_replicates,
            config.project.seed + 801 + group_index * 2,
        )
        for row in curve_rows:
            if row["operator_group"] == group and row["pca_dimension"] == selected_dimension:
                row["primary_selected_dimension"] = True
        train_groups = train_meta["base_world_id"].astype(str).to_numpy()
        val_groups = val_meta["base_world_id"].astype(str).to_numpy()
        for replicate in range(config.statistics.permutation_replicates):
            shuffled_train = _group_permute(y_train, train_groups, rng)
            shuffled_val = _group_permute(y_val, val_groups, rng)
            permuted_candidates: list[tuple[float, int, ResidualProbe]] = []
            for dimension in dimensions:
                probe = ResidualProbe(_slice_pca(full_pca, dimension), alpha).fit(
                    train_residual, shuffled_train
                )
                val_r2, _ = regression_metrics(shuffled_val, probe.predict(val_residual))
                permuted_candidates.append((val_r2, dimension, probe))
            _, null_dimension, null_probe = max(
                permuted_candidates,
                key=lambda item: (np.nan_to_num(item[0], nan=-np.inf), -item[1]),
            )
            null_r2, null_mae = regression_metrics(y_test, null_probe.predict(test_residual))
            null_rows.append(
                {
                    "operator_group": group,
                    "replicate": replicate,
                    "selected_dimension": null_dimension,
                    "test_r2": null_r2,
                    "test_mae": null_mae,
                }
            )
        group_null = [row["test_r2"] for row in null_rows if row["operator_group"] == group]
        p_value = (1 + sum(value >= observed_r2 for value in group_null)) / (1 + len(group_null))
        path = output_dir / group_slug(group) / "pressure_probe.safetensors"
        if selected_probe.coefficient is None or selected_probe.intercept is None:
            raise RuntimeError("selected residual probe is not fitted")
        import torch
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "pca_mean": torch.from_numpy(selected_probe.pca.mean.astype(np.float32)),
                "pca_components": torch.from_numpy(
                    selected_probe.pca.components.astype(np.float32)
                ),
                "pca_variance": torch.from_numpy(
                    selected_probe.pca.explained_variance.astype(np.float32)
                ),
                "coefficient": torch.from_numpy(selected_probe.coefficient.astype(np.float32)),
                "intercept": torch.tensor([selected_probe.intercept], dtype=torch.float32),
            },
            path,
            metadata={
                "operator_group": group,
                "fit_split": "train",
                "dimension_selected_on": "validation",
                "selected_dimension": str(selected_dimension),
                "test_used_for_selection": "false",
            },
        )
        probe_records.append(
            {
                "operator_group": group,
                "world_variant": group.split("|")[0],
                "coordinate_condition": group.split("|")[1],
                "path": str(path.relative_to(run_dir)),
                "sha256": file_hash(path),
                "selected_dimension": selected_dimension,
                "alpha": alpha,
                "test_r2": observed_r2,
                "test_mae": observed_mae,
                "test_r2_ci_95": [r2_bootstrap.lower, r2_bootstrap.upper],
                "test_mae_ci_95": [mae_bootstrap.lower, mae_bootstrap.upper],
                "bootstrap_replicates": config.statistics.bootstrap_replicates,
                "permutation_p_value": p_value,
                "null_r2_95th_percentile": float(np.nanpercentile(group_null, 95)),
                "train_rows": len(train_meta),
                "validation_rows": len(val_meta),
                "test_rows": len(test_meta),
            }
        )
    curve = pd.DataFrame(curve_rows)
    nulls = pd.DataFrame(null_rows)
    curve_path = output_dir / "dimension_curve.parquet"
    null_path = output_dir / "permutation_nulls.parquet"
    curve.to_parquet(curve_path, index=False, compression="zstd")
    nulls.to_parquet(null_path, index=False, compression="zstd")
    manifest = {
        "schema_version": "gct-probes-v1",
        "status": "complete",
        "config_hash": config.config_hash,
        "primary_layer": layer,
        "selection_freeze_hash": selection["freeze_hash"],
        "probe_alpha": alpha,
        "permutation_replicates": config.statistics.permutation_replicates,
        "probes": probe_records,
        "dimension_curve": artifact_record(curve_path, run_dir, "probe_dimension_curve"),
        "permutation_nulls": artifact_record(null_path, run_dir, "probe_permutation_nulls"),
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    update_run_manifest(
        run_dir, probe_manifest_hash=file_hash(manifest_path), status="probes_complete"
    )
    return run_dir
