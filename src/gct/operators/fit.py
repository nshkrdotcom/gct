"""Validation-only layer/rank/PCA selection and frozen operator fitting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from gct.analysis.pairs import load_layer_dataset, paired_data, parsed_deltas
from gct.config import ExperimentConfig
from gct.metrics.distances import MetricSpace
from gct.operators.affine import AffineRidgeTransport
from gct.operators.baselines import IdentityTransport, MeanShiftTransport
from gct.operators.generator import ContinuousGeneratorTransport
from gct.operators.low_rank import LowRankResidualTransport
from gct.provenance import update_run_manifest
from gct.storage.hashes import canonical_hash, file_hash
from gct.storage.manifests import artifact_record, read_json, write_json_atomic

PRIMARY_OPERATOR_GROUP = "primary|explicit_coordinate|pressure_shift"


def group_slug(group: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-")


def _available_layers(run_dir: Path) -> list[int]:
    manifest = read_json(run_dir / "activations" / "manifest.json")
    first = manifest["shards"][0]
    return [int(value) for value in first["layer_numbers"]]


def _metric_space_arrays(space: MetricSpace) -> dict[str, torch.Tensor]:
    return {
        "standard_mean": torch.from_numpy(space.standardizer.mean.astype(np.float32)),
        "standard_scale": torch.from_numpy(space.standardizer.scale.astype(np.float32)),
        "pca_mean": torch.from_numpy(space.pca.mean.astype(np.float32)),
        "pca_components": torch.from_numpy(space.pca.components.astype(np.float32)),
        "pca_variance": torch.from_numpy(space.pca.explained_variance.astype(np.float32)),
    }


def _fit_metric_space(frame: pd.DataFrame, values: np.ndarray, dimension: int) -> MetricSpace:
    train = values[frame["split"].to_numpy() == "train"]
    return MetricSpace.fit(train, dimension)


def _candidate_scan(config: ExperimentConfig, run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer in _available_layers(run_dir):
        frame, values = load_layer_dataset(run_dir, layer)
        train = paired_data(frame, values, group=PRIMARY_OPERATOR_GROUP, split="train")
        validation = paired_data(frame, values, group=PRIMARY_OPERATOR_GROUP, split="validation")
        if len(train.metadata) < 2 or len(validation.metadata) < 1:
            raise ValueError("primary operator group lacks train/validation pairs")
        for dimension in config.preprocessing.pca_dims:
            space = _fit_metric_space(frame, values, dimension)
            identity = space.distances(validation.source, validation.target)
            for rank in config.operators.low_rank.ranks:
                model = LowRankResidualTransport(
                    rank=rank, alpha=config.operators.low_rank.ridge_alpha
                ).fit(train.source, train.target)
                predicted = model.predict(validation.source)
                distances = space.distances(predicted, validation.target)
                rows.append(
                    {
                        "fit_split": "train",
                        "selection_split": "validation",
                        "operator_group": PRIMARY_OPERATOR_GROUP,
                        "layer": layer,
                        "pca_dimension_requested": dimension,
                        "pca_dimension_actual": space.pca.components.shape[0],
                        "rank_requested": rank,
                        "rank_actual": model.a.shape[1] if model.a is not None else 0,
                        "whitened_l2": float(np.mean(distances["whitened_l2"])),
                        "standardized_l2": float(np.mean(distances["standardized_l2"])),
                        "cosine": float(np.mean(distances["cosine"])),
                        "identity_whitened_l2": float(np.mean(identity["whitened_l2"])),
                    }
                )
    return pd.DataFrame(rows)


def _select_candidate(scan: pd.DataFrame) -> dict[str, int | float | str]:
    ranked = scan.sort_values(
        ["whitened_l2", "rank_actual", "pca_dimension_actual", "layer"], kind="stable"
    )
    best = ranked.iloc[0]
    return {
        "primary_layer": int(best["layer"]),
        "pca_dimension": int(best["pca_dimension_actual"]),
        "rank": int(best["rank_requested"]),
        "criterion": "minimum validation whitened_l2; ties prefer lower rank/dimension/layer",
        "selection_split": "validation",
        "fit_split": "train",
        "validation_whitened_l2": float(best["whitened_l2"]),
    }


def fit_transport_operators(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    activation_manifest = read_json(run_dir / "activations" / "manifest.json")
    if activation_manifest.get("status") != "complete":
        raise ValueError("complete activation artifacts are required")
    output_dir = run_dir / "operators"
    output_dir.mkdir(parents=True, exist_ok=True)
    scan = _candidate_scan(config, run_dir)
    scan_path = output_dir / "validation_selection.parquet"
    scan.to_parquet(scan_path, index=False, compression="zstd")
    selection = _select_candidate(scan)
    layer = int(selection["primary_layer"])
    pca_dimension = int(selection["pca_dimension"])
    frame, values = load_layer_dataset(run_dir, layer)
    space = _fit_metric_space(frame, values, pca_dimension)
    preprocessing_dir = run_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    space_path = preprocessing_dir / "metric_space.safetensors"
    save_file(
        _metric_space_arrays(space),
        space_path,
        metadata={
            "fit_split": "train",
            "selected_on": "validation",
            "primary_layer": str(layer),
            "pca_dimension": str(space.pca.components.shape[0]),
            "config_hash": config.config_hash,
        },
    )
    edges = frame[frame["source_sample_id"].notna()].copy()
    edges["operator_group"] = (
        edges["world_variant"].astype(str)
        + "|"
        + edges["coordinate_condition"].astype(str)
        + "|"
        + edges["transform_name"].astype(str)
    )
    supported = {
        "pressure_shift",
        "concentration_shift",
        "fluid_swap",
        "nuisance_rewrite",
        "nuisance_inverse",
    }
    groups = sorted(
        group for group in edges["operator_group"].unique() if group.rsplit("|", 1)[-1] in supported
    )
    records: list[dict[str, Any]] = []
    for group in groups:
        train = paired_data(frame, values, group=group, split="train")
        validation = paired_data(frame, values, group=group, split="validation")
        if len(train.metadata) < 2 or len(validation.metadata) < 1:
            continue
        rank_scores: list[tuple[float, int]] = []
        for rank in config.operators.low_rank.ranks:
            candidate = LowRankResidualTransport(rank, config.operators.low_rank.ridge_alpha).fit(
                train.source, train.target
            )
            score = float(
                np.mean(
                    space.distances(candidate.predict(validation.source), validation.target)[
                        "whitened_l2"
                    ]
                )
            )
            rank_scores.append((score, rank))
        _, selected_rank = min(rank_scores, key=lambda item: (item[0], item[1]))
        models: list[Any] = [
            IdentityTransport().fit(train.source, train.target),
            MeanShiftTransport().fit(train.source, train.target),
            AffineRidgeTransport(pca_dimension, config.operators.affine_ridge_alpha).fit(
                train.source, train.target
            ),
            LowRankResidualTransport(selected_rank, config.operators.low_rank.ridge_alpha).fit(
                train.source, train.target
            ),
        ]
        group_dir = output_dir / group_slug(group)
        for model in models:
            path = group_dir / f"{model.model_type}.safetensors"
            model.save(
                path,
                {
                    "operator_group": group,
                    "layer": layer,
                    "fit_split": "train",
                    "rank_selected_on": "validation"
                    if model.model_type == "low_rank_residual"
                    else None,
                },
            )
            records.append(
                {
                    "operator_group": group,
                    "model_type": model.model_type,
                    "path": str(path.relative_to(run_dir)),
                    "sha256": file_hash(path),
                    "capacity": model.capacity(),
                    "fit_rows": len(train.metadata),
                    "validation_rows": len(validation.metadata),
                }
            )
        transform_name = group.rsplit("|", 1)[-1]
        if config.operators.generator.enabled and transform_name in {
            "pressure_shift",
            "concentration_shift",
        }:
            generator_scores: list[tuple[float, int, ContinuousGeneratorTransport]] = []
            train_deltas = parsed_deltas(train.metadata)
            validation_deltas = parsed_deltas(validation.metadata)
            for dimension in config.operators.generator.reduced_dims:
                generator = ContinuousGeneratorTransport(
                    dimension, config.operators.generator.regularization
                ).fit_with_deltas(train.source, train.target, train_deltas)
                prediction = generator.predict_delta(validation.source, validation_deltas)
                score = float(
                    np.mean(space.distances(prediction, validation.target)["whitened_l2"])
                )
                generator_scores.append((score, dimension, generator))
            _, _, generator = min(generator_scores, key=lambda item: (item[0], item[1]))
            path = group_dir / "continuous_generator.safetensors"
            generator.save(
                path,
                {
                    "operator_group": group,
                    "layer": layer,
                    "fit_split": "train",
                    "dimension_selected_on": "validation",
                },
            )
            records.append(
                {
                    "operator_group": group,
                    "model_type": generator.model_type,
                    "path": str(path.relative_to(run_dir)),
                    "sha256": file_hash(path),
                    "capacity": generator.capacity(),
                    "fit_rows": len(train.metadata),
                    "validation_rows": len(validation.metadata),
                }
            )
    selection_payload = {
        **selection,
        "config_hash": config.config_hash,
        "activation_manifest_hash": file_hash(run_dir / "activations" / "manifest.json"),
        "test_data_used": False,
    }
    selection_payload["freeze_hash"] = canonical_hash(selection_payload)
    selection_path = output_dir / "selection_frozen.json"
    write_json_atomic(selection_path, selection_payload)
    manifest = {
        "schema_version": "gct-operators-v1",
        "status": "complete",
        "config_hash": config.config_hash,
        "model_revision": activation_manifest["model_revision"],
        "selection": selection_payload,
        "selection_artifact": artifact_record(selection_path, run_dir, "frozen_selection"),
        "validation_scan": artifact_record(scan_path, run_dir, "validation_selection"),
        "metric_space": artifact_record(space_path, run_dir, "train_fit_metric_space"),
        "operators": records,
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    update_run_manifest(
        run_dir,
        primary_layer=layer,
        frozen_selection_hash=selection_payload["freeze_hash"],
        operator_manifest_hash=file_hash(manifest_path),
        status="operators_complete",
    )
    return run_dir
