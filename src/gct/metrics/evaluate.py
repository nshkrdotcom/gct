"""Frozen held-out evaluation of all empirical defect families and behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from safetensors.torch import load_file

from gct.analysis.pairs import load_layer_dataset, paired_data, parsed_deltas
from gct.config import ExperimentConfig
from gct.metrics.cycles import cycle_defect
from gct.metrics.descent import matching_proxy
from gct.metrics.distances import MetricSpace
from gct.metrics.squares import commuting_square_defect
from gct.metrics.transport import normalized_defect, normalized_improvement
from gct.operators.generator import ContinuousGeneratorTransport
from gct.operators.registry import load_transport
from gct.preprocessing.pca import PCASpace
from gct.preprocessing.scaling import Standardizer
from gct.provenance import update_run_manifest
from gct.storage.hashes import file_hash
from gct.storage.manifests import artifact_record, read_json, write_json_atomic


def load_metric_space(run_dir: Path) -> MetricSpace:
    values = load_file(run_dir / "preprocessing" / "metric_space.safetensors", device="cpu")
    return MetricSpace(
        Standardizer(values["standard_mean"].numpy(), values["standard_scale"].numpy()),
        PCASpace(
            values["pca_mean"].numpy(),
            values["pca_components"].numpy(),
            values["pca_variance"].numpy(),
        ),
    )


def _operator_lookup(manifest: dict[str, Any], run_dir: Path) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for record in manifest["operators"]:
        path = run_dir / str(record["path"])
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"operator hash mismatch: {path}")
        result[(str(record["operator_group"]), str(record["model_type"]))] = load_transport(path)
    return result


def _distance_rows(
    metadata: pd.DataFrame,
    group: str,
    model_type: str,
    distances: dict[str, np.ndarray],
    identity: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, values in distances.items():
        normalized = normalized_defect(values, identity[metric])
        improvement = normalized_improvement(values, identity[metric])
        for index, value in enumerate(values):
            item = metadata.iloc[index]
            rows.append(
                {
                    "sample_id": str(item["sample_id"]),
                    "source_sample_id": str(item["source_sample_id"]),
                    "base_world_id": str(item["base_world_id"]),
                    "split": str(item["split"]),
                    "world_variant": str(item["world_variant"]),
                    "coordinate_condition": str(item["coordinate_condition"]),
                    "transform_family": str(item["transform_family"]),
                    "transform_name": str(item["transform_name"]),
                    "operator_group": group,
                    "model_type": model_type,
                    "metric": metric,
                    "raw_defect": float(value),
                    "identity_defect": float(identity[metric][index]),
                    "normalized_defect": float(normalized[index]),
                    "improvement_over_identity": float(improvement[index]),
                }
            )
    return rows


def _evaluate_transport(
    frame: pd.DataFrame,
    values: np.ndarray,
    space: MetricSpace,
    operators: dict[tuple[str, str], Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = sorted({key[0] for key in operators})
    for group in groups:
        pairs = paired_data(frame, values, group=group)
        if pairs.metadata.empty:
            continue
        identity_distances = space.distances(pairs.source, pairs.target)
        for candidate_group, model_type in sorted(operators):
            if candidate_group != group:
                continue
            model = operators[(group, model_type)]
            if isinstance(model, ContinuousGeneratorTransport):
                prediction = model.predict_delta(pairs.source, parsed_deltas(pairs.metadata))
            else:
                prediction = model.predict(pairs.source)
            distances = space.distances(prediction, pairs.target)
            rows.extend(
                _distance_rows(pairs.metadata, group, model_type, distances, identity_distances)
            )
    return pd.DataFrame(rows)


def _evaluate_cycles(
    frame: pd.DataFrame,
    values: np.ndarray,
    space: MetricSpace,
    operators: dict[tuple[str, str], Any],
) -> pd.DataFrame:
    position = {str(sample): index for index, sample in enumerate(frame["sample_id"])}
    forward_rows = frame[frame["transform_name"] == "nuisance_rewrite"].copy()
    output: list[dict[str, Any]] = []
    for row in forward_rows.itertuples(index=False):
        prefix = f"{row.world_variant}|{row.coordinate_condition}|"
        forward = operators.get((prefix + "nuisance_rewrite", "low_rank_residual"))
        inverse = operators.get((prefix + "nuisance_inverse", "low_rank_residual"))
        if forward is None or inverse is None:
            continue
        start = values[[position[str(row.source_sample_id)]]]
        endpoint = inverse.predict(forward.predict(start))
        distances = cycle_defect(start, endpoint, space)
        for metric, metric_values in distances.items():
            output.append(
                {
                    "cycle_id": str(row.cycle_id),
                    "base_world_id": str(row.base_world_id),
                    "split": str(row.split),
                    "world_variant": str(row.world_variant),
                    "coordinate_condition": str(row.coordinate_condition),
                    "metric": metric,
                    "cycle_defect": float(metric_values[0]),
                    "cycle_kind": "oracle_identity_nuisance",
                }
            )
    return pd.DataFrame(output)


def _evaluate_squares(
    frame: pd.DataFrame,
    values: np.ndarray,
    space: MetricSpace,
    operators: dict[tuple[str, str], Any],
) -> pd.DataFrame:
    position = {str(sample): index for index, sample in enumerate(frame["sample_id"])}
    final_rows = frame[frame["transform_name"] == "square_final"]
    output: list[dict[str, Any]] = []
    for row in final_rows.itertuples(index=False):
        prefix = f"{row.world_variant}|{row.coordinate_condition}|"
        pressure = operators.get((prefix + "pressure_shift", "continuous_generator"))
        concentration = operators.get((prefix + "concentration_shift", "continuous_generator"))
        if not isinstance(pressure, ContinuousGeneratorTransport) or not isinstance(
            concentration, ContinuousGeneratorTransport
        ):
            continue
        parameters = json.loads(str(row.transform_parameters_json))
        delta_p = float(parameters["delta_p"])
        delta_m = float(parameters["delta_m"])
        start = values[[position[str(row.source_sample_id)]]]
        target = values[[position[str(row.sample_id)]]]
        route_pm = concentration.predict_delta(pressure.predict_delta(start, delta_p), delta_m)
        route_mp = pressure.predict_delta(concentration.predict_delta(start, delta_m), delta_p)
        square = commuting_square_defect(route_pm, route_mp, space)
        pm_target = space.distances(route_pm, target)
        mp_target = space.distances(route_mp, target)
        for metric in square:
            output.append(
                {
                    "square_id": str(row.square_id),
                    "base_world_id": str(row.base_world_id),
                    "split": str(row.split),
                    "world_variant": str(row.world_variant),
                    "metric": metric,
                    "commuting_square_defect": float(square[metric][0]),
                    "route_pm_target_defect": float(pm_target[metric][0]),
                    "route_mp_target_defect": float(mp_target[metric][0]),
                    "delta_p": delta_p,
                    "delta_m": delta_m,
                    "label": "empirical commuting-square defect",
                }
            )
    return pd.DataFrame(output)


def _evaluate_generator_composition(
    frame: pd.DataFrame,
    values: np.ndarray,
    space: MetricSpace,
    operators: dict[tuple[str, str], Any],
) -> pd.DataFrame:
    """Evaluate T_(a+b) against T_b T_a and both routes against observed targets."""
    output: list[dict[str, Any]] = []
    for (group, model_type), model in sorted(operators.items()):
        transform_name = group.rsplit("|", 1)[-1]
        if model_type != "continuous_generator" or transform_name not in {
            "pressure_shift",
            "concentration_shift",
        }:
            continue
        if not isinstance(model, ContinuousGeneratorTransport):
            raise TypeError("generator registry entry has the wrong concrete type")
        pairs = paired_data(frame, values, group=group)
        if pairs.metadata.empty:
            continue
        deltas = parsed_deltas(pairs.metadata)
        first_deltas = deltas / 2.0
        second_deltas = deltas - first_deltas
        direct = model.predict_delta(pairs.source, deltas)
        first = model.predict_delta(pairs.source, first_deltas)
        composed = model.predict_delta(first, second_deltas)
        composition = space.distances(composed, direct)
        direct_target = space.distances(direct, pairs.target)
        composed_target = space.distances(composed, pairs.target)
        for metric, metric_values in composition.items():
            for index, value in enumerate(metric_values):
                row = pairs.metadata.iloc[index]
                output.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "base_world_id": str(row["base_world_id"]),
                        "split": str(row["split"]),
                        "world_variant": str(row["world_variant"]),
                        "coordinate_condition": str(row["coordinate_condition"]),
                        "transform_name": transform_name,
                        "operator_group": group,
                        "metric": metric,
                        "delta_total": float(deltas[index]),
                        "delta_a": float(first_deltas[index]),
                        "delta_b": float(second_deltas[index]),
                        "composition_defect": float(value),
                        "direct_target_defect": float(direct_target[metric][index]),
                        "composed_target_defect": float(composed_target[metric][index]),
                        "label": "empirical continuous-generator composition proxy",
                    }
                )
    return pd.DataFrame(output)


def _evaluate_descent(frame: pd.DataFrame, values: np.ndarray, space: MetricSpace) -> pd.DataFrame:
    pairs = paired_data(frame, values)
    mask = pairs.metadata["transform_name"].to_numpy() == "nuisance_rewrite"
    metadata = pairs.metadata.loc[mask].reset_index(drop=True)
    distances = matching_proxy(pairs.source[mask], pairs.target[mask], space)
    output: list[dict[str, Any]] = []
    for metric, metric_values in distances.items():
        for index, value in enumerate(metric_values):
            row = metadata.iloc[index]
            output.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "base_world_id": str(row["base_world_id"]),
                    "split": str(row["split"]),
                    "world_variant": str(row["world_variant"]),
                    "coordinate_condition": str(row["coordinate_condition"]),
                    "metric": metric,
                    "matching_descent_proxy": float(value),
                    "representation_dependent": True,
                }
            )
    return pd.DataFrame(output)


def _evaluate_behavior_edges(
    run_dir: Path, frame: pd.DataFrame, values: np.ndarray, tolerance: float
) -> pd.DataFrame:
    results = pd.read_parquet(run_dir / "behavior" / "results.parquet")
    behavior = results.set_index("sample_id")
    position = {str(sample): index for index, sample in enumerate(frame["sample_id"])}
    rows: list[dict[str, Any]] = []
    for target in frame[frame["source_sample_id"].notna()].itertuples(index=False):
        item = cast(Any, target)
        source_meta = frame.iloc[position[str(item.source_sample_id)]]
        target_behavior = cast(pd.Series, behavior.loc[str(item.sample_id)])
        source_behavior = cast(pd.Series, behavior.loc[str(item.source_sample_id)])
        target_answer = target_behavior["parsed_answer"]
        source_answer = source_behavior["parsed_answer"]
        parsed_pair = pd.notna(target_answer) and pd.notna(source_answer)
        flip = (
            bool(abs(float(cast(Any, target_answer)) - float(cast(Any, source_answer))) > tolerance)
            if parsed_pair
            else None
        )
        correction = None
        if parsed_pair and item.transform_family == "substantive":
            before = abs(float(cast(Any, source_answer)) - float(item.oracle_target))
            after = abs(float(cast(Any, target_answer)) - float(item.oracle_target))
            correction = bool(after < before)
        rows.append(
            {
                "sample_id": str(item.sample_id),
                "source_sample_id": str(item.source_sample_id),
                "base_world_id": str(item.base_world_id),
                "split": str(item.split),
                "world_variant": str(item.world_variant),
                "coordinate_condition": str(item.coordinate_condition),
                "transform_family": str(item.transform_family),
                "transform_name": str(item.transform_name),
                "parse_status": str(target_behavior["parse_status"]),
                "absolute_oracle_error": (
                    float(cast(Any, target_behavior["absolute_error"]))
                    if pd.notna(target_behavior["absolute_error"])
                    else None
                ),
                "within_tolerance": bool(target_behavior["within_tolerance"]),
                "answer_flip": flip,
                "correction": correction,
                "character_count": int(item.char_count),
                "character_count_difference": int(
                    item.char_count - int(cast(Any, source_meta["char_count"]))
                ),
                "token_count": int(item.token_count),
                "token_count_difference": int(
                    item.token_count - int(cast(Any, source_meta["token_count"]))
                ),
                "activation_norm": float(item.activation_norm),
                "oracle_delta_magnitude": abs(
                    float(item.oracle_target) - float(cast(Any, source_meta["oracle_target"]))
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_metrics(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    operator_manifest = read_json(run_dir / "operators" / "manifest.json")
    behavior_manifest = read_json(run_dir / "behavior" / "manifest.json")
    if (
        operator_manifest.get("status") != "complete"
        or behavior_manifest.get("status") != "complete"
    ):
        raise ValueError("operators and behavior must be complete before metric evaluation")
    if (
        operator_manifest.get("config_hash") != config.config_hash
        or behavior_manifest.get("config_hash") != config.config_hash
    ):
        raise ValueError("operator/behavior config differs from metric config")
    selection = read_json(run_dir / "operators" / "selection_frozen.json")
    if selection.get("test_data_used") is not False:
        raise ValueError("frozen selection does not certify validation-only selection")
    layer = int(selection["primary_layer"])
    frame, values = load_layer_dataset(run_dir, layer)
    space = load_metric_space(run_dir)
    operators = _operator_lookup(operator_manifest, run_dir)
    tables = {
        "transport_edges": _evaluate_transport(frame, values, space, operators),
        "cycles": _evaluate_cycles(frame, values, space, operators),
        "squares": _evaluate_squares(frame, values, space, operators),
        "generator_composition": _evaluate_generator_composition(frame, values, space, operators),
        "descent": _evaluate_descent(frame, values, space),
        "behavior_edges": _evaluate_behavior_edges(
            run_dir, frame, values, config.metrics.behavior_tolerance_c
        ),
    }
    output_dir = run_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False, compression="zstd")
        artifacts.append(artifact_record(path, run_dir, f"metrics_{name}"))
    manifest = {
        "schema_version": "gct-metrics-v1",
        "status": "complete",
        "config_hash": config.config_hash,
        "selection_freeze_hash": selection["freeze_hash"],
        "operator_manifest_hash": file_hash(run_dir / "operators" / "manifest.json"),
        "behavior_manifest_hash": file_hash(run_dir / "behavior" / "manifest.json"),
        "primary_layer": layer,
        "test_evaluated_after_freeze": True,
        "tables": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    update_run_manifest(
        run_dir, metrics_manifest_hash=file_hash(manifest_path), status="metrics_complete"
    )
    return run_dir
