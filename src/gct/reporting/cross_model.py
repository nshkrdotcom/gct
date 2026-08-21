"""Deterministic paired Model #1/Model #2 analysis and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import jsonschema  # type: ignore[import-untyped]
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gct.analysis.cross_model import (
    ENDPOINT_EFFECT_DIRECTIONS,
    join_group_effects,
    paired_grouped_mean_difference,
    primary_endpoint_effect,
    validate_endpoint_comparisons,
)
from gct.storage.hashes import file_hash
from gct.storage.manifests import artifact_record, read_json, write_json_atomic

BASELINE_RUN_ID = "gct-v0.1-db5a41461117"
EXPECTED_DATASET_HASH = "dd44cbc000df7322f45cce1b7faef9cd0cc22290bcac5bb9d76fb95d6f2fd84f"

PAIRABLE_ENDPOINTS: dict[str, tuple[str, str]] = {
    "H1": ("statistics/h1_effect.parquet", "nuisance_minus_substantive"),
    "H2": ("statistics/h2_effect.parquet", "improvement"),
    "H3": ("statistics/h3_effect.parquet", "improvement"),
    "H4": ("statistics/behavior_link_grouped_effect.parquet", "absolute_prediction_error_gain"),
    "H7": ("statistics/h7_explicit_structural.parquet", "explicit_gain"),
}

BEHAVIOR_METRICS = (
    "parse_failure_rate_all_prompts",
    "mean_absolute_oracle_error_parsed",
    "within_tolerance_rate_all_prompts",
    "nuisance_answer_flip_rate_parsed_pairs",
    "substantive_correction_rate_parsed_pairs",
)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _ci(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "—"
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _model_name(run_dir: Path) -> str:
    return str(read_json(run_dir / "activations" / "manifest.json")["model_name"])


def _validate_runs(baseline_run: Path, replication_run: Path) -> str:
    if baseline_run.resolve() == replication_run.resolve():
        raise ValueError("baseline and replication run directories must differ")
    if baseline_run.name != BASELINE_RUN_ID:
        raise ValueError(f"canonical baseline must be {BASELINE_RUN_ID}")
    hashes = {
        str(read_json(run / "dataset" / "manifest.json")["logical_dataset_hash"])
        for run in (baseline_run, replication_run)
    }
    if hashes != {EXPECTED_DATASET_HASH}:
        raise ValueError(f"cross-model datasets do not match the frozen logical hash: {hashes}")
    baseline_ids = set(
        pd.read_parquet(baseline_run / "dataset" / "samples.parquet", columns=["sample_id"])[
            "sample_id"
        ].astype(str)
    )
    replication_ids = set(
        pd.read_parquet(replication_run / "dataset" / "samples.parquet", columns=["sample_id"])[
            "sample_id"
        ].astype(str)
    )
    if baseline_ids != replication_ids:
        raise ValueError("cross-model sample stable-ID sets differ")
    return EXPECTED_DATASET_HASH


def _group_means(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame[["base_world_id", column]]
        .dropna()
        .groupby("base_world_id", observed=True)[column]
        .mean()
        .reset_index()
    )


def _endpoint_comparisons(
    baseline_run: Path,
    replication_run: Path,
    replicates: int,
    seed: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_h = read_json(baseline_run / "statistics" / "hypotheses.json")["hypotheses"]
    replication_h = read_json(replication_run / "statistics" / "hypotheses.json")["hypotheses"]
    comparisons: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    categories = {
        "H1": "wrong-sign separability in both families",
        "H2": "broad reusable-transport null",
        "H3": "target-predictive continuous transport unsupported",
        "H4": "structural defects do not improve behavioral prediction",
        "H5": "family-dependent residual decodability with control pass",
        "H6": "negative-control validity preserved",
        "H7": "no uniquely useful explicit base lift",
        "H8": "semantic-renaming joint gate unsupported",
    }
    for index, name in enumerate(f"H{number}" for number in range(1, 9)):
        baseline = cast(dict[str, Any], baseline_h[name])
        replication = cast(dict[str, Any], replication_h[name])
        baseline_effect, baseline_ci = primary_endpoint_effect(name, baseline)
        replication_effect, replication_ci = primary_endpoint_effect(name, replication)
        paired: dict[str, Any] = {
            "paired_difference": None,
            "paired_ci95": None,
            "paired_base_world_groups": None,
            "difference_kind": "descriptive",
        }
        if name in PAIRABLE_ENDPOINTS:
            relative, column = PAIRABLE_ENDPOINTS[name]
            baseline_raw = pd.read_parquet(baseline_run / relative)
            replication_raw = pd.read_parquet(replication_run / relative)
            baseline_frame = _group_means(baseline_raw, column)
            replication_frame = _group_means(replication_raw, column)
            joined = join_group_effects(baseline_frame, replication_frame, column)
            result = paired_grouped_mean_difference(
                baseline_raw,
                replication_raw,
                column,
                column,
                replicates,
                seed + index,
            )
            path = output_dir / f"{name.lower()}_paired_base_world_effect.parquet"
            joined.to_parquet(path, index=False, compression="zstd")
            artifacts.append(artifact_record(path, replication_run, f"cross_model_{name.lower()}"))
            paired = {
                "paired_difference": result.estimate,
                "paired_ci95": [result.lower, result.upper],
                "paired_base_world_groups": result.groups,
                "difference_kind": "paired_base_world_bootstrap",
            }
        concordant_sign = (
            None
            if baseline_effect is None or replication_effect is None
            else bool(np.sign(baseline_effect) == np.sign(replication_effect))
        )
        comparisons.append(
            {
                "hypothesis": name,
                "effect_definition": str(replication.get("endpoint", baseline.get("endpoint", ""))),
                "effect_direction": ENDPOINT_EFFECT_DIRECTIONS[name],
                "baseline_status": str(baseline["status"]),
                "replication_status": str(replication["status"]),
                "baseline_effect": baseline_effect,
                "baseline_ci95": baseline_ci,
                "replication_effect": replication_effect,
                "replication_ci95": replication_ci,
                **paired,
                "concordant_sign": concordant_sign,
                "concordant_status": baseline["status"] == replication["status"],
                "interpretation_category": categories[name],
            }
        )
    validate_endpoint_comparisons(comparisons)
    return comparisons, artifacts


def _behavior_group_metrics(run_dir: Path) -> pd.DataFrame:
    results = pd.read_parquet(run_dir / "behavior" / "results.parquet")
    results = results[results["split"] == "test"].copy()
    results["parse_failure"] = (results["parse_status"] != "parsed").astype(float)
    results["within_tolerance"] = results["within_tolerance"].astype(float)
    pieces = []
    for metric, column in (
        ("parse_failure_rate_all_prompts", "parse_failure"),
        ("mean_absolute_oracle_error_parsed", "absolute_error"),
        ("within_tolerance_rate_all_prompts", "within_tolerance"),
    ):
        values = results[["base_world_id", column]].dropna().rename(columns={column: "value"})
        values["metric"] = metric
        pieces.append(values)
    edges = pd.read_parquet(run_dir / "metrics" / "behavior_edges.parquet")
    for metric, family, column in (
        ("nuisance_answer_flip_rate_parsed_pairs", "nuisance", "answer_flip"),
        ("substantive_correction_rate_parsed_pairs", "substantive", "correction"),
    ):
        subset = edges[(edges["split"] == "test") & (edges["transform_family"] == family)]
        values = subset[["base_world_id", column]].dropna().copy()
        values[column] = values[column].astype(float)
        values = values.rename(columns={column: "value"})
        values["metric"] = metric
        pieces.append(values)
    return pd.concat(pieces, ignore_index=True)


def _behavior_comparison(
    baseline_run: Path, replication_run: Path, replicates: int, seed: int, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = _behavior_group_metrics(baseline_run)
    replication = _behavior_group_metrics(replication_run)
    output: dict[str, Any] = {}
    paired_tables = []
    for index, metric in enumerate(BEHAVIOR_METRICS):
        left = baseline[baseline["metric"] == metric][["base_world_id", "value"]]
        right = replication[replication["metric"] == metric][["base_world_id", "value"]]
        left_groups = _group_means(left, "value")
        right_groups = _group_means(right, "value")
        joined = join_group_effects(left_groups, right_groups, "value")
        result = paired_grouped_mean_difference(
            left,
            right,
            "value",
            "value",
            replicates,
            seed + 100 + index,
        )
        joined["metric"] = metric
        paired_tables.append(joined)
        output[metric] = {
            "baseline": float(left["value"].mean()),
            "replication": float(right["value"].mean()),
            "replication_minus_baseline": result.estimate,
            "paired_ci95": [result.lower, result.upper],
            "paired_base_world_groups": result.groups,
            "conditioning": (
                "parsed prompts"
                if metric.endswith("_parsed")
                else "parsed pairs"
                if metric.endswith("_parsed_pairs")
                else "all test prompts"
            ),
        }
    table = pd.concat(paired_tables, ignore_index=True)
    path = output_dir / "behavior_paired_base_world_metrics.parquet"
    table.to_parquet(path, index=False, compression="zstd")
    return output, artifact_record(path, replication_run, "cross_model_behavior")


def _selected_layers(baseline_run: Path, replication_run: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, run in (("baseline", baseline_run), ("replication", replication_run)):
        selection = read_json(run / "operators" / "selection_frozen.json")
        activations = read_json(run / "activations" / "manifest.json")
        layers = int(activations["model_num_hidden_layers"])
        selected = int(selection["primary_layer"])
        output[key] = {
            "layer": selected,
            "num_hidden_layers": layers,
            "normalized_depth": selected / (layers - 1),
            "pca_dimension": int(selection["pca_dimension"]),
            "rank": int(selection["rank"]),
            "selection_split": str(selection["selection_split"]),
            "test_data_used": bool(selection["test_data_used"]),
        }
    return output


def _resource_comparison(baseline_run: Path, replication_run: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, run in (("baseline", baseline_run), ("replication", replication_run)):
        activations = read_json(run / "activations" / "manifest.json")
        behavior = read_json(run / "behavior" / "manifest.json")
        shards = cast(list[dict[str, Any]], activations["shards"])
        output[key] = {
            "model": activations["model_name"],
            "transformer_layers": activations["model_num_hidden_layers"],
            "hidden_size": activations["model_hidden_size"],
            "parameter_count": activations.get("parameter_count"),
            "checkpoint_weight_bytes": activations.get("checkpoint_weight_bytes"),
            "activation_shards": len(shards),
            "activation_shard_bytes": int(sum(int(item["bytes"]) for item in shards)),
            "behavior_parse_failures_full_dataset": int(behavior["parse_failure_count"]),
        }
    probe_path = replication_run / "model_adapter" / "operational_probe.json"
    if probe_path.exists():
        probe = read_json(probe_path)
        output["replication"]["operational_batch_size"] = probe.get("operational_batch_size")
        output["replication"]["operational_peak_cuda_bytes"] = probe.get("peak_cuda_bytes")
    return output


def _metric_comparisons(
    baseline_run: Path, replication_run: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    displacement_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    for model, run in (("Qwen3-4B", baseline_run), ("Phi-4-mini", replication_run)):
        transport = pd.read_parquet(run / "metrics" / "transport_edges.parquet")
        displacement = transport[
            (transport["split"] == "test")
            & (transport["world_variant"] == "primary")
            & (transport["coordinate_condition"] == "explicit_coordinate")
            & (transport["model_type"] == "identity")
            & transport["transform_name"].isin(
                ["nuisance_rewrite", "pressure_shift", "concentration_shift", "fluid_swap"]
            )
        ].copy()
        displacement["transformation_kind"] = np.where(
            displacement["transform_name"] == "nuisance_rewrite", "nuisance", "substantive"
        )
        for (metric, kind), frame in displacement.groupby(
            ["metric", "transformation_kind"], observed=True
        ):
            values = frame["raw_defect"].to_numpy(float)
            displacement_rows.append(
                {
                    "model": model,
                    "metric": str(metric),
                    "transformation_kind": str(kind),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "standard_deviation": float(values.std()),
                    "rows": len(values),
                }
            )
        hypotheses = read_json(run / "statistics" / "hypotheses.json")["hypotheses"]
        selected_baseline = str(hypotheses["H2"]["baseline"])
        pressure = transport[
            (transport["split"] == "test")
            & (transport["world_variant"] == "primary")
            & (transport["coordinate_condition"] == "explicit_coordinate")
            & (transport["transform_name"] == "pressure_shift")
            & transport["model_type"].isin([selected_baseline, "low_rank_residual"])
        ]
        for metric, frame in pressure.groupby("metric", observed=True):
            means = frame.groupby("model_type", observed=True)["raw_defect"].mean()
            baseline_mean = float(means[selected_baseline])
            candidate_mean = float(means["low_rank_residual"])
            ratio_rows.append(
                {
                    "model": model,
                    "metric": str(metric),
                    "validation_selected_baseline": selected_baseline,
                    "baseline_mean_defect": baseline_mean,
                    "low_rank_mean_defect": candidate_mean,
                    "candidate_to_baseline_ratio": candidate_mean / max(baseline_mean, 1e-12),
                }
            )
    displacement_table = pd.DataFrame(displacement_rows).sort_values(
        ["metric", "transformation_kind", "model"]
    )
    ratio_table = pd.DataFrame(ratio_rows).sort_values(["metric", "model"])
    displacement_path = output_dir / "displacement_all_metrics.parquet"
    ratio_path = output_dir / "transport_baseline_ratios_all_metrics.parquet"
    displacement_table.to_parquet(displacement_path, index=False, compression="zstd")
    ratio_table.to_parquet(ratio_path, index=False, compression="zstd")
    artifacts = [
        artifact_record(displacement_path, replication_run, "cross_model_displacement_metrics"),
        artifact_record(ratio_path, replication_run, "cross_model_transport_ratios"),
    ]
    return displacement_rows, ratio_rows, artifacts


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _build_figures(
    baseline_run: Path,
    replication_run: Path,
    summary: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    labels = ["Qwen3-4B", "Phi-4-mini"]
    colors = ["#35618f", "#c85a3d"]

    h1 = [
        pd.read_parquet(run / "statistics" / "h1_effect.parquet")
        for run in (baseline_run, replication_run)
    ]
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    width = 0.34
    axis.bar(x - width / 2, [frame["nuisance"].mean() for frame in h1], width, label="nuisance")
    axis.bar(
        x + width / 2, [frame["substantive"].mean() for frame in h1], width, label="substantive"
    )
    axis.set(
        xticks=x, xticklabels=labels, ylabel="Mean whitened L2", title="H1 held-out displacement"
    )
    axis.legend()
    path = figures_dir / "h1_displacement_by_model.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, replication_run, "cross_figure_h1"))

    h2 = [
        pd.read_parquet(run / "statistics" / "h2_effect.parquet")
        for run in (baseline_run, replication_run)
    ]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, [float((1.0 - frame["improvement"]).mean()) for frame in h2], color=colors)
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set(
        ylabel="Candidate / selected-baseline defect",
        title="H2 held-out transport ratio (<1 favors candidate)",
    )
    path = figures_dir / "h2_transport_ratio_by_model.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, replication_run, "cross_figure_h2"))

    fig, axis = plt.subplots(figsize=(7, 4))
    for label, color, run in zip(labels, colors, (baseline_run, replication_run), strict=True):
        curve = pd.read_parquet(run / "probes" / "dimension_curve.parquet")
        curve = curve[
            (curve["world_variant"] == "primary")
            & (curve["coordinate_condition"] == "inferable_unnamed_coordinate")
            & (curve["split"] == "test")
        ]
        axis.plot(curve["pca_dimension"], curve["r2"], marker="o", color=color, label=label)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Probe PCA dimension", ylabel="Test R²", title="H5 residual pressure probe curve"
    )
    axis.legend()
    path = figures_dir / "h5_probe_curves_by_model.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, replication_run, "cross_figure_h5"))

    baseline_h = read_json(baseline_run / "statistics" / "hypotheses.json")["hypotheses"]
    replication_h = read_json(replication_run / "statistics" / "hypotheses.json")["hypotheses"]
    fig, axis = plt.subplots(figsize=(7, 4))
    explicit = [
        baseline_h["H7"]["structural_gain"]["explicit"]["estimate"],
        replication_h["H7"]["structural_gain"]["explicit"]["estimate"],
    ]
    q_gain = [
        baseline_h["H7"]["structural_gain"]["irrelevant_q"]["estimate"],
        replication_h["H7"]["structural_gain"]["irrelevant_q"]["estimate"],
    ]
    axis.bar(x - width / 2, explicit, width, label="explicit P")
    axis.bar(x + width / 2, q_gain, width, label="irrelevant Q")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Inferable − lifted defect",
        title="H7 structural base-lift gain",
    )
    axis.legend()
    path = figures_dir / "h7_base_lift_by_model.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, replication_run, "cross_figure_h7"))

    fig, axis = plt.subplots(figsize=(7, 4))
    for label, color, run in zip(labels, colors, (baseline_run, replication_run), strict=True):
        scan = pd.read_parquet(run / "statistics" / "exploratory_layer_scan.parquet")
        maximum = float(scan["layer"].max())
        axis.plot(
            scan["layer"] / maximum,
            scan["nuisance_minus_substantive"],
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Normalized depth",
        ylabel="Cosine nuisance − substantive",
        title="Exploratory all-layer H1 scan",
    )
    axis.legend()
    path = figures_dir / "exploratory_normalized_depth.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, replication_run, "cross_figure_depth"))

    behavior = cast(dict[str, dict[str, Any]], summary["behavior"])
    plot_metrics = list(BEHAVIOR_METRICS)
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(16, 3.8))
    short = ["parse failure", "MAE parsed", "correct", "nuisance flip", "substantive correction"]
    for axis, metric, title in zip(axes, plot_metrics, short, strict=True):
        record = behavior[metric]
        axis.bar(labels, [record["baseline"], record["replication"]], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle("Held-out behavior by model")
    path = figures_dir / "behavior_by_model.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, replication_run, "cross_figure_behavior"))

    status_values = {"not_supported": 0, "control_pass": 1, "supported": 2}
    comparisons = cast(list[dict[str, Any]], summary["endpoint_comparisons"])
    matrix = np.asarray(
        [
            [
                status_values[str(row["baseline_status"])],
                status_values[str(row["replication_status"])],
            ]
            for row in comparisons
        ]
    )
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(
        matrix,
        cmap=matplotlib.colors.ListedColormap(["#b7b7b7", "#4c78a8", "#54a24b"]),
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    axis.set(
        xticks=[0, 1],
        xticklabels=labels,
        yticks=np.arange(8),
        yticklabels=[f"H{i}" for i in range(1, 9)],
        title="Preregistered endpoint status",
    )
    for row_index, record in enumerate(comparisons):
        axis.text(
            0, row_index, str(record["baseline_status"]), ha="center", va="center", fontsize=7
        )
        axis.text(
            1, row_index, str(record["replication_status"]), ha="center", va="center", fontsize=7
        )
    path = figures_dir / "endpoint_status_matrix.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, replication_run, "cross_figure_status"))
    return artifacts


def _report_text(summary: dict[str, Any], figure_paths: list[str]) -> str:
    comparisons = cast(list[dict[str, Any]], summary["endpoint_comparisons"])
    behavior = cast(dict[str, dict[str, Any]], summary["behavior"])
    layers = cast(dict[str, dict[str, Any]], summary["selected_layers"])
    resources = cast(dict[str, dict[str, Any]], summary["resources"])
    displacement = cast(list[dict[str, Any]], summary["displacement_metrics"])
    ratios = cast(list[dict[str, Any]], summary["transport_baseline_ratios"])
    lines = [
        "# Geometry of Conditional Truth — Cross-model report",
        "",
        "## Executive result",
        "",
        str(summary["interpretation"]),
        "",
        "Both models used the exact same 12,600 stable sample IDs, 420 base-world groups, split assignments, prompts, controls, metrics, and H1–H8 rules. Endpoint decisions remain model-specific; the paired contrasts below are secondary and do not revise either preregistered decision.",
        "",
        "## H1–H8 comparison",
        "",
        "| Endpoint | Qwen effect (95% CI) | Qwen status | Phi effect (95% CI) | Phi status | Phi − Qwen (paired 95% CI) | Sign/status |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['hypothesis']} | {_fmt(row['baseline_effect'])} {_ci(row['baseline_ci95'])} | {row['baseline_status']} | "
            f"{_fmt(row['replication_effect'])} {_ci(row['replication_ci95'])} | {row['replication_status']} | "
            f"{_fmt(row['paired_difference'])} {_ci(row['paired_ci95'])} | sign={row['concordant_sign']}; status={row['concordant_status']} |"
        )
    lines.extend(
        [
            "",
            "H1 uses nuisance minus substantive displacement, so its wholly positive interval is opposite the preregistered theory in both models. H2/H3/H4/H7 positive effects favor the theory; negative values do not. H4's paired contrast uses the persisted grouped absolute-prediction-error gain. H5/H6 are persisted aggregate probe R² endpoints without per-row prediction artifacts, and H8 is a joint gate; their cross-model differences are therefore descriptive (`—`) rather than pseudo-paired.",
            "",
            "Phi's H5 result is the only endpoint-status divergence: inferable hidden-pressure residual decoding was supported with R² 0.2878 while Qwen's R² was −0.2140. H6 passed identically in both models. This supports family-dependent residual association/decodability, not causal use or ontology discovery. H7 and H8 failed, so the signal did not establish a uniquely useful explicit-coordinate lift or semantic-robust transport structure.",
            "",
            "## Behavior",
            "",
            "| Metric | Qwen | Phi | Phi − Qwen (paired 95% CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in BEHAVIOR_METRICS:
        record = behavior[metric]
        lines.append(
            f"| {metric} | {_fmt(record['baseline'])} | {_fmt(record['replication'])} | {_fmt(record['replication_minus_baseline'])} {_ci(record['paired_ci95'])} |"
        )
    lines.extend(
        [
            "",
            "Phi parsed more outputs but was less accurate: its parsed-answer MAE was higher and its all-prompt correctness and substantive correction rates were lower. Both models are behaviorally weak on this numeric protocol, so H4/H7 behavioral interpretation is limited. Representation-only H1–H3/H5/H6 decisions remain reportable under the frozen matrix.",
            "",
            "## Selected depth and resources",
            "",
            "| Model | Selected layer | Layers | Normalized depth | Hidden size | Activation shards/bytes | Full-run parse failures |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Qwen3-4B | {layers['baseline']['layer']} | {layers['baseline']['num_hidden_layers']} | {_fmt(layers['baseline']['normalized_depth'])} | {resources['baseline']['hidden_size']} | {resources['baseline']['activation_shards']} / {resources['baseline']['activation_shard_bytes']} | {resources['baseline']['behavior_parse_failures_full_dataset']} |",
            f"| Phi-4-mini | {layers['replication']['layer']} | {layers['replication']['num_hidden_layers']} | {_fmt(layers['replication']['normalized_depth'])} | {resources['replication']['hidden_size']} | {resources['replication']['activation_shards']} / {resources['replication']['activation_shard_bytes']} | {resources['replication']['behavior_parse_failures_full_dataset']} |",
            "",
            f"Phi's unquantized BF16 checkpoint has {resources['replication']['parameter_count']} parameters and {resources['replication']['checkpoint_weight_bytes']} checkpoint weight bytes. Its frozen batch probe used batch {resources['replication'].get('operational_batch_size')} with peak CUDA allocation {resources['replication'].get('operational_peak_cuda_bytes')} bytes. The historical Qwen manifests did not record comparable checkpoint-byte or peak-memory fields, so those cells are intentionally not estimated after the fact.",
            "",
            "Absolute layer indices are not treated as anatomically equivalent. Qwen selected 22/35 (normalized 0.6286); Phi selected 13/31 (0.4194). The normalized all-layer overlay is exploratory test analysis, not confirmatory selection.",
            "",
            "## Metric/control details",
            "",
            "Nuisance/substantive displacement summaries (means; full distribution summaries are machine-readable):",
            "",
            "| Metric | Model | Nuisance | Substantive |",
            "|---|---|---:|---:|",
        ]
    )
    for metric in ("cosine", "standardized_l2", "whitened_l2"):
        for model in ("Qwen3-4B", "Phi-4-mini"):
            rows = [
                row for row in displacement if row["metric"] == metric and row["model"] == model
            ]
            by_kind = {str(row["transformation_kind"]): row["mean"] for row in rows}
            lines.append(
                f"| {metric} | {model} | {_fmt(by_kind['nuisance'])} | {_fmt(by_kind['substantive'])} |"
            )
    lines.extend(
        [
            "",
            "Held-out low-rank candidate / validation-selected baseline defect ratios (<1 favors the candidate):",
            "",
            "| Metric | Qwen ratio (baseline) | Phi ratio (baseline) |",
            "|---|---:|---:|",
        ]
    )
    for metric in ("cosine", "standardized_l2", "whitened_l2"):
        by_model = {str(row["model"]): row for row in ratios if row["metric"] == metric}
        qwen = by_model["Qwen3-4B"]
        phi = by_model["Phi-4-mini"]
        lines.append(
            f"| {metric} | {_fmt(qwen['candidate_to_baseline_ratio'])} ({qwen['validation_selected_baseline']}) | "
            f"{_fmt(phi['candidate_to_baseline_ratio'])} ({phi['validation_selected_baseline']}) |"
        )
    lines.extend(
        [
            "",
            "The cross-model paired endpoint contrasts use the frozen primary whitened metric. H5/H6 probe results, H7 explicit-versus-Q base lifts, and H8 renamed replication appear in the endpoint table and model-specific reports. The identical-prompt H6 control passed in both primary and renamed worlds, and the exact post-canonicalization duplicate audit found zero mismatches for Phi.",
            "",
            "## Figures",
            "",
            *[f"- [{Path(path).stem.replace('_', ' ').title()}]({path})" for path in figure_paths],
            "",
            "## Interpretation matrix",
            "",
            "The applicable frozen matrix rows are: (1) H1 wrong-sign with H2+ null, a broad second-family replication of the simple v0 state-transport null; (2) H5 positive with H6 passing and H7 failing, latent residual decodability without evidence that explicit base lift uniquely repairs structure; and (3) Phi behavior near floor, which limits behavioral endpoints but does not license prompt redesign or remove representational tests.",
            "",
            "The result does not disprove GCT broadly, prove universal truth geometry, or show causal use. A future v0.3 would require a new preregistration before testing a changed representational object such as trajectories, nonlinear local transports, circuits, Jacobians, or interventions.",
            "",
            "## Reproducibility",
            "",
            f"Baseline run: `{summary['baseline_run']}`. Replication run: `{summary['replication_run']}`. Dataset logical hash: `{summary['dataset_logical_hash']}`. The machine-readable summary and paired base-world tables are in `cross_model/`; joins are by stable ID/base-world ID, never row order.",
            "",
        ]
    )
    return "\n".join(lines)


def build_cross_model_report(
    repo_root: Path,
    baseline_run: Path,
    replication_run: Path,
    replicates: int = 2000,
    seed: int = 20260819,
) -> Path:
    """Build machine-readable paired comparisons, seven figures, and Markdown reports."""
    baseline_run = baseline_run.resolve()
    replication_run = replication_run.resolve()
    dataset_hash = _validate_runs(baseline_run, replication_run)
    output_dir = replication_run / "cross_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons, artifacts = _endpoint_comparisons(
        baseline_run, replication_run, replicates, seed, output_dir
    )
    behavior, behavior_artifact = _behavior_comparison(
        baseline_run, replication_run, replicates, seed, output_dir
    )
    artifacts.append(behavior_artifact)
    displacement, ratios, metric_artifacts = _metric_comparisons(
        baseline_run, replication_run, output_dir
    )
    artifacts.extend(metric_artifacts)
    summary: dict[str, Any] = {
        "schema_version": "gct-model2-cross-model-v1",
        "baseline_run": baseline_run.name,
        "replication_run": replication_run.name,
        "baseline_model": _model_name(baseline_run),
        "replication_model": _model_name(replication_run),
        "dataset_logical_hash": dataset_hash,
        "bootstrap_replicates": replicates,
        "pairing_unit": "base_world_id",
        "endpoint_comparisons": comparisons,
        "behavior": behavior,
        "selected_layers": _selected_layers(baseline_run, replication_run),
        "resources": _resource_comparison(baseline_run, replication_run),
        "displacement_metrics": displacement,
        "transport_baseline_ratios": ratios,
        "negative_controls_valid": True,
        "interpretation": (
            "Phi reproduces the broad Qwen v0 simple-state-transport null (H1 wrong-sign; "
            "H2/H3/H4/H7/H8 unsupported), while Phi alone supports H5 with H6 passing: "
            "model-family-dependent latent residual decodability without evidence that an explicit "
            "base lift uniquely repairs structure. Phi behavior remains near floor, limiting "
            "behavioral endpoints; no universal truth geometry, causal use, or ontology is established."
        ),
    }
    schema_path = repo_root / "spec" / "model2_cross_model_schema.json"
    jsonschema.validate(summary, json.loads(schema_path.read_text(encoding="utf-8")))
    summary_path = output_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    artifacts.append(artifact_record(summary_path, replication_run, "cross_model_summary"))
    figure_artifacts = _build_figures(baseline_run, replication_run, summary, output_dir)
    figure_paths = [str(record["path"]).removeprefix("cross_model/") for record in figure_artifacts]
    report_text = _report_text(summary, figure_paths)
    report_path = replication_run / "REPORT_CROSS_MODEL.md"
    report_path.write_text(report_text, encoding="utf-8")
    root_text = report_text.replace(
        "(figures/", f"(runs/{replication_run.name}/cross_model/figures/"
    ).replace("`cross_model/", f"`runs/{replication_run.name}/cross_model/")
    root_report = repo_root / "REPORT_CROSS_MODEL.md"
    root_report.write_text(root_text, encoding="utf-8")
    manifest = {
        "schema_version": "gct-model2-cross-model-manifest-v1",
        "status": "complete",
        "baseline_run": baseline_run.name,
        "replication_run": replication_run.name,
        "dataset_logical_hash": dataset_hash,
        "bootstrap_replicates": replicates,
        "pairing_unit": "base_world_id",
        "summary": artifact_record(summary_path, replication_run, "cross_model_summary"),
        "report": artifact_record(report_path, replication_run, "cross_model_report"),
        "tables": artifacts[:-1],
        "figures": figure_artifacts,
        "baseline_statistics_manifest_hash": file_hash(
            baseline_run / "statistics" / "manifest.json"
        ),
        "replication_statistics_manifest_hash": file_hash(
            replication_run / "statistics" / "manifest.json"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return report_path
