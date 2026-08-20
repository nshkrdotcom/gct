"""Figures derived exclusively from recorded metric/statistic tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gct.storage.manifests import artifact_record


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_figures(run_dir: Path) -> list[dict[str, Any]]:
    output = run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    selection = pd.read_parquet(run_dir / "operators" / "validation_selection.parquet")
    selection_best = selection.groupby("layer", observed=True)["whitened_l2"].min()
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        selection_best.index.to_numpy(),
        selection_best.to_numpy(dtype=float),
        marker="o",
        linewidth=1.5,
    )
    axis.set(
        title="Validation-only layer selection",
        xlabel="Transformer layer",
        ylabel="Whitened L2 defect",
    )
    path = output / "validation_layer_selection.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, run_dir, "figure_validation_layer_selection"))

    transport = pd.read_parquet(run_dir / "metrics" / "transport_edges.parquet")
    comparison = (
        transport[
            (transport["split"] == "test")
            & (transport["world_variant"] == "primary")
            & (transport["coordinate_condition"] == "explicit_coordinate")
            & (transport["transform_name"] == "pressure_shift")
            & (transport["metric"] == "whitened_l2")
        ]
        .groupby("model_type", observed=True)["raw_defect"]
        .mean()
        .sort_values()
    )
    fig, axis = plt.subplots(figsize=(8, 4))
    comparison.plot.bar(ax=axis, color="#35618f")
    axis.set(title="Held-out test transport defect", xlabel="Operator", ylabel="Mean whitened L2")
    axis.tick_params(axis="x", rotation=30)
    path = output / "test_transport_models.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, run_dir, "figure_test_transport_models"))

    probe_manifest = __import__("json").loads(
        (run_dir / "probes" / "manifest.json").read_text(encoding="utf-8")
    )
    probes = pd.DataFrame(probe_manifest["probes"])
    probes["label"] = probes["world_variant"] + "\n" + probes["coordinate_condition"]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(probes["label"], probes["test_r2"], color="#4c956c", label="Observed test R²")
    axis.scatter(
        probes["label"],
        probes["null_r2_95th_percentile"],
        color="#b23a48",
        marker="_",
        s=220,
        label="Permutation null 95th percentile",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(title="Held-out test hidden-pressure recovery", ylabel="R²", xlabel="Arm")
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    path = output / "test_hidden_pressure_probes.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, run_dir, "figure_test_hidden_pressure_probes"))

    lift = (
        transport[
            (transport["split"] == "test")
            & (transport["transform_name"] == "pressure_shift")
            & (transport["model_type"] == "low_rank_residual")
            & (transport["metric"] == "whitened_l2")
        ]
        .groupby(["world_variant", "coordinate_condition"], observed=True)["raw_defect"]
        .mean()
        .unstack(0)
    )
    fig, axis = plt.subplots(figsize=(9, 4.5))
    lift.plot.bar(ax=axis)
    axis.set(
        title="Held-out test base-lift structural comparison",
        xlabel="Coordinate condition",
        ylabel="Mean whitened L2",
    )
    axis.tick_params(axis="x", rotation=25)
    path = output / "test_base_lift.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, run_dir, "figure_test_base_lift"))

    layer_scan = pd.read_parquet(run_dir / "statistics" / "exploratory_layer_scan.parquet")
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(layer_scan["layer"], layer_scan["nuisance_minus_substantive"], marker="o")
    axis.axhline(0, color="black", linewidth=0.8)
    significant = layer_scan[layer_scan["fdr_adjusted_p"] <= layer_scan["fdr_alpha"]]
    axis.scatter(
        significant["layer"],
        significant["nuisance_minus_substantive"],
        color="#b23a48",
        label="BH-FDR significant",
    )
    axis.set(
        title="Held-out test exploratory layer scan (BH-FDR corrected)",
        xlabel="Transformer layer",
        ylabel="Cosine displacement: nuisance − substantive",
    )
    if len(significant):
        axis.legend()
    path = output / "test_exploratory_layer_scan.png"
    _save(fig, path)
    artifacts.append(artifact_record(path, run_dir, "figure_test_layer_scan"))
    return artifacts
