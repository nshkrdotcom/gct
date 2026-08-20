"""Preregistered H1-H8 statistics, controls, MDL sensitivity, and exploratory FDR."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from gct.analysis.behavior_link import evaluate_behavior_link
from gct.analysis.bootstrap import BootstrapResult, grouped_bootstrap_mean
from gct.analysis.multiple_testing import benjamini_hochberg
from gct.analysis.pairs import load_layer_dataset, paired_data
from gct.analysis.tables import paired_condition_difference
from gct.config import ExperimentConfig
from gct.metrics.distances import cosine_distance
from gct.provenance import update_run_manifest
from gct.storage.hashes import file_hash
from gct.storage.manifests import artifact_record, read_json, write_json_atomic


def _bootstrap_payload(result: BootstrapResult) -> dict[str, Any]:
    return {
        "estimate": result.estimate,
        "ci_95": [result.lower, result.upper],
        "bootstrap_replicates": result.replicates,
        "base_world_groups": result.groups,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _bootstrap(
    frame: pd.DataFrame, value: str, config: ExperimentConfig, seed_offset: int
) -> BootstrapResult:
    return grouped_bootstrap_mean(
        frame,
        value,
        "base_world_id",
        config.statistics.bootstrap_replicates,
        config.project.seed + seed_offset,
    )


def _h1(transport: pd.DataFrame, config: ExperimentConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    subset = transport[
        (transport["split"] == "test")
        & (transport["world_variant"] == "primary")
        & (transport["coordinate_condition"] == "explicit_coordinate")
        & (transport["model_type"] == "identity")
        & (transport["metric"] == "whitened_l2")
        & transport["transform_name"].isin(
            ["nuisance_rewrite", "pressure_shift", "concentration_shift", "fluid_swap"]
        )
    ].copy()
    subset["kind"] = np.where(
        subset["transform_name"] == "nuisance_rewrite", "nuisance", "substantive"
    )
    pivot = (
        subset.groupby(["base_world_id", "kind"], observed=True)["raw_defect"]
        .mean()
        .unstack("kind")
        .dropna()
    )
    pivot["nuisance_minus_substantive"] = pivot["nuisance"] - pivot["substantive"]
    effect = pivot.reset_index()
    result = _bootstrap(effect, "nuisance_minus_substantive", config, 1)
    supported = result.upper < 0
    return (
        {
            "title": config.preregistration["H1"].title,
            "endpoint": "held-out whitened displacement: nuisance minus substantive",
            **_bootstrap_payload(result),
            "status": "supported" if supported else "not_supported",
        },
        effect,
    )


def _reusable_effect(
    transport: pd.DataFrame, config: ExperimentConfig, world_variant: str, seed_offset: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_filter = (
        (transport["world_variant"] == world_variant)
        & (transport["coordinate_condition"] == "explicit_coordinate")
        & (transport["transform_name"] == "pressure_shift")
        & (transport["metric"] == "whitened_l2")
    )
    validation = transport[base_filter & (transport["split"] == "validation")]
    simple = validation[validation["model_type"].isin(["identity", "mean_shift"])]
    baseline = str(simple.groupby("model_type")["raw_defect"].mean().idxmin())
    test = transport[
        base_filter
        & (transport["split"] == "test")
        & transport["model_type"].isin([baseline, "low_rank_residual"])
    ]
    pivot = (
        test.groupby(["base_world_id", "model_type"], observed=True)["raw_defect"]
        .mean()
        .unstack("model_type")
        .dropna()
    )
    pivot["improvement"] = 1.0 - pivot["low_rank_residual"] / pivot[baseline].clip(lower=1e-12)
    effect = pivot.reset_index()
    result = _bootstrap(effect, "improvement", config, seed_offset)
    return (
        {
            "endpoint": f"held-out low-rank improvement over validation-selected {baseline}",
            "baseline_selected_on": "validation",
            "baseline": baseline,
            **_bootstrap_payload(result),
            "status": "supported" if result.lower > 0 else "not_supported",
        },
        effect,
    )


def _h3(
    transport: pd.DataFrame,
    squares: pd.DataFrame,
    composition: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    subset = transport[
        (transport["split"] == "test")
        & (transport["world_variant"] == "primary")
        & (transport["coordinate_condition"] == "explicit_coordinate")
        & (transport["metric"] == "whitened_l2")
        & transport["transform_name"].isin(["pressure_shift", "concentration_shift"])
        & transport["model_type"].isin(["identity", "continuous_generator"])
    ]
    pivot = (
        subset.groupby(["base_world_id", "model_type"], observed=True)["raw_defect"]
        .mean()
        .unstack("model_type")
        .dropna()
    )
    pivot["improvement"] = 1.0 - pivot["continuous_generator"] / pivot["identity"].clip(lower=1e-12)
    effect = pivot.reset_index()
    result = _bootstrap(effect, "improvement", config, 3)
    square = squares[
        (squares["split"] == "test")
        & (squares["world_variant"] == "primary")
        & (squares["metric"] == "whitened_l2")
    ]
    mismatch = float(square["commuting_square_defect"].mean()) if len(square) else float("nan")
    route_error = (
        float(square[["route_pm_target_defect", "route_mp_target_defect"]].to_numpy(float).mean())
        if len(square)
        else float("nan")
    )
    composition_subset = composition[
        (composition["split"] == "test")
        & (composition["world_variant"] == "primary")
        & (composition["coordinate_condition"] == "explicit_coordinate")
        & (composition["metric"] == "whitened_l2")
    ]
    composition_defect = (
        float(composition_subset["composition_defect"].mean())
        if len(composition_subset)
        else float("nan")
    )
    composition_target = (
        float(composition_subset["composed_target_defect"].mean())
        if len(composition_subset)
        else float("nan")
    )
    supported = result.lower > 0 and math.isfinite(mismatch) and mismatch < route_error
    return (
        {
            "title": config.preregistration["H3"].title,
            "endpoint": "held-out generator improvement and commuting-route mismatch",
            **_bootstrap_payload(result),
            "mean_commuting_square_defect": mismatch,
            "mean_route_to_target_defect": route_error,
            "mean_generator_composition_defect": composition_defect,
            "mean_composed_route_to_target_defect": composition_target,
            "status": "supported" if supported else "not_supported",
        },
        effect,
    )


def _probe_record(probes: dict[str, Any], world: str, condition: str) -> dict[str, Any] | None:
    return next(
        (
            record
            for record in probes["probes"]
            if record["world_variant"] == world and record["coordinate_condition"] == condition
        ),
        None,
    )


def _base_lift(
    transport: pd.DataFrame,
    behavior: pd.DataFrame,
    config: ExperimentConfig,
    world_variant: str,
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    structural = transport[
        (transport["split"] == "test")
        & (transport["world_variant"] == world_variant)
        & (transport["transform_name"] == "pressure_shift")
        & (transport["model_type"] == "low_rank_residual")
        & (transport["metric"] == "whitened_l2")
    ]
    explicit_struct = paired_condition_difference(
        structural,
        left_condition="inferable_unnamed_coordinate",
        right_condition="explicit_coordinate",
        value="raw_defect",
        output_name="explicit_gain",
    )
    q_struct = paired_condition_difference(
        structural,
        left_condition="inferable_unnamed_coordinate",
        right_condition="irrelevant_coordinate",
        value="raw_defect",
        output_name="q_gain",
    )
    explicit_struct_result = _bootstrap(explicit_struct, "explicit_gain", config, seed_offset)
    q_struct_result = _bootstrap(q_struct, "q_gain", config, seed_offset + 1)
    behavior_subset = behavior[
        (behavior["split"] == "test")
        & (behavior["world_variant"] == world_variant)
        & (behavior["transform_name"] == "pressure_shift")
    ].dropna(subset=["absolute_oracle_error"])
    explicit_behavior = paired_condition_difference(
        behavior_subset,
        left_condition="inferable_unnamed_coordinate",
        right_condition="explicit_coordinate",
        value="absolute_oracle_error",
        output_name="explicit_gain",
    )
    q_behavior = paired_condition_difference(
        behavior_subset,
        left_condition="inferable_unnamed_coordinate",
        right_condition="irrelevant_coordinate",
        value="absolute_oracle_error",
        output_name="q_gain",
    )
    if len(explicit_behavior["base_world_id"].unique()) >= 2:
        explicit_behavior_result = _bootstrap(
            explicit_behavior, "explicit_gain", config, seed_offset + 2
        )
        q_behavior_result = _bootstrap(q_behavior, "q_gain", config, seed_offset + 3)
        behavior_payload: dict[str, Any] = {
            "explicit": _bootstrap_payload(explicit_behavior_result),
            "irrelevant_q": _bootstrap_payload(q_behavior_result),
        }
        behavior_pass = (
            explicit_behavior_result.lower > 0
            and explicit_behavior_result.estimate > q_behavior_result.estimate
        )
    else:
        behavior_payload = {"status": "inconclusive_due_to_parse_failures"}
        behavior_pass = False
    structural_pass = (
        explicit_struct_result.lower > 0
        and explicit_struct_result.estimate > q_struct_result.estimate
    )
    return (
        {
            "endpoint": "explicit-coordinate gain versus inferable omission and irrelevant-Q lift",
            "estimate": explicit_struct_result.estimate,
            "ci_95": [explicit_struct_result.lower, explicit_struct_result.upper],
            "structural_gain": {
                "explicit": _bootstrap_payload(explicit_struct_result),
                "irrelevant_q": _bootstrap_payload(q_struct_result),
            },
            "behavioral_gain": behavior_payload,
            "status": "supported" if structural_pass and behavior_pass else "not_supported",
        },
        {
            "explicit_structural": explicit_struct,
            "q_structural": q_struct,
            "explicit_behavior": explicit_behavior,
            "q_behavior": q_behavior,
        },
    )


def _mdl_table(
    transport: pd.DataFrame,
    operator_manifest: dict[str, Any],
    config: ExperimentConfig,
) -> pd.DataFrame:
    subset = transport[
        (transport["split"] == "test")
        & (transport["transform_name"] == "pressure_shift")
        & (transport["model_type"] == "low_rank_residual")
        & (transport["metric"] == "whitened_l2")
    ]
    means = (
        subset.groupby(["world_variant", "coordinate_condition"], observed=True)["raw_defect"]
        .mean()
        .reset_index()
    )
    capacities = {
        (
            str(record["operator_group"]).split("|")[0],
            str(record["operator_group"]).split("|")[1],
        ): float(record["capacity"]["effective_parameters"])
        for record in operator_manifest["operators"]
        if record["model_type"] == "low_rank_residual"
        and str(record["operator_group"]).endswith("|pressure_shift")
    }
    max_parameters = max(capacities.values(), default=1.0)
    output: list[dict[str, Any]] = []
    for world, world_frame in means.groupby("world_variant", observed=True):
        scale = max(float(world_frame["raw_defect"].max()), 1e-12)
        for row in world_frame.itertuples(index=False):
            added = (
                1
                if row.coordinate_condition in {"explicit_coordinate", "irrelevant_coordinate"}
                else 0
            )
            parameter_fraction = (
                capacities.get((str(world), str(row.coordinate_condition)), 0) / max_parameters
            )
            complexity = float(added + parameter_fraction)
            normalized = float(float(cast(Any, row.raw_defect)) / scale)
            for lambda_value in config.mdl.lambda_values:
                output.append(
                    {
                        "world_variant": str(world),
                        "coordinate_condition": str(row.coordinate_condition),
                        "lambda": lambda_value,
                        "normalized_heldout_defect": normalized,
                        "added_coordinates": added,
                        "operator_parameter_fraction": parameter_fraction,
                        "complexity": complexity,
                        "mdl_score": normalized + lambda_value * complexity,
                    }
                )
    return pd.DataFrame(output)


def _behavior_primary_summary(
    run_dir: Path, behavior_edges: pd.DataFrame, config: ExperimentConfig
) -> pd.DataFrame:
    raw = pd.read_parquet(run_dir / "behavior" / "results.parquet")
    metadata = pd.read_parquet(
        run_dir / "dataset" / "samples.parquet",
        columns=["sample_id", "base_world_id", "split", "world_variant", "coordinate_condition"],
    )
    sample_behavior = raw.merge(metadata, on=["sample_id", "base_world_id", "split"], how="left")
    output: list[dict[str, Any]] = []

    def summarize(frame: pd.DataFrame, column: str, metric: str, seed: int, scope: str) -> None:
        clean = frame.dropna(subset=[column]).copy()
        if clean["base_world_id"].nunique() < 2:
            return
        clean[column] = clean[column].astype(float)
        result = _bootstrap(clean, column, config, seed)
        output.append(
            {
                "split": "test",
                "scope": scope,
                "metric": metric,
                **_bootstrap_payload(result),
                "rows": len(clean),
            }
        )

    scopes: list[tuple[str, pd.DataFrame]] = [("all", sample_behavior)]
    scopes.extend(
        (f"{world}|{condition}", group)
        for (world, condition), group in sample_behavior.groupby(
            ["world_variant", "coordinate_condition"], observed=True
        )
    )
    for scope_index, (scope, scoped_samples) in enumerate(scopes):
        test_samples = scoped_samples[scoped_samples["split"] == "test"].copy()
        summarize(
            test_samples,
            "absolute_error",
            "mean_absolute_oracle_error_parsed",
            4000 + scope_index * 10,
            scope,
        )
        summarize(
            test_samples,
            "within_tolerance",
            "within_tolerance_rate_all_prompts",
            4001 + scope_index * 10,
            scope,
        )

    nuisance = behavior_edges[
        (behavior_edges["split"] == "test") & (behavior_edges["transform_family"] == "nuisance")
    ]
    substantive = behavior_edges[
        (behavior_edges["split"] == "test") & (behavior_edges["transform_family"] == "substantive")
    ]
    summarize(nuisance, "answer_flip", "nuisance_answer_flip_rate_parsed_pairs", 4900, "all")
    summarize(
        substantive,
        "correction",
        "substantive_correction_rate_parsed_pairs",
        4901,
        "all",
    )
    return pd.DataFrame(output)


def _exploratory_layer_scan(run_dir: Path, config: ExperimentConfig) -> pd.DataFrame:
    activation_manifest = read_json(run_dir / "activations" / "manifest.json")
    layers = [int(value) for value in activation_manifest["shards"][0]["layer_numbers"]]
    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for layer in layers:
        frame, values = load_layer_dataset(run_dir, layer)
        pairs = paired_data(frame, values, split="test")
        metadata = pairs.metadata
        mask = (
            (metadata["world_variant"].to_numpy() == "primary")
            & (metadata["coordinate_condition"].to_numpy() == "explicit_coordinate")
            & metadata["transform_name"]
            .isin(["nuisance_rewrite", "pressure_shift", "concentration_shift", "fluid_swap"])
            .to_numpy()
        )
        selected = metadata.loc[mask].copy()
        selected["distance"] = cosine_distance(pairs.source[mask], pairs.target[mask])
        selected["kind"] = np.where(
            selected["transform_name"] == "nuisance_rewrite", "nuisance", "substantive"
        )
        pivot = (
            selected.groupby(["base_world_id", "kind"], observed=True)["distance"]
            .mean()
            .unstack("kind")
            .dropna()
        )
        pivot["effect"] = pivot["nuisance"] - pivot["substantive"]
        result = _bootstrap(pivot.reset_index(), "effect", config, 1000 + layer)
        p_lower = (1 + int(np.sum(result.draws <= 0))) / (1 + len(result.draws))
        p_upper = (1 + int(np.sum(result.draws >= 0))) / (1 + len(result.draws))
        p_value = min(1.0, 2 * min(p_lower, p_upper))
        raw_p.append(p_value)
        rows.append(
            {
                "layer": layer,
                "split": "test_exploratory",
                "metric": "cosine",
                "nuisance_minus_substantive": result.estimate,
                "ci_lower": result.lower,
                "ci_upper": result.upper,
                "raw_p": p_value,
            }
        )
    adjusted = benjamini_hochberg(np.array(raw_p))
    for row, value in zip(rows, adjusted, strict=True):
        row["fdr_adjusted_p"] = float(value)
        row["fdr_alpha"] = config.statistics.fdr_alpha
    return pd.DataFrame(rows)


def run_statistics(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    metrics_manifest = read_json(run_dir / "metrics" / "manifest.json")
    probe_manifest = read_json(run_dir / "probes" / "manifest.json")
    operator_manifest = read_json(run_dir / "operators" / "manifest.json")
    if metrics_manifest.get("status") != "complete" or probe_manifest.get("status") != "complete":
        raise ValueError("metrics and probes must be complete before statistics")
    transport = pd.read_parquet(run_dir / "metrics" / "transport_edges.parquet")
    squares = pd.read_parquet(run_dir / "metrics" / "squares.parquet")
    composition = pd.read_parquet(run_dir / "metrics" / "generator_composition.parquet")
    behavior = pd.read_parquet(run_dir / "metrics" / "behavior_edges.parquet")
    h1, h1_effect = _h1(transport, config)
    h2, h2_effect = _reusable_effect(transport, config, "primary", 2)
    h2["title"] = config.preregistration["H2"].title
    h3, h3_effect = _h3(transport, squares, composition, config)
    behavior_link, behavior_predictions = evaluate_behavior_link(transport, behavior)
    link_by_name = {row["feature_set"]: row for row in behavior_link}
    h4_comparable = "confounds_only" in link_by_name and "confounds_plus_defect" in link_by_name
    h4_supported = h4_comparable and (
        link_by_name["confounds_plus_defect"]["r2"] > link_by_name["confounds_only"]["r2"]
    )
    prediction_pivot = pd.DataFrame()
    h4_gain: BootstrapResult | None = None
    if not behavior_predictions.empty:
        prediction_pivot = (
            behavior_predictions.pivot(
                index=["sample_id", "base_world_id"],
                columns="feature_set",
                values="absolute_prediction_error",
            )
            .dropna()
            .reset_index()
        )
        prediction_pivot["absolute_prediction_error_gain"] = (
            prediction_pivot["confounds_only"] - prediction_pivot["confounds_plus_defect"]
        )
        h4_gain = _bootstrap(prediction_pivot, "absolute_prediction_error_gain", config, 44)
    h4 = {
        "title": config.preregistration["H4"].title,
        "endpoint": "held-out absolute-error prediction versus trivial confounds",
        "models": behavior_link,
        "r2_gain_over_confounds": (
            link_by_name["confounds_plus_defect"]["r2"] - link_by_name["confounds_only"]["r2"]
            if h4_comparable
            else None
        ),
        "grouped_absolute_prediction_error_gain": (
            _bootstrap_payload(h4_gain) if h4_gain is not None else None
        ),
        "status": (
            "supported" if h4_supported else "not_supported" if behavior_link else "inconclusive"
        ),
    }
    inferable = _probe_record(probe_manifest, "primary", "inferable_unnamed_coordinate")
    unobservable = _probe_record(probe_manifest, "primary", "unobservable_coordinate")
    if inferable is None or unobservable is None:
        raise ValueError("required primary hidden-coordinate probe records are missing")
    h5 = {
        "title": config.preregistration["H5"].title,
        "endpoint": "held-out inferable-arm residual pressure recovery",
        "test_r2": inferable["test_r2"],
        "test_mae": inferable["test_mae"],
        "permutation_p_value": inferable["permutation_p_value"],
        "null_r2_95th_percentile": inferable["null_r2_95th_percentile"],
        "test_r2_ci_95": inferable["test_r2_ci_95"],
        "test_mae_ci_95": inferable["test_mae_ci_95"],
        "status": (
            "supported"
            if inferable["test_r2"] > 0
            and inferable["permutation_p_value"] <= config.statistics.test_alpha
            else "not_supported"
        ),
    }
    h6_pass = unobservable["test_r2"] <= unobservable["null_r2_95th_percentile"]
    h6 = {
        "title": config.preregistration["H6"].title,
        "endpoint": "held-out identical-prompt unobservable pressure recovery",
        "test_r2": unobservable["test_r2"],
        "test_mae": unobservable["test_mae"],
        "permutation_p_value": unobservable["permutation_p_value"],
        "null_r2_95th_percentile": unobservable["null_r2_95th_percentile"],
        "test_r2_ci_95": unobservable["test_r2_ci_95"],
        "test_mae_ci_95": unobservable["test_mae_ci_95"],
        "status": "control_pass" if h6_pass else "control_fail_investigate_leakage",
    }
    h7, h7_tables = _base_lift(transport, behavior, config, "primary", 70)
    h7["title"] = config.preregistration["H7"].title
    alias_h2, _ = _reusable_effect(transport, config, "renamed", 80)
    alias_h5_record = _probe_record(probe_manifest, "renamed", "inferable_unnamed_coordinate")
    alias_h7, _ = _base_lift(transport, behavior, config, "renamed", 90)
    if alias_h5_record is None:
        raise ValueError("renamed inferable probe is missing")
    alias_h5_supported = (
        alias_h5_record["test_r2"] > 0
        and alias_h5_record["permutation_p_value"] <= config.statistics.test_alpha
    )
    h8_supported = (
        alias_h2["status"] == "supported"
        and alias_h5_supported
        and alias_h7["status"] == "supported"
    )
    h8 = {
        "title": config.preregistration["H8"].title,
        "endpoint": "renamed-world repetition of H2, H5, and H7",
        "H2": alias_h2,
        "H5": {
            "test_r2": alias_h5_record["test_r2"],
            "test_mae": alias_h5_record["test_mae"],
            "test_r2_ci_95": alias_h5_record["test_r2_ci_95"],
            "test_mae_ci_95": alias_h5_record["test_mae_ci_95"],
            "permutation_p_value": alias_h5_record["permutation_p_value"],
            "status": "supported" if alias_h5_supported else "not_supported",
        },
        "H7": alias_h7,
        "status": "supported" if h8_supported else "not_supported",
    }
    if not config.reporting.scientific_claims_allowed:
        hypotheses_to_mark: list[dict[str, Any]] = [h1, h2, h3, h4, h5, h6, h7, h8]
        for hypothesis in hypotheses_to_mark:
            hypothesis["development_status"] = hypothesis["status"]
            hypothesis["status"] = "development_only"
    hypotheses = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7, "H8": h8}
    output_dir = run_dir / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "h1_effect": h1_effect,
        "h2_effect": h2_effect,
        "h3_effect": h3_effect,
        "behavior_link": pd.DataFrame(behavior_link),
        "behavior_link_predictions": behavior_predictions,
        "behavior_link_grouped_effect": prediction_pivot,
        "behavior_primary_summary": _behavior_primary_summary(run_dir, behavior, config),
        "mdl_sensitivity": _mdl_table(transport, operator_manifest, config),
        "exploratory_layer_scan": _exploratory_layer_scan(run_dir, config),
        **{f"h7_{name}": table for name, table in h7_tables.items()},
    }
    artifacts: list[dict[str, Any]] = []
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False, compression="zstd")
        artifacts.append(artifact_record(path, run_dir, f"statistics_{name}"))
    summary_path = output_dir / "hypotheses.json"
    write_json_atomic(
        summary_path,
        {
            "schema_version": "gct-hypotheses-v1",
            "config_hash": config.config_hash,
            "evaluation_split": "held-out test",
            "hypotheses": _json_safe(hypotheses),
        },
    )
    manifest = {
        "schema_version": "gct-statistics-v1",
        "status": "complete",
        "config_hash": config.config_hash,
        "bootstrap_replicates": config.statistics.bootstrap_replicates,
        "permutation_replicates": config.statistics.permutation_replicates,
        "hypotheses": artifact_record(summary_path, run_dir, "hypothesis_summary"),
        "tables": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    update_run_manifest(
        run_dir, statistics_manifest_hash=file_hash(manifest_path), status="statistics_complete"
    )
    return run_dir
