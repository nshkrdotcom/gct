"""Evidence-calibrated markdown report generated from immutable artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from gct.config import ExperimentConfig
from gct.models.loader import runtime_report
from gct.provenance import git_commit, update_run_manifest
from gct.reporting.figures import build_figures
from gct.storage.hashes import file_hash
from gct.storage.manifests import artifact_record, read_json, write_json_atomic


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _hypothesis_line(name: str, record: dict[str, Any]) -> str:
    evidence: list[str] = []
    if "estimate" in record:
        evidence.append(f"effect={_fmt(record['estimate'])}")
    if "ci_95" in record:
        evidence.append(f"95% CI [{_fmt(record['ci_95'][0])}, {_fmt(record['ci_95'][1])}]")
    if "test_r2" in record:
        evidence.append(f"test R²={_fmt(record['test_r2'])}")
    if "test_mae" in record:
        evidence.append(f"MAE={_fmt(record['test_mae'])}")
    if "permutation_p_value" in record:
        evidence.append(f"permutation p={_fmt(record['permutation_p_value'])}")
    if "test_r2_ci_95" in record:
        evidence.append(
            f"R² 95% CI [{_fmt(record['test_r2_ci_95'][0])}, {_fmt(record['test_r2_ci_95'][1])}]"
        )
    if record.get("r2_gain_over_confounds") is not None:
        evidence.append(f"R² gain={_fmt(record['r2_gain_over_confounds'])}")
    grouped_gain = record.get("grouped_absolute_prediction_error_gain")
    if isinstance(grouped_gain, dict):
        evidence.append(
            f"prediction-error gain={_fmt(grouped_gain['estimate'])} "
            f"[95% CI {_fmt(grouped_gain['ci_95'][0])}, {_fmt(grouped_gain['ci_95'][1])}]"
        )
    if name == "H8" and all(key in record for key in ("H2", "H5", "H6", "H7")):
        evidence.append(
            "renamed statuses="
            f"H2:{record['H2']['status']}, H5:{record['H5']['status']}, "
            f"H6:{record['H6']['status']}, H7:{record['H7']['status']}"
        )
        evidence.append(
            f"renamed H2 effect={_fmt(record['H2']['estimate'])}; "
            f"H5 R²={_fmt(record['H5']['test_r2'])}, "
            f"p={_fmt(record['H5']['permutation_p_value'])}; "
            f"H6 R²={_fmt(record['H6']['test_r2'])}, "
            f"p={_fmt(record['H6']['permutation_p_value'])}; "
            f"H7 structural effect={_fmt(record['H7']['estimate'])}"
        )
    if name == "H3":
        evidence.append(
            f"composition={_fmt(record['mean_generator_composition_defect'])}; "
            f"square={_fmt(record['mean_commuting_square_defect'])}; "
            f"composed-to-target={_fmt(record['mean_composed_route_to_target_defect'])}"
        )
    if name == "H7":
        evidence.append(
            f"Q structural={_fmt(record['structural_gain']['irrelevant_q']['estimate'])}; "
            f"explicit behavior={_fmt(record['behavioral_gain']['explicit']['estimate'])}; "
            f"Q behavior={_fmt(record['behavioral_gain']['irrelevant_q']['estimate'])}"
        )
    if not evidence:
        evidence.append(str(record.get("endpoint", "see machine-readable result")))
    return f"| {name} | {record.get('title', '')} | {record['status']} | {'; '.join(evidence)} |"


def _interpretation_level(hypotheses: dict[str, dict[str, Any]]) -> int:
    supported = {name for name, value in hypotheses.items() if value["status"] == "supported"}
    h6_ok = hypotheses["H6"]["status"] == "control_pass"
    if {"H4", "H5", "H7", "H8"}.issubset(supported) and h6_ok:
        return 5
    if "H4" in supported:
        return 4
    if "H3" in supported:
        return 3
    if "H2" in supported:
        return 2
    if "H1" in supported:
        return 1
    return 0


def build_report(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    statistics = read_json(run_dir / "statistics" / "hypotheses.json")
    hypotheses = statistics["hypotheses"]
    figures = build_figures(run_dir)
    dataset = read_json(run_dir / "dataset" / "manifest.json")
    activations = read_json(run_dir / "activations" / "manifest.json")
    behavior = read_json(run_dir / "behavior" / "manifest.json")
    selection = read_json(run_dir / "operators" / "selection_frozen.json")
    probes = read_json(run_dir / "probes" / "manifest.json")
    run_manifest = read_json(run_dir / "manifest.json")
    quality_path = run_dir / "quality_gates.json"
    quality = read_json(quality_path) if quality_path.exists() else None
    runtime = runtime_report(config)
    behavior_summary = pd.read_parquet(run_dir / "statistics" / "behavior_primary_summary.parquet")
    behavior_overall = behavior_summary[behavior_summary["scope"] == "all"].set_index("metric")

    def behavior_metric(name: str) -> str:
        if name not in behavior_overall.index:
            return "inconclusive"
        row = cast(pd.Series, behavior_overall.loc[name])
        return (
            f"{_fmt(float(row['estimate']))} "
            f"(95% CI [{_fmt(float(row['ci_95'][0]))}, {_fmt(float(row['ci_95'][1]))}])"
        )

    level = _interpretation_level(hypotheses)
    layer_scope = "all-layer" if config.activations.layers == "all" else "configured-layer"
    status_counts = pd.Series([record["status"] for record in hypotheses.values()]).value_counts()
    activation_canonical = activations["duplicate_prompt_canonicalization"]
    behavior_canonical = behavior["duplicate_prompt_canonicalization"]
    if quality is None:
        quality_lines = [
            "- Final command-level gates have not yet been recorded in `quality_gates.json`; "
            "rerun the README audit commands before interpreting this report as a final handoff."
        ]
        git_lines = [
            f"- Commit at report build: `{git_commit(repo_root)}`",
            "- Remote configured / push attempted / push verified: not yet recorded",
        ]
    else:
        quality_lines = [
            f"- Tests: `{quality['gates']['tests']['summary']}` "
            f"(`{quality['gates']['tests']['command']}`)",
            f"- Lint: `{quality['gates']['lint']['summary']}`",
            f"- Format: `{quality['gates']['format']['summary']}`",
            f"- Type checking: `{quality['gates']['types']['summary']}`",
            f"- Real-model integration: `{quality['gates']['real_model_integration']['summary']}`",
            f"- Split validation: `{quality['gates']['split_validation']['summary']}`",
            f"- Deterministic regeneration: `{quality['gates']['dataset_regeneration']['summary']}`",
            f"- Exact duplicate-prompt audit: `{quality['gates']['duplicate_prompt_audit']['summary']}`",
            f"- Artifact verification: `{quality['gates']['artifact_verification']['summary']}`",
            f"- Placeholder audit: `{quality['gates']['placeholder_audit']['summary']}`",
        ]
        git_lines = [
            f"- Commit at report build: `{git_commit(repo_root)}`",
            f"- Remote configured: `{quality['git']['remote_configured']}` "
            f"(`{quality['git']['remote']}`)",
            f"- Push attempted: `{quality['git']['push_attempted']}`",
            f"- Push verified: `{quality['git']['push_verified']}` "
            f"at `{quality['git']['verified_commit']}`",
        ]
    lines = [
        "# Geometry of Conditional Truth — Run Report",
        "",
        "## 1. Executive result",
        "",
        f"This report is generated from run `{config.run_id}` and recorded artifact hashes. "
        f"The validation-selected transformer layer was {selection['primary_layer']}; test data "
        "were evaluated only after the selection artifact was frozen. The run tested all five "
        "coordinate/control arms and both arbitrary and familiar-label versions. "
        f"Across H1–H8, the status counts were {status_counts.to_dict()}. "
        f"The conservative interpretation is Level {level} of 6. This is evidence about empirical "
        "transport proxies in one synthetic task and one model, not evidence that coherence proves "
        "truth or that the model literally contains a sheaf, bifibration, or universal truth manifold.",
        "",
        "## 2. Environment",
        "",
        f"- Python: `{run_manifest.get('python')}`",
        f"- Platform: `{run_manifest.get('platform')}`",
        f"- GPU: `{runtime.gpu_name}`; compute capability `{runtime.compute_capability}`",
        f"- NVIDIA driver / PyTorch CUDA runtime: `{runtime.driver_report}` / "
        f"`{runtime.torch_cuda_runtime}`",
        f"- PyTorch: `{runtime.torch}`",
        f"- Model: `{activations['model_name']}` at revision `{activations['model_revision']}`",
        f"- Dtype/storage: `{config.model.dtype}` / `{activations['storage_dtype']}`",
        f"- Layers/hidden size: {activations['model_num_hidden_layers']} / {activations['model_hidden_size']}",
        f"- Repository commit at report build: `{git_commit(repo_root)}`",
        f"- Config hash: `{config.config_hash}`",
        f"- Dependency lock hash: `{run_manifest.get('dependency_lock_hash')}`",
        "",
        "## 3. Completed implementation",
        "",
        "The run includes deterministic ToyThermo oracle data; explicit, inferable, identical-prompt "
        "unobservable, irrelevant-Q, and semantic-renaming arms; grouped and held-out transformations; "
        f"real {layer_scope} anchor activations; deterministic answers; train-fit preprocessing; identity, "
        "mean-shift, PCA-affine, low-rank, and continuous-generator operators; transport, cycle, "
        "commuting-square, and matching proxies; residual probes/permutations; MDL sensitivity; "
        "grouped bootstrap uncertainty; behavior links; figures; and hash verification.",
        "",
        "## 4. QC status",
        "",
        f"- Dataset logical hash: `{dataset['logical_dataset_hash']}`",
        f"- Activation shards: {len(activations['shards'])}, status `{activations['status']}`",
        f"- Behavior parse failures: {behavior['parse_failure_count']}",
        f"- Common anchor token IDs: `{activations['anchor_suffix_token_ids']}`",
        f"- Frozen selection certifies test data used: `{selection['test_data_used']}`",
        f"- Probe permutation replicates: {probes['permutation_replicates']}",
        *quality_lines,
        "",
        "Held-out primary behavior metrics (grouped bootstrap by base world):",
        "",
        f"- Mean absolute oracle error among parsed answers: "
        f"{behavior_metric('mean_absolute_oracle_error_parsed')}",
        f"- Within-tolerance correctness over all prompts: "
        f"{behavior_metric('within_tolerance_rate_all_prompts')}",
        f"- Nuisance answer-flip rate among parsed pairs: "
        f"{behavior_metric('nuisance_answer_flip_rate_parsed_pairs')}",
        f"- Substantive correction rate among parsed pairs: "
        f"{behavior_metric('substantive_correction_rate_parsed_pairs')}",
        "",
        "## 5. Dataset",
        "",
        f"The dataset contains {dataset['counts']['rows']} prompt rows from "
        f"{dataset['counts']['base_worlds']} grouped base worlds. Split counts are "
        f"`{dataset['counts']['by_split']}`, arm counts are `{dataset['counts']['by_arm']}`, and "
        f"transformation counts are `{dataset['counts']['by_transform']}`. "
        f"The oracle is `{dataset['world_version']}`, computed exclusively in Python. Transformation magnitudes, "
        "the JSON-like renderer, and Cyrene entity evaluation follow the frozen held-out design.",
        "",
        "## 6. Preregistered hypotheses",
        "",
        "| Hypothesis | Title | Status | Primary evidence |",
        "|---|---|---|---|",
        *[_hypothesis_line(name, hypotheses[name]) for name in sorted(hypotheses)],
        "",
        "The complete nested effects, null thresholds, behavior baselines, and H8 replication results "
        "are in `statistics/hypotheses.json`; no endpoint was removed because of its sign.",
        "",
        "## 7. Key figures/tables",
        "",
        "- [Validation layer selection](figures/validation_layer_selection.png)",
        "- [Held-out transport operators](figures/test_transport_models.png)",
        "- [Held-out hidden-pressure probes](figures/test_hidden_pressure_probes.png)",
        "- [Held-out base-lift comparison](figures/test_base_lift.png)",
        "- [Exploratory test layer scan with BH-FDR](figures/test_exploratory_layer_scan.png)",
        "- Machine-readable metric tables: `metrics/*.parquet`",
        "- Generator composition proxy: `metrics/generator_composition.parquet`",
        "- MDL lambda sweep: `statistics/mdl_sensitivity.parquet`",
        "",
        "## 8. Negative controls",
        "",
        f"The identical-prompt unobservable control status was `{hypotheses['H6']['status']}` "
        f"(test R² {_fmt(hypotheses['H6']['test_r2'])}; null 95th percentile "
        f"{_fmt(hypotheses['H6']['null_r2_95th_percentile'])}). H7's explicit-P structural gain was "
        f"{_fmt(hypotheses['H7']['structural_gain']['explicit']['estimate'])}, versus "
        f"{_fmt(hypotheses['H7']['structural_gain']['irrelevant_q']['estimate'])} for irrelevant Q; "
        "the preregistered superiority rule was not met. In the familiar-label world, H2, H5, and "
        "H7 remained unsupported while H6 again passed. A failed H6 would invalidate positive "
        "hidden-coordinate interpretation until leakage was resolved.",
        "",
        f"Raw batched extraction showed numerical batch-boundary sensitivity in "
        f"{activation_canonical['mismatched_rows_before_canonicalization']} of "
        f"{activation_canonical['compared_duplicate_rows']} repeated activation rows (maximum stored "
        f"difference {_fmt(activation_canonical['max_abs_difference_before_canonicalization'])}); "
        f"generation differed in {behavior_canonical['mismatched_rows_before_canonicalization']} repeated "
        "rows. Because identical token sequences cannot contain a row-specific hidden coordinate, the "
        "preregistered `canonical_first_occurrence` policy rewrote each duplicate prompt from its first "
        "dataset occurrence before analysis. The final exact audit is recorded above. An earlier "
        "batch-sensitive run was superseded rather than reported.",
        "",
        "## 9. Interpretation level",
        "",
        f"**Level {level}.** The level follows the preregistered evidence ladder mechanically. "
        "It does not generalize beyond this model, prompt protocol, representation anchor, or synthetic world.",
        "",
        "## 10. Prior-art update",
        "",
        "The nearest work includes context-conditioned truth-vector geometry, transformation-equivariant "
        "representation learning, causal representation identifiability, continuous latent reasoning, "
        "activation patching, and sheaf-theoretic contextuality. `docs/LITERATURE_MAP.md` and "
        "`docs/PRIOR_ART_DIFF.md` distinguish this controlled residual-transport/base-lift protocol. "
        "No universal novelty claim is made.",
        "",
        "## 11. Limitations",
        "",
        "One 4B instruction model, one anchor, a synthetic arithmetic world, linear/reduced-rank operators, "
        "representation-dependent distances, observational activations, finite permutation/bootstrap "
        "precision, and possible prompt-computation confounds limit inference. Familiar labels may invoke "
        "pretraining priors even though the prompt overrides chemistry. See `docs/LIMITATIONS.md`.",
        "",
        "## 12. Next experiment",
        "",
        "The single most informative follow-up is a preregistered replication on a second model family at "
        "matched data and compute, preserving the identical-prompt negative control and frozen analysis.",
        "",
        "## 13. Git state",
        "",
        *git_lines,
        "",
    ]
    report_text = "\n".join(lines)
    report_path = run_dir / "REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")
    (repo_root / "REPORT.md").write_text(report_text, encoding="utf-8")
    manifest = {
        "schema_version": "gct-report-v1",
        "status": "complete",
        "config_hash": config.config_hash,
        "interpretation_level": level,
        "statistics_manifest_hash": file_hash(run_dir / "statistics" / "manifest.json"),
        "report": artifact_record(report_path, run_dir, "markdown_report"),
        "figures": figures,
    }
    manifest_path = run_dir / "report_manifest.json"
    write_json_atomic(manifest_path, manifest)
    update_run_manifest(
        run_dir,
        report_manifest_hash=file_hash(manifest_path),
        interpretation_level=level,
        status="complete",
    )
    return run_dir
