"""One-command resumable pipeline with content-verified stage reuse."""

from __future__ import annotations

from pathlib import Path

from gct.analysis.stats import run_statistics
from gct.config import ExperimentConfig
from gct.data.generate import build_dataset, validate_dataset_path
from gct.metrics.evaluate import evaluate_metrics
from gct.operators.fit import fit_transport_operators
from gct.probes.hidden_coordinate import fit_hidden_coordinate_probes
from gct.provenance import initialize_run
from gct.reporting.report import build_report
from gct.storage.activations import evaluate_behavior_shards, extract_activation_shards
from gct.storage.manifests import read_json


def _complete(path: Path, config: ExperimentConfig) -> bool:
    if not path.is_file():
        return False
    manifest = read_json(path)
    return (
        manifest.get("status") == "complete" and manifest.get("config_hash") == config.config_hash
    )


def run_pipeline(config: ExperimentConfig, repo_root: Path, *, resume: bool = False) -> Path:
    run_dir = initialize_run(config, repo_root)
    dataset_manifest = run_dir / "dataset" / "manifest.json"
    if resume and dataset_manifest.is_file():
        validate_dataset_path(run_dir)
    else:
        build_dataset(config, repo_root)
    extract_activation_shards(config, repo_root, resume=resume)
    evaluate_behavior_shards(config, repo_root, resume=resume)
    if not (resume and _complete(run_dir / "operators" / "manifest.json", config)):
        fit_transport_operators(config, repo_root)
    if not (resume and _complete(run_dir / "probes" / "manifest.json", config)):
        fit_hidden_coordinate_probes(config, repo_root)
    if not (resume and _complete(run_dir / "metrics" / "manifest.json", config)):
        evaluate_metrics(config, repo_root)
    if not (resume and _complete(run_dir / "statistics" / "manifest.json", config)):
        run_statistics(config, repo_root)
    if not (resume and _complete(run_dir / "report_manifest.json", config)):
        build_report(config, repo_root)
    return run_dir
