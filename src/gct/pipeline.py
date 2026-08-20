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
from gct.storage.hashes import file_hash
from gct.storage.manifests import read_json


def _artifact_records_valid(value: object, run_dir: Path) -> bool:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            path = run_dir / str(value["path"])
            if not path.is_file() or file_hash(path) != value["sha256"]:
                return False
            if "bytes" in value and path.stat().st_size != value["bytes"]:
                return False
        return all(_artifact_records_valid(item, run_dir) for item in value.values())
    if isinstance(value, list):
        return all(_artifact_records_valid(item, run_dir) for item in value)
    return True


def _complete(stage: str, path: Path, config: ExperimentConfig, run_dir: Path) -> bool:
    if not path.is_file():
        return False
    manifest = read_json(path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("config_hash") != config.config_hash
        or not _artifact_records_valid(manifest, run_dir)
    ):
        return False
    dependencies: dict[str, tuple[Path, str]] = {
        "operators": (
            run_dir / "activations" / "manifest.json",
            "activation_manifest_hash",
        ),
        "probes": (run_dir / "operators" / "manifest.json", "operator_manifest_hash"),
        "metrics": (run_dir / "operators" / "manifest.json", "operator_manifest_hash"),
        "statistics": (run_dir / "metrics" / "manifest.json", "metrics_manifest_hash"),
        "report": (run_dir / "statistics" / "manifest.json", "statistics_manifest_hash"),
    }
    dependency = dependencies.get(stage)
    return dependency is None or (
        dependency[0].is_file() and manifest.get(dependency[1]) == file_hash(dependency[0])
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
    if not (
        resume and _complete("operators", run_dir / "operators" / "manifest.json", config, run_dir)
    ):
        fit_transport_operators(config, repo_root)
    if not (resume and _complete("probes", run_dir / "probes" / "manifest.json", config, run_dir)):
        fit_hidden_coordinate_probes(config, repo_root)
    if not (
        resume and _complete("metrics", run_dir / "metrics" / "manifest.json", config, run_dir)
    ):
        evaluate_metrics(config, repo_root)
    if not (
        resume
        and _complete("statistics", run_dir / "statistics" / "manifest.json", config, run_dir)
    ):
        run_statistics(config, repo_root)
    if not (resume and _complete("report", run_dir / "report_manifest.json", config, run_dir)):
        build_report(config, repo_root)
    return run_dir
