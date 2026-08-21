"""Run-level provenance and environment capture."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from gct.config import ExperimentConfig, write_config_snapshot
from gct.storage.hashes import canonical_hash, file_hash, lock_hash
from gct.storage.manifests import read_json, write_json_atomic


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def dependency_versions() -> dict[str, str]:
    packages = [
        "torch",
        "transformers",
        "huggingface-hub",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "safetensors",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def initialize_run(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    manifest_path = run_dir / "manifest.json"
    if config_path.exists():
        from gct.config import load_config

        existing = load_config(config_path)
        if existing.config_hash != config.config_hash:
            raise ValueError("run directory contains a different config hash")
    else:
        write_config_snapshot(config, config_path)
    current_commit = git_commit(repo_root)
    current_lock_hash = lock_hash(repo_root)
    current_dependencies = dependency_versions()
    base: dict[str, Any] = {
        "run_id": config.run_id,
        "protocol_version": config.project.protocol_version,
        "config_hash": config.config_hash,
        "git_commit_at_creation": current_commit,
        "dependency_lock_hash": current_lock_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": current_dependencies,
        "model": config.model.model_dump(mode="json"),
        "seed": config.project.seed,
        "status": "initialized",
        "artifacts": [],
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("config_hash") != config.config_hash:
            raise ValueError("manifest config hash differs from requested config")
        base.update(existing_manifest)
        if not (run_dir / "activations" / "manifest.json").exists():
            # A pre-extraction engineering retry may repair its runtime without
            # scientific artifacts. Freeze the repaired environment at the first
            # activation manifest rather than retaining a failed import attempt.
            base["dependency_lock_hash"] = current_lock_hash
            base["dependencies"] = current_dependencies
        base["git_commit_at_last_resume"] = current_commit
        base["dependency_lock_hash_at_last_resume"] = current_lock_hash
    write_json_atomic(manifest_path, base)
    return run_dir


def update_run_manifest(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    write_json_atomic(path, manifest)


def freeze_model2_preregistration(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    path = run_dir / "preregistration_frozen.json"
    adapter_audit_path = run_dir / "model_adapter" / "anchor_audit.json"
    operational_probe_path = run_dir / "model_adapter" / "operational_probe.json"
    if not adapter_audit_path.is_file():
        raise ValueError("full model-adapter audit must precede Model #2 preregistration freeze")
    if not operational_probe_path.is_file():
        raise ValueError("operational batch probe must precede Model #2 preregistration freeze")
    payload: dict[str, Any] = {
        "schema_version": "gct-model2-preregistration-v2",
        "status": "frozen",
        "config_hash": config.config_hash,
        "config_snapshot_sha256": file_hash(run_dir / "config.yaml"),
        "dataset_manifest_sha256": file_hash(run_dir / "dataset" / "manifest.json"),
        "model_adapter_audit_sha256": file_hash(adapter_audit_path),
        "operational_probe_sha256": file_hash(operational_probe_path),
        "handoff_model2_protocol_sha256": (
            "82436a66ac3b398d2998aa9d946a92da6f0ae0456d6e893b9a6da228412895b4"
        ),
        "hypotheses": {
            name: hypothesis.model_dump(mode="json")
            for name, hypothesis in sorted(config.preregistration.items())
        },
        "fit_split": "train",
        "selection_split": "validation",
        "evaluation_split": "test",
        "test_data_used": False,
        "test_metrics_viewed": False,
    }
    payload["freeze_hash"] = canonical_hash(payload)
    if path.is_file():
        if read_json(path) != payload:
            raise ValueError(
                "existing Model #2 preregistration freeze differs from current protocol"
            )
        return run_dir
    write_json_atomic(path, payload)
    update_run_manifest(run_dir, preregistration_freeze_hash=payload["freeze_hash"])
    return run_dir
