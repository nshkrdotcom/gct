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
from gct.storage.hashes import lock_hash
from gct.storage.manifests import write_json_atomic


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
    base: dict[str, Any] = {
        "run_id": config.run_id,
        "protocol_version": config.project.protocol_version,
        "config_hash": config.config_hash,
        "git_commit_at_creation": git_commit(repo_root),
        "dependency_lock_hash": lock_hash(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
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
    write_json_atomic(manifest_path, base)
    return run_dir


def update_run_manifest(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    write_json_atomic(path, manifest)
