"""End-to-end run verification against hashes and scientific-completeness gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gct.config import load_config
from gct.data.generate import validate_dataset_path
from gct.pipeline import _complete
from gct.storage.hashes import file_hash
from gct.storage.manifests import read_json, verify_artifact


def _verify_record(record: dict[str, Any], run_dir: Path, errors: list[str]) -> None:
    if {"path", "sha256", "bytes"}.issubset(record):
        errors.extend(verify_artifact(record, run_dir))


def verify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    config = load_config(run_dir / "config.yaml")
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("config_hash") != config.config_hash:
        errors.append("run manifest config hash mismatch")
    try:
        dataset_result = validate_dataset_path(run_dir)
    except Exception as exc:  # audit must collect all failures
        dataset_result = {"valid": False}
        errors.append(f"dataset validation failed: {exc}")
    required = {
        "activations": run_dir / "activations" / "manifest.json",
        "behavior": run_dir / "behavior" / "manifest.json",
        "operators": run_dir / "operators" / "manifest.json",
        "probes": run_dir / "probes" / "manifest.json",
        "metrics": run_dir / "metrics" / "manifest.json",
        "statistics": run_dir / "statistics" / "manifest.json",
        "report": run_dir / "report_manifest.json",
    }
    manifests: dict[str, dict[str, Any]] = {}
    for stage, path in required.items():
        if not path.is_file():
            errors.append(f"missing {stage} manifest: {path}")
            continue
        stage_manifest = read_json(path)
        manifests[stage] = stage_manifest
        if stage_manifest.get("status") != "complete":
            errors.append(f"{stage} status is not complete")
        if stage_manifest.get("config_hash") != config.config_hash:
            errors.append(f"{stage} config hash mismatch")
        if stage in {"operators", "probes", "metrics", "statistics", "report"} and not _complete(
            stage, path, config, run_dir
        ):
            errors.append(f"{stage} dependency or artifact hash verification failed")
        for value in stage_manifest.values():
            if isinstance(value, dict):
                _verify_record(value, run_dir, errors)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _verify_record(item, run_dir, errors)
    if "activations" in manifests:
        activation = manifests["activations"]
        if len(activation.get("shards", [])) != activation.get("total_shards"):
            errors.append("activation shard count differs from manifest total")
        for shard in activation.get("shards", []):
            path = run_dir / str(shard["path"])
            index_path = run_dir / str(shard["index_path"])
            if not path.is_file() or file_hash(path) != shard.get("sha256"):
                errors.append(f"invalid activation shard {path}")
            if not index_path.is_file() or file_hash(index_path) != shard.get("index_sha256"):
                errors.append(f"invalid activation shard index {index_path}")
        resolved = activation.get("model_revision")
        recorded = manifest.get("model", {}).get("resolved_revision")
        if recorded is not None and resolved != recorded:
            errors.append("model revision differs between run and activation manifests")
    if "behavior" in manifests:
        if len(manifests["behavior"].get("shards", [])) != manifests["behavior"].get(
            "total_shards"
        ):
            errors.append("behavior shard count differs from manifest total")
        for shard in manifests["behavior"].get("shards", []):
            path = run_dir / str(shard["path"])
            if not path.is_file() or file_hash(path) != shard.get("sha256"):
                errors.append(f"invalid behavior shard {path}")
    selection_path = run_dir / "operators" / "selection_frozen.json"
    if selection_path.is_file():
        selection = read_json(selection_path)
        if selection.get("test_data_used") is not False:
            errors.append("selection artifact does not certify validation-only choice")
    scientific_complete = (
        not errors
        and config.reporting.scientific_claims_allowed
        and config.activations.layers == "all"
        and config.statistics.permutation_replicates >= 1000
        and manifests.get("activations", {}).get("model_name") == config.model.name
    )
    if not config.reporting.scientific_claims_allowed:
        warnings.append("configuration explicitly forbids scientific claims")
    if config.statistics.permutation_replicates < 1000:
        warnings.append("fewer than 1,000 final permutation replicates")
    return {
        "run_id": config.run_id,
        "valid": not errors,
        "scientifically_complete": scientific_complete,
        "dataset": dataset_result,
        "errors": errors,
        "warnings": warnings,
        "verified_stages": sorted(manifests),
    }
