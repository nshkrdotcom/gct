"""End-to-end run verification against hashes and scientific-completeness gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gct.config import load_config
from gct.data.generate import validate_dataset_path
from gct.pipeline import _complete
from gct.storage.hashes import canonical_hash, file_hash
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
        frozen = dict(selection)
        recorded_freeze_hash = frozen.pop("freeze_hash", None)
        if recorded_freeze_hash != canonical_hash(frozen):
            errors.append("selection artifact freeze hash mismatch")
    phi_model = config.model.name == "microsoft/Phi-4-mini-instruct"
    if phi_model and config.reporting.scientific_claims_allowed:
        activation = manifests.get("activations", {})
        architecture = activation.get("model_architecture", {})
        if architecture.get("num_hidden_layers") != 32 or architecture.get("hidden_size") != 3072:
            errors.append("Model #2 activation architecture is not 32 layers / hidden size 3072")
        if not activation.get("model_source_files"):
            errors.append("Model #2 activation manifest lacks model/tokenizer/source file hashes")
        if "FINAL=" not in str(activation.get("anchor_semantics")):
            errors.append("Model #2 activation anchor does not certify the FINAL= prefill")
        adapter_audit_path = run_dir / "model_adapter" / "anchor_audit.json"
        operational_probe_path = run_dir / "model_adapter" / "operational_probe.json"
        preregistration_path = run_dir / "preregistration_frozen.json"
        if not adapter_audit_path.is_file():
            errors.append("Model #2 full-dataset adapter audit is missing")
        else:
            adapter_audit = read_json(adapter_audit_path)
            if (
                adapter_audit.get("unique_prompts_audited") != 10080
                or adapter_audit.get("duplicate_token_mismatches") != 0
                or adapter_audit.get("unobservable_token_mismatches") != 0
                or activation.get("model_adapter_audit_hash") != file_hash(adapter_audit_path)
            ):
                errors.append("Model #2 full-dataset adapter audit is incomplete or inconsistent")
        if not operational_probe_path.is_file():
            errors.append("Model #2 operational batch probe is missing")
        else:
            operational_probe = read_json(operational_probe_path)
            if operational_probe.get(
                "operational_batch_size"
            ) != config.hardware.initial_batch_size or activation.get(
                "operational_probe_hash"
            ) != file_hash(operational_probe_path):
                errors.append("Model #2 operational batch probe is inconsistent")
        if not preregistration_path.is_file():
            errors.append("Model #2 preregistration freeze is missing")
        else:
            preregistration = read_json(preregistration_path)
            frozen = dict(preregistration)
            freeze_hash = frozen.pop("freeze_hash", None)
            if (
                freeze_hash != canonical_hash(frozen)
                or preregistration.get("test_data_used") is not False
                or preregistration.get("test_metrics_viewed") is not False
                or activation.get("preregistration_freeze_hash") != freeze_hash
            ):
                errors.append("Model #2 preregistration freeze is invalid or inconsistent")
        duplicate_audit_path = run_dir / "duplicate_audit.json"
        if not duplicate_audit_path.is_file():
            errors.append("Model #2 exact post-canonicalization duplicate audit is missing")
        else:
            duplicate_audit = read_json(duplicate_audit_path)
            mismatch_keys = (
                "mismatched_activation_rows",
                "mismatched_activation_state_comparisons",
                "mismatched_duplicate_responses",
                "unobservable_activation_mismatches",
                "unobservable_response_mismatches",
            )
            if (
                duplicate_audit.get("status") != "complete"
                or duplicate_audit.get("stored_states_per_row") != 33
                or any(duplicate_audit.get(key) != 0 for key in mismatch_keys)
            ):
                errors.append("Model #2 exact post-canonicalization duplicate audit failed")
        source = read_json(run_dir / "dataset" / "manifest.json").get("source", {})
        if (
            source.get("run_id") != "gct-v0.1-db5a41461117"
            or source.get("copy_byte_identical") is not True
        ):
            errors.append("Model #2 dataset is not certified as an exact Model #1 copy")
        hypotheses_path = run_dir / "statistics" / "hypotheses.json"
        if hypotheses_path.is_file():
            hypotheses = read_json(hypotheses_path).get("hypotheses", {})
            if set(hypotheses) != {f"H{index}" for index in range(1, 9)}:
                errors.append("Model #2 does not report every preregistered H1-H8 endpoint")
            elif hypotheses["H6"].get("status") != "control_pass":
                errors.append("Model #2 H6 negative control failed; scientific run is contaminated")
        cross_manifest_path = run_dir / "cross_model" / "manifest.json"
        if not cross_manifest_path.is_file():
            errors.append("Model #2 paired cross-model manifest is missing")
        else:
            cross_manifest = read_json(cross_manifest_path)
            if (
                cross_manifest.get("status") != "complete"
                or cross_manifest.get("baseline_run") != "gct-v0.1-db5a41461117"
                or cross_manifest.get("replication_run") != config.run_id
                or cross_manifest.get("pairing_unit") != "base_world_id"
            ):
                errors.append("Model #2 paired cross-model manifest is inconsistent")
            for key in ("summary", "report"):
                record = cross_manifest.get(key)
                if not isinstance(record, dict):
                    errors.append(f"Model #2 cross-model manifest lacks {key} record")
                else:
                    _verify_record(record, run_dir, errors)
            for key in ("tables", "figures"):
                records = cross_manifest.get(key)
                if not isinstance(records, list) or not records:
                    errors.append(f"Model #2 cross-model manifest lacks {key}")
                else:
                    for record in records:
                        if isinstance(record, dict):
                            _verify_record(record, run_dir, errors)
            cross_summary_path = run_dir / "cross_model" / "summary.json"
            if cross_summary_path.is_file():
                cross_summary = read_json(cross_summary_path)
                comparisons = cross_summary.get("endpoint_comparisons", [])
                endpoint_names = {
                    str(record.get("hypothesis"))
                    for record in comparisons
                    if isinstance(record, dict)
                }
                h6 = next(
                    (
                        record
                        for record in comparisons
                        if isinstance(record, dict) and record.get("hypothesis") == "H6"
                    ),
                    None,
                )
                if endpoint_names != {f"H{index}" for index in range(1, 9)}:
                    errors.append("Model #2 cross-model summary does not contain H1-H8 exactly")
                if (
                    cross_summary.get("negative_controls_valid") is not True
                    or h6 is None
                    or h6.get("baseline_status") != "control_pass"
                    or h6.get("replication_status") != "control_pass"
                ):
                    errors.append("Model #2 cross-model report lacks mandatory H6 control pass")
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
