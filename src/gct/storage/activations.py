"""Resumable, hash-verified activation and behavior shards."""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gct.config import ExperimentConfig
from gct.data.generate import validate_dataset_path
from gct.data.prompts import PromptRenderer
from gct.models.behavior import RESPONSE_PREFILL, generate_batch, parse_numeric_answer
from gct.models.extract import extract_batch
from gct.models.loader import load_model_and_tokenizer, resolve_model_revision, runtime_report
from gct.provenance import update_run_manifest
from gct.storage.hashes import file_hash
from gct.storage.manifests import artifact_record, read_json, write_json_atomic
from gct.worlds.toythermo import ToyThermo

T = TypeVar("T")


def _deduplicate_prompts(frame: pd.DataFrame) -> tuple[list[str], list[int]]:
    unique: list[str] = []
    positions: dict[str, int] = {}
    inverse: list[int] = []
    for prompt_hash, prompt in zip(frame["prompt_hash"], frame["prompt"], strict=True):
        key = str(prompt_hash)
        if key not in positions:
            positions[key] = len(unique)
            unique.append(str(prompt))
        elif unique[positions[key]] != str(prompt):
            raise ValueError("prompt hash collision")
        inverse.append(positions[key])
    return unique, inverse


def _dynamic_batches(
    items: list[str],
    initial_batch_size: int,
    dynamic: bool,
    function: Callable[[list[str]], T],
) -> list[T]:
    results: list[T] = []
    cursor = 0
    batch_size = min(initial_batch_size, max(1, len(items)))
    while cursor < len(items):
        current = items[cursor : cursor + batch_size]
        try:
            results.append(function(current))
            cursor += len(current)
        except torch.cuda.OutOfMemoryError:
            if not dynamic or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
    return results


def _representative_prompts(frame: pd.DataFrame) -> list[str]:
    chosen = (
        frame.sort_values("sample_id")
        .groupby(["coordinate_condition", "world_variant", "renderer_variant"], observed=True)
        .head(1)
    )
    return chosen["prompt"].astype(str).tolist()


def _valid_shard(record: dict[str, Any], run_dir: Path) -> bool:
    tensor_path = run_dir / str(record["path"])
    index_path = run_dir / str(record["index_path"])
    return (
        tensor_path.is_file()
        and index_path.is_file()
        and file_hash(tensor_path) == record.get("sha256")
        and file_hash(index_path) == record.get("index_sha256")
    )


def _canonicalize_activation_duplicates(
    frame: pd.DataFrame, manifest: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    """Make identical prompt hashes bitwise identical across shard/batch boundaries."""
    duplicate_hashes = set(
        frame.loc[frame["prompt_hash"].duplicated(keep=False), "prompt_hash"].astype(str)
    )
    canonical: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    compared_rows = 0
    mismatched_rows = 0
    rewritten_shards = 0
    max_abs_difference = 0.0
    records = sorted(manifest["shards"], key=lambda item: int(item["shard_number"]))
    for record in records:
        path = run_dir / str(record["path"])
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        tensors = load_file(path, device="cpu")
        activations = tensors["activations"]
        embeddings = tensors["embeddings"]
        start = int(record["start"])
        rows = int(record["rows"])
        shard_hashes = frame.iloc[start : start + rows]["prompt_hash"].astype(str).tolist()
        changed = False
        for offset, prompt_hash in enumerate(shard_hashes):
            if prompt_hash not in duplicate_hashes:
                continue
            current = (activations[offset], embeddings[offset])
            if prompt_hash not in canonical:
                canonical[prompt_hash] = (current[0].clone(), current[1].clone())
                continue
            compared_rows += 1
            canonical_activation, canonical_embedding = canonical[prompt_hash]
            activation_equal = torch.equal(current[0], canonical_activation)
            embedding_equal = torch.equal(current[1], canonical_embedding)
            if activation_equal and embedding_equal:
                continue
            mismatched_rows += 1
            max_abs_difference = max(
                max_abs_difference,
                float((current[0].float() - canonical_activation.float()).abs().max()),
                float((current[1].float() - canonical_embedding.float()).abs().max()),
            )
            activations[offset].copy_(canonical_activation)
            embeddings[offset].copy_(canonical_embedding)
            changed = True
        if changed:
            save_file(
                {"activations": activations.contiguous(), "embeddings": embeddings.contiguous()},
                path,
                metadata=metadata,
            )
            record["sha256"] = file_hash(path)
            record["bytes"] = path.stat().st_size
            rewritten_shards += 1
    manifest["shards"] = records
    return {
        "status": "complete",
        "policy": "canonical_first_occurrence",
        "duplicate_prompt_hashes": len(duplicate_hashes),
        "compared_duplicate_rows": compared_rows,
        "mismatched_rows_before_canonicalization": mismatched_rows,
        "rewritten_shards": rewritten_shards,
        "max_abs_difference_before_canonicalization": max_abs_difference,
    }


def extract_activation_shards(
    config: ExperimentConfig, repo_root: Path, *, resume: bool = False
) -> Path:
    run_dir = config.run_dir(repo_root)
    dataset_validation = validate_dataset_path(run_dir)
    dataset_manifest = read_json(run_dir / "dataset" / "manifest.json")
    frame = pd.read_parquet(run_dir / "dataset" / "samples.parquet")
    output_dir = run_dir / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    existing = read_json(manifest_path) if manifest_path.exists() else None
    if existing is not None:
        for key, expected in (
            ("config_hash", config.config_hash),
            ("dataset_hash", dataset_manifest["logical_dataset_hash"]),
        ):
            if existing.get(key) != expected:
                raise ValueError(f"activation manifest {key} mismatch")
        if existing.get("status") == "complete" and all(
            _valid_shard(record, run_dir) for record in existing.get("shards", [])
        ):
            if resume:
                canonicalization = existing.get("duplicate_prompt_canonicalization", {})
                if canonicalization.get("status") != "complete":
                    existing["duplicate_prompt_canonicalization"] = (
                        _canonicalize_activation_duplicates(frame, existing, run_dir)
                    )
                    write_json_atomic(manifest_path, existing)
                    update_run_manifest(
                        run_dir,
                        activation_manifest_hash=file_hash(manifest_path),
                        status="activations_complete",
                    )
                return run_dir
            raise FileExistsError("valid activation artifacts already exist; pass --resume")
        revision = str(existing["model_revision"])
        shard_records = {
            int(record["shard_number"]): record for record in existing.get("shards", [])
        }
    else:
        revision = resolve_model_revision(config)
        shard_records = {}
    model, tokenizer, loaded_revision = load_model_and_tokenizer(config, revision)
    if loaded_revision != revision:
        raise RuntimeError("loaded model revision differs from frozen activation revision")
    renderer = PromptRenderer(ToyThermo(config.world))
    anchor_suffix = renderer.assert_common_anchor(
        tokenizer,
        _representative_prompts(frame),
        config.activations.common_suffix_tokens,
    )
    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    shard_size = config.activations.shard_size
    total_shards = math.ceil(len(frame) / shard_size)
    base_manifest: dict[str, Any] = {
        "schema_version": "gct-activations-v1",
        "status": "in_progress",
        "config_hash": config.config_hash,
        "dataset_hash": dataset_manifest["logical_dataset_hash"],
        "dataset_rows": len(frame),
        "model_name": config.model.name,
        "model_revision": revision,
        "tokenizer_name": tokenizer.name_or_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_revision": revision,
        "anchor_suffix_token_ids": list(anchor_suffix),
        "anchor_semantics": "final token of official assistant-generation anchor",
        "embedding_stored_separately": True,
        "model_num_hidden_layers": num_layers,
        "model_hidden_size": hidden_size,
        "configured_layers": config.activations.layers,
        "storage_dtype": config.activations.storage_dtype,
        "extraction_settings": {
            "output_hidden_states": True,
            "use_cache": False,
            "attention_implementation": "sdpa",
            "save_full_sequence": False,
            "token_position": config.activations.token_position,
            "chat_template": config.model.chat_template,
            "enable_thinking": False,
            "duplicate_prompt_policy": config.activations.duplicate_prompt_policy,
        },
        "runtime": runtime_report(config).to_dict(),
        "total_shards": total_shards,
        "shards": list(shard_records.values()),
    }
    write_json_atomic(manifest_path, base_manifest)
    for shard_number, start in enumerate(range(0, len(frame), shard_size)):
        prior = shard_records.get(shard_number)
        if resume and prior is not None and _valid_shard(prior, run_dir):
            continue
        shard_frame = frame.iloc[start : start + shard_size].reset_index(drop=True)
        unique_prompts, inverse = _deduplicate_prompts(shard_frame)
        parts = _dynamic_batches(
            unique_prompts,
            config.hardware.initial_batch_size,
            config.hardware.dynamic_batching,
            lambda prompts: extract_batch(model, tokenizer, prompts, config),
        )
        layer_numbers = parts[0].layer_numbers
        if any(part.layer_numbers != layer_numbers for part in parts):
            raise RuntimeError("layer mapping changed between extraction batches")
        unique_activations = torch.cat([part.activations for part in parts], dim=0)
        unique_embeddings = torch.cat([part.embeddings for part in parts], dim=0)
        unique_counts = torch.cat([part.token_counts for part in parts], dim=0)
        unique_anchor_ids = torch.cat([part.anchor_token_ids for part in parts], dim=0)
        mapping = torch.tensor(inverse, dtype=torch.long)
        activations = unique_activations[mapping].contiguous()
        embeddings = unique_embeddings[mapping].contiguous()
        shard_name = f"activations-{shard_number:05d}.safetensors"
        shard_path = output_dir / shard_name
        save_file(
            {"activations": activations, "embeddings": embeddings},
            shard_path,
            metadata={
                "config_hash": config.config_hash,
                "dataset_hash": str(dataset_manifest["logical_dataset_hash"]),
                "model_revision": revision,
                "layer_numbers": json.dumps(layer_numbers),
            },
        )
        index = pd.DataFrame(
            {
                "sample_id": shard_frame["sample_id"].astype(str),
                "shard_number": shard_number,
                "shard_offset": range(len(shard_frame)),
                "token_count": unique_counts[mapping].numpy(),
                "anchor_token_id": unique_anchor_ids[mapping].numpy(),
                "prompt_hash": shard_frame["prompt_hash"].astype(str),
            }
        )
        index_path = output_dir / f"index-{shard_number:05d}.parquet"
        index.to_parquet(index_path, index=False, compression="zstd")
        record = {
            "shard_number": shard_number,
            "start": start,
            "rows": len(shard_frame),
            "unique_prompts_computed": len(unique_prompts),
            "path": str(shard_path.relative_to(run_dir)),
            "sha256": file_hash(shard_path),
            "bytes": shard_path.stat().st_size,
            "index_path": str(index_path.relative_to(run_dir)),
            "index_sha256": file_hash(index_path),
            "tensor_shape": list(activations.shape),
            "embedding_shape": list(embeddings.shape),
            "layer_numbers": list(layer_numbers),
        }
        shard_records[shard_number] = record
        base_manifest["shards"] = [shard_records[key] for key in sorted(shard_records)]
        write_json_atomic(manifest_path, base_manifest)
    all_indices = [
        pd.read_parquet(run_dir / str(shard_records[number]["index_path"]))
        for number in range(total_shards)
    ]
    full_index = pd.concat(all_indices, ignore_index=True)
    if full_index["sample_id"].tolist() != frame["sample_id"].astype(str).tolist():
        raise RuntimeError("activation index order differs from dataset order")
    full_index_path = output_dir / "index.parquet"
    full_index.to_parquet(full_index_path, index=False, compression="zstd")
    base_manifest["shards"] = [shard_records[key] for key in sorted(shard_records)]
    base_manifest["duplicate_prompt_canonicalization"] = _canonicalize_activation_duplicates(
        frame, base_manifest, run_dir
    )
    base_manifest.update(
        {
            "status": "complete",
            "index": artifact_record(full_index_path, run_dir, "activation_index"),
            "shards": [shard_records[key] for key in sorted(shard_records)],
            "dataset_validation": dataset_validation,
        }
    )
    write_json_atomic(manifest_path, base_manifest)
    update_run_manifest(
        run_dir,
        model={**config.model.model_dump(mode="json"), "resolved_revision": revision},
        activation_manifest_hash=file_hash(manifest_path),
        status="activations_complete",
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir


def load_activation_layer(run_dir: Path, layer_number: int) -> tuple[pd.DataFrame, torch.Tensor]:
    manifest = read_json(run_dir / "activations" / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("activation extraction is incomplete")
    index = pd.read_parquet(run_dir / "activations" / "index.parquet")
    pieces: list[torch.Tensor] = []
    for record in manifest["shards"]:
        layer_numbers = [int(value) for value in record["layer_numbers"]]
        if layer_number not in layer_numbers:
            raise ValueError(f"layer {layer_number} was not extracted")
        with safe_open(run_dir / str(record["path"]), framework="pt", device="cpu") as tensors:
            activation_slice = tensors.get_slice("activations")
            pieces.append(activation_slice[:, layer_numbers.index(layer_number), :])
    values = torch.cat(pieces, dim=0)
    if len(values) != len(index):
        raise RuntimeError("activation tensor/index row count mismatch")
    return index, values


def _canonicalize_behavior_duplicates(
    frame: pd.DataFrame,
    records: dict[int, dict[str, Any]],
    run_dir: Path,
    tolerance: float,
) -> dict[str, Any]:
    """Make greedy responses a deterministic function of prompt hash across shards."""
    duplicate_hashes = set(
        frame.loc[frame["prompt_hash"].duplicated(keep=False), "prompt_hash"].astype(str)
    )
    canonical: dict[str, str] = {}
    compared_rows = 0
    mismatched_rows = 0
    rewritten_shards = 0
    for shard_number in sorted(records):
        record = records[shard_number]
        path = run_dir / str(record["path"])
        results = pd.read_parquet(path)
        start = int(record["start"])
        shard_frame = frame.iloc[start : start + len(results)].reset_index(drop=True)
        changed = False
        for offset, prompt_hash in enumerate(shard_frame["prompt_hash"].astype(str)):
            if prompt_hash not in duplicate_hashes:
                continue
            raw_output = str(results.at[offset, "raw_output"])
            if prompt_hash not in canonical:
                canonical[prompt_hash] = raw_output
            else:
                compared_rows += 1
                if raw_output != canonical[prompt_hash]:
                    mismatched_rows += 1
                    raw_output = canonical[prompt_hash]
                    results.at[offset, "raw_output"] = raw_output
                    changed = True
            parsed = parse_numeric_answer(raw_output)
            oracle = float(cast(Any, shard_frame.at[offset, "oracle_target"]))
            error = abs(float(parsed.value) - oracle) if parsed.value is not None else None
            expected = {
                "parse_status": parsed.status,
                "parsed_answer": parsed.value,
                "absolute_error": error,
                "within_tolerance": error is not None and error <= tolerance,
            }
            for column, value in expected.items():
                current = results.at[offset, column]
                equal = (pd.isna(current) and value is None) or current == value
                if not equal:
                    results.at[offset, column] = value
                    changed = True
        if changed:
            results.to_parquet(path, index=False, compression="zstd")
            record["sha256"] = file_hash(path)
            record["bytes"] = path.stat().st_size
            rewritten_shards += 1
    return {
        "status": "complete",
        "policy": "canonical_first_occurrence",
        "duplicate_prompt_hashes": len(duplicate_hashes),
        "compared_duplicate_rows": compared_rows,
        "mismatched_rows_before_canonicalization": mismatched_rows,
        "rewritten_shards": rewritten_shards,
    }


def _consolidate_behavior_results(
    frame: pd.DataFrame,
    records: dict[int, dict[str, Any]],
    total_shards: int,
    run_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    all_results = [pd.read_parquet(run_dir / str(records[i]["path"])) for i in range(total_shards)]
    result_frame = pd.concat(all_results, ignore_index=True)
    if result_frame["sample_id"].tolist() != frame["sample_id"].astype(str).tolist():
        raise RuntimeError("behavior result order differs from dataset order")
    result_path = run_dir / "behavior" / "results.parquet"
    result_frame.to_parquet(result_path, index=False, compression="zstd")
    return result_path, result_frame


def evaluate_behavior_shards(
    config: ExperimentConfig, repo_root: Path, *, resume: bool = False
) -> Path:
    run_dir = config.run_dir(repo_root)
    validate_dataset_path(run_dir)
    dataset_manifest = read_json(run_dir / "dataset" / "manifest.json")
    frame = pd.read_parquet(run_dir / "dataset" / "samples.parquet")
    output_dir = run_dir / "behavior"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    existing = read_json(manifest_path) if manifest_path.exists() else None
    records: dict[int, dict[str, Any]] = {}
    if existing is not None:
        if existing.get("config_hash") != config.config_hash:
            raise ValueError("behavior manifest config mismatch")
        if existing.get("dataset_hash") != dataset_manifest["logical_dataset_hash"]:
            raise ValueError("behavior manifest dataset mismatch")
        revision = str(existing["model_revision"])
        records = {int(item["shard_number"]): item for item in existing.get("shards", [])}
        if existing.get("status") == "complete" and all(
            file_hash(run_dir / str(item["path"])) == item.get("sha256")
            for item in records.values()
        ):
            if resume:
                canonicalization = existing.get("duplicate_prompt_canonicalization", {})
                if canonicalization.get("status") != "complete":
                    total_shards = int(existing["total_shards"])
                    existing["duplicate_prompt_canonicalization"] = (
                        _canonicalize_behavior_duplicates(
                            frame,
                            records,
                            run_dir,
                            config.metrics.behavior_tolerance_c,
                        )
                    )
                    result_path, result_frame = _consolidate_behavior_results(
                        frame, records, total_shards, run_dir
                    )
                    existing["shards"] = [records[key] for key in sorted(records)]
                    existing["results"] = artifact_record(result_path, run_dir, "behavior_results")
                    existing["parse_failure_count"] = int(
                        (result_frame["parse_status"] != "parsed").sum()
                    )
                    write_json_atomic(manifest_path, existing)
                    update_run_manifest(
                        run_dir,
                        behavior_manifest_hash=file_hash(manifest_path),
                        status="behavior_complete",
                    )
                return run_dir
            raise FileExistsError("valid behavior artifacts already exist; pass --resume")
    else:
        activation_manifest_path = run_dir / "activations" / "manifest.json"
        revision = (
            str(read_json(activation_manifest_path)["model_revision"])
            if activation_manifest_path.exists()
            else resolve_model_revision(config)
        )
    model, tokenizer, loaded_revision = load_model_and_tokenizer(config, revision)
    if loaded_revision != revision:
        raise RuntimeError("behavior model revision differs from frozen revision")
    shard_size = config.activations.shard_size
    total_shards = math.ceil(len(frame) / shard_size)
    manifest: dict[str, Any] = {
        "schema_version": "gct-behavior-v1",
        "status": "in_progress",
        "config_hash": config.config_hash,
        "dataset_hash": dataset_manifest["logical_dataset_hash"],
        "model_name": config.model.name,
        "model_revision": revision,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": config.model.max_new_tokens,
            "response_prefill": RESPONSE_PREFILL,
            "protocol": "deterministic-greedy-prefill-v1",
            "duplicate_prompt_policy": config.model.duplicate_prompt_policy,
        },
        "total_shards": total_shards,
        "shards": list(records.values()),
    }
    write_json_atomic(manifest_path, manifest)
    for shard_number, start in enumerate(range(0, len(frame), shard_size)):
        prior = records.get(shard_number)
        if (
            resume
            and prior is not None
            and (run_dir / str(prior["path"])).is_file()
            and file_hash(run_dir / str(prior["path"])) == prior.get("sha256")
        ):
            continue
        shard_frame = frame.iloc[start : start + shard_size].reset_index(drop=True)
        unique_prompts, inverse = _deduplicate_prompts(shard_frame)
        batches = _dynamic_batches(
            unique_prompts,
            config.hardware.initial_batch_size,
            config.hardware.dynamic_batching,
            lambda prompts: generate_batch(model, tokenizer, prompts, config),
        )
        unique_outputs = [output for batch in batches for output in batch]
        outputs = [unique_outputs[position] for position in inverse]
        parsed = [parse_numeric_answer(output) for output in outputs]
        values = [item.value for item in parsed]
        errors = [
            abs(float(value) - float(oracle)) if value is not None else None
            for value, oracle in zip(values, shard_frame["oracle_target"], strict=True)
        ]
        results = pd.DataFrame(
            {
                "sample_id": shard_frame["sample_id"].astype(str),
                "base_world_id": shard_frame["base_world_id"].astype(str),
                "split": shard_frame["split"].astype(str),
                "raw_output": outputs,
                "parse_status": [item.status for item in parsed],
                "parsed_answer": values,
                "oracle_target": shard_frame["oracle_target"].astype(float),
                "absolute_error": errors,
                "within_tolerance": [
                    error is not None and error <= config.metrics.behavior_tolerance_c
                    for error in errors
                ],
            }
        )
        path = output_dir / f"behavior-{shard_number:05d}.parquet"
        results.to_parquet(path, index=False, compression="zstd")
        record = {
            "shard_number": shard_number,
            "start": start,
            "rows": len(results),
            "unique_prompts_computed": len(unique_prompts),
            "path": str(path.relative_to(run_dir)),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        records[shard_number] = record
        manifest["shards"] = [records[key] for key in sorted(records)]
        write_json_atomic(manifest_path, manifest)
    canonicalization = _canonicalize_behavior_duplicates(
        frame, records, run_dir, config.metrics.behavior_tolerance_c
    )
    result_path, result_frame = _consolidate_behavior_results(frame, records, total_shards, run_dir)
    manifest.update(
        {
            "status": "complete",
            "results": artifact_record(result_path, run_dir, "behavior_results"),
            "shards": [records[key] for key in sorted(records)],
            "parse_failure_count": int((result_frame["parse_status"] != "parsed").sum()),
            "duplicate_prompt_canonicalization": canonicalization,
        }
    )
    write_json_atomic(manifest_path, manifest)
    update_run_manifest(
        run_dir, behavior_manifest_hash=file_hash(manifest_path), status="behavior_complete"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir
