"""Full-dataset model-adapter token and anchor audit before extraction."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from gct.config import ExperimentConfig
from gct.data.generate import validate_dataset_path
from gct.models.adapters import RESPONSE_PREFILL, get_model_adapter
from gct.models.extract import extract_batch
from gct.models.loader import load_model_and_tokenizer, load_tokenizer, resolve_model_revision
from gct.provenance import update_run_manifest
from gct.storage.hashes import file_hash
from gct.storage.manifests import read_json, write_json_atomic


def audit_model_adapter(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    dataset = validate_dataset_path(run_dir)
    output_dir = run_dir / "model_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "anchor_audit.json"
    if path.is_file():
        existing = read_json(path)
        if (
            existing.get("status") == "complete"
            and existing.get("config_hash") == config.config_hash
            and existing.get("dataset_hash") == dataset["logical_dataset_hash"]
        ):
            return run_dir
        raise ValueError("existing model-adapter audit differs from the requested run")
    frame = pd.read_parquet(run_dir / "dataset" / "samples.parquet")
    unique = frame.drop_duplicates("prompt_hash", keep="first").sort_values("prompt_hash")
    revision = resolve_model_revision(config)
    tokenizer = load_tokenizer(config, revision)
    adapter = get_model_adapter(config)
    suffix_width = config.activations.common_suffix_tokens
    configured_suffix: tuple[int, ...] | None = None
    token_ids_by_hash: dict[str, tuple[int, ...]] = {}
    digest = hashlib.sha256()
    for row in unique.itertuples(index=False):
        prompt_hash = str(row.prompt_hash)
        ids = tuple(adapter.input_ids(tokenizer, str(row.prompt), "activation"))
        suffix = ids[-suffix_width:]
        if configured_suffix is None:
            configured_suffix = suffix
        elif suffix != configured_suffix:
            raise ValueError(f"anchor suffix drift for prompt hash {prompt_hash}")
        token_ids_by_hash[prompt_hash] = ids
        digest.update(prompt_hash.encode())
        digest.update(b"\0")
        digest.update(json.dumps(ids, separators=(",", ":")).encode())
        digest.update(b"\n")
    if configured_suffix is None:
        raise ValueError("dataset contains no prompts")
    if configured_suffix[-1] != tokenizer.encode(RESPONSE_PREFILL, add_special_tokens=False)[-1]:
        raise ValueError("activation suffix does not end at the FINAL= response prefix")
    duplicate_rows = frame[frame["prompt_hash"].duplicated(keep=False)]
    duplicate_mismatches = 0
    for _, group in duplicate_rows.groupby("prompt_hash", sort=False):
        sequences = {token_ids_by_hash[str(value)] for value in group["prompt_hash"]}
        duplicate_mismatches += int(len(sequences) != 1)
    by_id = frame.set_index("sample_id", drop=False)
    unobservable = frame[
        (frame["coordinate_condition"] == "unobservable_coordinate")
        & (frame["transform_name"] == "pressure_shift")
    ]
    unobservable_mismatches = 0
    for row in unobservable.itertuples(index=False):
        source = by_id.loc[str(row.source_sample_id)]
        if token_ids_by_hash[str(row.prompt_hash)] != token_ids_by_hash[str(source.prompt_hash)]:
            unobservable_mismatches += 1
    if duplicate_mismatches or unobservable_mismatches:
        raise ValueError("Phi tokenization violated an exact duplicate/unobservable control")
    coverage_columns = [
        "arm",
        "coordinate_condition",
        "world_variant",
        "renderer_variant",
        "transform_name",
    ]
    payload: dict[str, Any] = {
        "schema_version": "gct-model-adapter-audit-v2",
        "status": "complete",
        "config_hash": config.config_hash,
        "dataset_hash": dataset["logical_dataset_hash"],
        "model_name": config.model.name,
        "model_revision": revision,
        "adapter": adapter.name,
        "tokenizer_name": tokenizer.name_or_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "official_chat_template": True,
        "response_prefill": RESPONSE_PREFILL,
        "response_prefill_token_ids": tokenizer.encode(RESPONSE_PREFILL, add_special_tokens=False),
        "anchor_suffix_token_ids": list(configured_suffix),
        "anchor_suffix_width": suffix_width,
        "rows_audited": len(frame),
        "unique_prompts_audited": len(unique),
        "token_sequence_logical_hash": digest.hexdigest(),
        "duplicate_rows_audited": len(duplicate_rows),
        "duplicate_token_mismatches": duplicate_mismatches,
        "unobservable_pairs_audited": len(unobservable),
        "unobservable_token_mismatches": unobservable_mismatches,
        "coverage": {
            column: sorted(frame[column].astype(str).unique().tolist())
            for column in coverage_columns
        },
    }
    write_json_atomic(path, payload)
    update_run_manifest(run_dir, model_adapter_audit_hash=file_hash(path))
    return run_dir


def probe_operational_batch(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = config.run_dir(repo_root)
    validate_dataset_path(run_dir)
    output_dir = run_dir / "model_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "operational_probe.json"
    if path.is_file():
        existing = read_json(path)
        if (
            existing.get("status") == "complete"
            and existing.get("config_hash") == config.config_hash
        ):
            return run_dir
        raise ValueError("existing operational probe differs from the requested config")
    frame = pd.read_parquet(run_dir / "dataset" / "samples.parquet")
    batch_size = config.hardware.initial_batch_size
    prompts = (
        frame.drop_duplicates("prompt_hash", keep="first")
        .sort_values(["char_count", "prompt_hash"], ascending=[False, True])
        .head(batch_size)["prompt"]
        .astype(str)
        .tolist()
    )
    if len(prompts) != batch_size:
        raise ValueError("dataset does not contain enough unique prompts for the batch probe")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer, revision = load_model_and_tokenizer(config)
    batch = extract_batch(model, tokenizer, prompts, config)
    payload = {
        "schema_version": "gct-operational-batch-probe-v2",
        "status": "complete",
        "config_hash": config.config_hash,
        "model_revision": revision,
        "device": config.model.device,
        "dtype": config.model.dtype,
        "operational_batch_size": batch_size,
        "dynamic_oom_recovery_enabled": config.hardware.dynamic_batching,
        "prompt_selection": "longest unique development prompts by character count",
        "prompts_evaluated": len(prompts),
        "activation_shape": list(batch.activations.shape),
        "embedding_shape": list(batch.embeddings.shape),
        "layer_numbers": list(batch.layer_numbers),
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }
    write_json_atomic(path, payload)
    update_run_manifest(run_dir, operational_probe_hash=file_hash(path))
    del model, tokenizer, batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir
