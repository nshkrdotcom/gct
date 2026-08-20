from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from gct.storage.activations import (
    _canonicalize_activation_duplicates,
    _canonicalize_behavior_duplicates,
)
from gct.storage.hashes import file_hash


def test_activation_duplicates_are_canonical_across_shards(tmp_path: Path) -> None:
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    records = []
    for shard, value in enumerate((0.0, 2.0)):
        path = activation_dir / f"activations-{shard:05d}.safetensors"
        save_file(
            {
                "activations": torch.full((1, 2, 3), value),
                "embeddings": torch.full((1, 3), value),
            },
            path,
            metadata={"test": "true"},
        )
        records.append(
            {
                "shard_number": shard,
                "start": shard,
                "rows": 1,
                "path": str(path.relative_to(tmp_path)),
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
            }
        )
    frame = pd.DataFrame({"prompt_hash": ["same", "same"]})
    manifest = {"shards": records}
    result = _canonicalize_activation_duplicates(frame, manifest, tmp_path)
    first = load_file(activation_dir / "activations-00000.safetensors")
    second = load_file(activation_dir / "activations-00001.safetensors")
    assert result["mismatched_rows_before_canonicalization"] == 1
    assert torch.equal(first["activations"], second["activations"])
    assert torch.equal(first["embeddings"], second["embeddings"])


def test_behavior_duplicates_reuse_response_but_recompute_oracle_error(tmp_path: Path) -> None:
    behavior_dir = tmp_path / "behavior"
    behavior_dir.mkdir()
    records: dict[int, dict[str, object]] = {}
    for shard, answer in enumerate((1.0, 2.0)):
        path = behavior_dir / f"behavior-{shard:05d}.parquet"
        pd.DataFrame(
            {
                "sample_id": [f"s{shard}"],
                "raw_output": [f"FINAL={answer}"],
                "parse_status": ["parsed"],
                "parsed_answer": [answer],
                "absolute_error": [0.0],
                "within_tolerance": [True],
            }
        ).to_parquet(path, index=False)
        records[shard] = {
            "shard_number": shard,
            "start": shard,
            "rows": 1,
            "path": str(path.relative_to(tmp_path)),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
    frame = pd.DataFrame(
        {"sample_id": ["s0", "s1"], "prompt_hash": ["same", "same"], "oracle_target": [1.0, 10.0]}
    )
    result = _canonicalize_behavior_duplicates(frame, records, tmp_path, tolerance=0.5)
    second = pd.read_parquet(behavior_dir / "behavior-00001.parquet").iloc[0]
    assert result["mismatched_rows_before_canonicalization"] == 1
    assert second["raw_output"] == "FINAL=1.0"
    assert second["parsed_answer"] == 1.0
    assert second["absolute_error"] == 9.0
    assert not second["within_tolerance"]
