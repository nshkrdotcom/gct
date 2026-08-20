"""Atomic JSON manifests with explicit artifact hashes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from gct.storage.hashes import file_hash


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return value


def artifact_record(path: Path, root: Path, kind: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "kind": kind,
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
    }


def verify_artifact(record: dict[str, Any], root: Path) -> list[str]:
    path = root / str(record["path"])
    errors: list[str] = []
    if not path.is_file():
        return [f"missing artifact: {path}"]
    actual = file_hash(path)
    if actual != record.get("sha256"):
        errors.append(f"hash mismatch for {path}: expected {record.get('sha256')}, got {actual}")
    if path.stat().st_size != record.get("bytes"):
        errors.append(f"size mismatch for {path}")
    return errors
