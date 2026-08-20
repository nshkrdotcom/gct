from __future__ import annotations

from pathlib import Path

from gct.config import ExperimentConfig
from gct.pipeline import _complete
from gct.storage.hashes import file_hash
from gct.storage.manifests import write_json_atomic


def test_downstream_resume_requires_dependency_and_artifact_hashes(
    ci_config: ExperimentConfig, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    activation_manifest = run_dir / "activations" / "manifest.json"
    write_json_atomic(activation_manifest, {"status": "complete"})
    artifact = run_dir / "operators" / "operator.safetensors"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"valid")
    operator_manifest = run_dir / "operators" / "manifest.json"
    write_json_atomic(
        operator_manifest,
        {
            "status": "complete",
            "config_hash": ci_config.config_hash,
            "activation_manifest_hash": file_hash(activation_manifest),
            "operators": [
                {
                    "path": "operators/operator.safetensors",
                    "sha256": file_hash(artifact),
                    "bytes": artifact.stat().st_size,
                }
            ],
        },
    )
    assert _complete("operators", operator_manifest, ci_config, run_dir)
    artifact.write_bytes(b"corrupt")
    assert not _complete("operators", operator_manifest, ci_config, run_dir)
