from __future__ import annotations

from pathlib import Path

from gct.config import load_config, write_config_snapshot
from gct.models.adapters import PHI4_MINI_MODEL_ID, PHI4_MINI_REVISION
from gct.provenance import freeze_model2_preregistration
from gct.reporting.report import scientific_report_name
from gct.storage.manifests import read_json, write_json_atomic


def test_model2_full_and_ci_configs_are_distinct_and_claim_scoped(repo_root: Path) -> None:
    full = load_config(repo_root / "configs" / "experiment_model2_phi4mini_full.yaml")
    ci = load_config(repo_root / "configs" / "experiment_model2_phi4mini_ci.yaml")
    assert full.model.name == ci.model.name == PHI4_MINI_MODEL_ID
    assert full.model.revision == ci.model.revision == PHI4_MINI_REVISION
    assert full.run_id != ci.run_id
    assert full.run_id != "gct-v0.1-db5a41461117"
    assert full.reporting.scientific_claims_allowed is True
    assert ci.reporting.scientific_claims_allowed is False
    assert scientific_report_name(full) == "REPORT_MODEL2.md"
    assert scientific_report_name(ci) is None


def test_model1_saved_config_hash_is_backward_compatible(repo_root: Path) -> None:
    config = load_config(repo_root / "runs" / "gct-v0.1-db5a41461117" / "config.yaml")
    assert config.config_hash == "db5a414611170ba43e29ab33a3e2a614056b423ef072ab8e594f038a0c231018"
    assert config.run_id == "gct-v0.1-db5a41461117"


def test_model2_preregistration_freeze_certifies_no_test_use(
    repo_root: Path, tmp_path: Path
) -> None:
    config = load_config(repo_root / "configs" / "experiment_model2_phi4mini_full.yaml")
    config = config.model_copy(
        update={"project": config.project.model_copy(update={"run_root": Path("runs")})}
    )
    run_dir = config.run_dir(tmp_path)
    write_config_snapshot(config, run_dir / "config.yaml")
    write_json_atomic(run_dir / "manifest.json", {"config_hash": config.config_hash})
    write_json_atomic(run_dir / "dataset" / "manifest.json", {"test": "dataset"})
    write_json_atomic(run_dir / "model_adapter" / "anchor_audit.json", {"test": "adapter"})
    write_json_atomic(run_dir / "model_adapter" / "operational_probe.json", {"test": "probe"})
    freeze_model2_preregistration(config, tmp_path)
    frozen = read_json(run_dir / "preregistration_frozen.json")
    assert frozen["test_data_used"] is False
    assert frozen["test_metrics_viewed"] is False
    assert set(frozen["hypotheses"]) == {f"H{index}" for index in range(1, 9)}
