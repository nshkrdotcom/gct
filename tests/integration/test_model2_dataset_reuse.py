from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gct.config import load_config
from gct.data.generate import build_dataset, validate_dataset_path
from gct.storage.hashes import file_hash
from gct.verify import verify_run


@pytest.mark.integration
def test_model2_reuses_exact_model1_dataset_rows_groups_and_splits(
    repo_root: Path, tmp_path: Path
) -> None:
    config = load_config(repo_root / "configs" / "experiment_model2_phi4mini_full.yaml")
    local = config.model_copy(
        update={
            "project": config.project.model_copy(update={"run_root": Path("runs")}),
            "dataset": config.dataset.model_copy(
                update={"source_run": repo_root / "runs" / "gct-v0.1-db5a41461117"}
            ),
        }
    )
    run_dir = build_dataset(local, tmp_path)
    result = validate_dataset_path(run_dir)
    baseline_dir = repo_root / "runs" / "gct-v0.1-db5a41461117"
    baseline = pd.read_parquet(baseline_dir / "dataset" / "samples.parquet")
    replication = pd.read_parquet(run_dir / "dataset" / "samples.parquet")
    assert result["logical_dataset_hash"] == (
        "dd44cbc000df7322f45cce1b7faef9cd0cc22290bcac5bb9d76fb95d6f2fd84f"
    )
    assert len(replication) == 12600
    assert replication["base_world_id"].nunique() == 420
    assert replication.groupby("split").size().to_dict() == {
        "test": 3600,
        "train": 7200,
        "validation": 1800,
    }
    stable = ["sample_id", "base_world_id", "split"]
    pd.testing.assert_frame_equal(replication[stable], baseline[stable], check_exact=True)
    assert file_hash(run_dir / "dataset" / "samples.parquet") == file_hash(
        baseline_dir / "dataset" / "samples.parquet"
    )


@pytest.mark.integration
def test_model1_run_still_verifies_after_model2_changes(repo_root: Path) -> None:
    result = verify_run(repo_root / "runs" / "gct-v0.1-db5a41461117")
    assert result["valid"] is True
    assert result["scientifically_complete"] is True
    assert result["errors"] == []
