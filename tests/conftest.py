from __future__ import annotations

from pathlib import Path

import pytest

from gct.config import ExperimentConfig, load_config


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ci_config(repo_root: Path) -> ExperimentConfig:
    return load_config(repo_root / "configs" / "experiment_ci.yaml")
