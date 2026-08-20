from __future__ import annotations

from gct.config import ExperimentConfig


def test_config_hash_is_stable(ci_config: ExperimentConfig) -> None:
    assert ci_config.config_hash == ci_config.config_hash
    assert ci_config.run_id.startswith("gct-v0.1-ci-")


def test_preregistration_is_complete(ci_config: ExperimentConfig) -> None:
    assert set(ci_config.preregistration) == {f"H{i}" for i in range(1, 9)}
    assert ci_config.preprocessing.fit_split == "train"
