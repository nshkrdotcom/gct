"""The identical-prompt arm must be degenerate by construction, and provably so."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from gct.verify import UNOBSERVABLE_CONTROL_GROUPS, unobservable_control_errors


def _write_probe(
    run_dir: Path,
    slug: str,
    *,
    coefficient: np.ndarray,
    pca_variance: np.ndarray,
) -> None:
    path = run_dir / "probes" / slug / "pressure_probe.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "coefficient": np.ascontiguousarray(coefficient, dtype=np.float32),
            "pca_variance": np.ascontiguousarray(pca_variance, dtype=np.float32),
            "intercept": np.array([1.4070290327072144], dtype=np.float32),
        },
        str(path),
    )


def _write_degenerate_run(run_dir: Path) -> None:
    for slug in UNOBSERVABLE_CONTROL_GROUPS:
        _write_probe(
            run_dir,
            slug,
            coefficient=np.zeros(32, dtype=np.float32),
            pca_variance=np.zeros(32, dtype=np.float32),
        )


def test_degenerate_identical_prompt_probe_is_accepted(tmp_path: Path) -> None:
    _write_degenerate_run(tmp_path)
    assert unobservable_control_errors(tmp_path) == []


def test_nonzero_coefficient_is_reported_as_leakage(tmp_path: Path) -> None:
    _write_degenerate_run(tmp_path)
    coefficient = np.zeros(32, dtype=np.float32)
    coefficient[3] = 1e-6
    _write_probe(
        tmp_path,
        UNOBSERVABLE_CONTROL_GROUPS[0],
        coefficient=coefficient,
        pca_variance=np.zeros(32, dtype=np.float32),
    )
    errors = unobservable_control_errors(tmp_path)
    assert len(errors) == 1
    assert UNOBSERVABLE_CONTROL_GROUPS[0] in errors[0]
    assert "coefficient" in errors[0]


def test_nonzero_residual_variance_is_reported_as_leakage(tmp_path: Path) -> None:
    _write_degenerate_run(tmp_path)
    variance = np.zeros(32, dtype=np.float32)
    variance[0] = 1e-9
    _write_probe(
        tmp_path,
        UNOBSERVABLE_CONTROL_GROUPS[1],
        coefficient=np.zeros(32, dtype=np.float32),
        pca_variance=variance,
    )
    errors = unobservable_control_errors(tmp_path)
    assert len(errors) == 1
    assert UNOBSERVABLE_CONTROL_GROUPS[1] in errors[0]
    assert "residual variance" in errors[0]


def test_missing_identical_prompt_probe_is_reported(tmp_path: Path) -> None:
    _write_degenerate_run(tmp_path)
    (tmp_path / "probes" / UNOBSERVABLE_CONTROL_GROUPS[0] / "pressure_probe.safetensors").unlink()
    errors = unobservable_control_errors(tmp_path)
    assert len(errors) == 1
    assert "missing" in errors[0]


@pytest.mark.integration
@pytest.mark.parametrize("run_id", ["gct-v0.1-db5a41461117", "gct-v0.2-phi4mini-7a87777ac843"])
def test_completed_runs_record_a_degenerate_identical_prompt_control(
    repo_root: Path, run_id: str
) -> None:
    """Both completed families stored an intercept-only probe over a zero residual."""
    run_dir = repo_root / "runs" / run_id
    assert unobservable_control_errors(run_dir) == []
    for slug in UNOBSERVABLE_CONTROL_GROUPS:
        tensors = load_file(str(run_dir / "probes" / slug / "pressure_probe.safetensors"))
        assert not np.any(tensors["coefficient"])
        assert not np.any(tensors["pca_variance"])
        assert not np.any(tensors["pca_mean"])


@pytest.mark.integration
def test_inferable_arm_probe_is_not_degenerate(repo_root: Path) -> None:
    """The contrast that gives the degeneracy check its content."""
    for run_id in ("gct-v0.1-db5a41461117", "gct-v0.2-phi4mini-7a87777ac843"):
        path = (
            repo_root
            / "runs"
            / run_id
            / "probes"
            / "primary-inferable-unnamed-coordinate-pressure-shift"
            / "pressure_probe.safetensors"
        )
        tensors = load_file(str(path))
        assert np.any(tensors["coefficient"])
        assert np.any(tensors["pca_variance"])
