from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gct.operators.affine import AffineRidgeTransport
from gct.operators.baselines import IdentityTransport, MeanShiftTransport
from gct.operators.generator import ContinuousGeneratorTransport
from gct.operators.low_rank import LowRankResidualTransport


def paired(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(40, 8)).astype(np.float32)
    left = rng.normal(size=(8, 2)).astype(np.float32)
    right = rng.normal(size=(2, 8)).astype(np.float32)
    target = source + (source @ right.T) @ left.T + 0.2
    return source, target


def test_identity_and_mean_shift() -> None:
    source, target = paired()
    assert np.array_equal(IdentityTransport().fit(source, target).predict(source), source)
    shifted = MeanShiftTransport().fit(source, source + 3).predict(source)
    assert np.allclose(shifted, source + 3)


def test_low_rank_factored_fit_and_serialization(tmp_path: Path) -> None:
    source, target = paired()
    model = LowRankResidualTransport(rank=2, alpha=1e-4).fit(source, target)
    prediction = model.predict(source)
    assert np.mean((prediction - target) ** 2) < 1e-4
    assert model.a is not None and model.b is not None
    assert model.a.shape == (8, 2) and model.b.shape == (2, 8)
    path = tmp_path / "operator.safetensors"
    model.save(path, {"fit_split": "train"})
    loaded = LowRankResidualTransport.load(path)
    assert np.allclose(loaded.predict(source), prediction)


def test_affine_ridge_round_trip(tmp_path: Path) -> None:
    source, target = paired()
    model = AffineRidgeTransport(8, alpha=1e-4).fit(source, target)
    path = tmp_path / "affine.safetensors"
    model.save(path)
    loaded = AffineRidgeTransport.load(path)
    assert np.allclose(model.predict(source), loaded.predict(source))


def test_generator_composition_is_lawful() -> None:
    source, _ = paired()
    generator = ContinuousGeneratorTransport(4)
    from gct.preprocessing.pca import PCASpace

    generator.pca = PCASpace.fit(source, 4)
    generator.generator = np.diag(np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float32))
    route = generator.reduced_route(source, (0.2, 0.3))
    direct = generator.reduced_route(source, (0.5,))
    assert np.allclose(route, direct, atol=1e-6)


def test_generator_fits_deltas_and_serializes(tmp_path: Path) -> None:
    source, _ = paired()
    deltas = np.tile(np.array([-0.01, 0.01], dtype=np.float32), len(source) // 2)
    target = source * np.exp(deltas[:, None] * 0.2)
    model = ContinuousGeneratorTransport(8, 1e-4).fit_with_deltas(source, target, deltas)
    path = tmp_path / "generator.safetensors"
    model.save(path)
    loaded = ContinuousGeneratorTransport.load(path)
    assert np.allclose(
        model.predict_delta(source[:2], deltas[:2]), loaded.predict_delta(source[:2], deltas[:2])
    )


def test_generator_rejects_zero_delta() -> None:
    source, target = paired()
    with pytest.raises(ValueError, match="nonzero"):
        ContinuousGeneratorTransport(4).fit_with_deltas(source, target, np.zeros(len(source)))
