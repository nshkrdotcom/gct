from __future__ import annotations

import numpy as np

from gct.preprocessing.pca import PCASpace
from gct.probes.hidden_coordinate import ResidualProbe, regression_metrics


def test_residual_probe_recovers_linear_coordinate() -> None:
    rng = np.random.default_rng(9)
    residual = rng.normal(size=(100, 6)).astype(np.float32)
    labels = 2 * residual[:, 0] - residual[:, 1]
    probe = ResidualProbe(PCASpace.fit(residual, 6), 1e-5).fit(residual, labels)
    r2, mae = regression_metrics(labels, probe.predict(residual))
    assert r2 > 0.999
    assert mae < 0.01


def test_constant_residual_cannot_explain_varying_label() -> None:
    residual = np.zeros((20, 4), dtype=np.float32)
    labels = np.linspace(0, 1, 20, dtype=np.float32)
    probe = ResidualProbe(PCASpace.fit(residual, 3), 1.0).fit(residual, labels)
    r2, _ = regression_metrics(labels, probe.predict(residual))
    assert r2 <= 0
