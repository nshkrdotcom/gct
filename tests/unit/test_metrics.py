from __future__ import annotations

import numpy as np
import pandas as pd

from gct.metrics.distances import MetricSpace, cosine_distance, root_mean_square_l2
from gct.metrics.evaluate import _evaluate_generator_composition
from gct.metrics.transport import normalized_improvement
from gct.operators.generator import ContinuousGeneratorTransport
from gct.preprocessing.pca import PCASpace


def test_analytic_distances() -> None:
    left = np.array([[1.0, 0.0], [1.0, 1.0]])
    right = np.array([[1.0, 0.0], [-1.0, -1.0]])
    assert np.allclose(cosine_distance(left, left), 0)
    assert np.allclose(cosine_distance(left, right), [0, 2])
    assert np.allclose(root_mean_square_l2(left[:1], np.zeros((1, 2))), np.sqrt(0.5))


def test_metric_space_is_train_fit() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(size=(20, 5))
    space = MetricSpace.fit(train, 3)
    assert space.fit_split == "train"
    distances = space.distances(train, train)
    assert set(distances) == {"cosine", "standardized_l2", "whitened_l2"}
    assert all(np.allclose(value, 0, atol=1e-6) for value in distances.values())


def test_normalized_improvement_handles_zero_identity() -> None:
    values = normalized_improvement(np.array([1.0, 0.5]), np.array([0.0, 1.0]))
    assert np.isnan(values[0])
    assert values[1] == 0.5


def test_generator_composition_metric_records_direct_and_observed_routes() -> None:
    values = np.array([[1.0, -1.0], [1.2, -0.9], [-0.4, 0.7]], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "sample_id": ["source", "target", "unused"],
            "source_sample_id": [None, "source", None],
            "base_world_id": ["world-a", "world-a", "world-b"],
            "split": ["test", "test", "train"],
            "world_variant": ["primary", "primary", "primary"],
            "coordinate_condition": [
                "explicit_coordinate",
                "explicit_coordinate",
                "explicit_coordinate",
            ],
            "transform_name": ["identity", "pressure_shift", "identity"],
            "transform_parameters_json": ["{}", '{"delta":0.4}', "{}"],
        }
    )
    generator = ContinuousGeneratorTransport(2)
    generator.pca = PCASpace.fit(values, 2)
    generator.generator = np.diag(np.array([0.2, -0.1], dtype=np.float32))
    space = MetricSpace.fit(values, 2)
    result = _evaluate_generator_composition(
        frame,
        values,
        space,
        {("primary|explicit_coordinate|pressure_shift", "continuous_generator"): generator},
    )
    assert set(result["metric"]) == {"cosine", "standardized_l2", "whitened_l2"}
    assert np.allclose(result["delta_a"], 0.2)
    assert np.allclose(result["delta_b"], 0.2)
    assert np.all(np.isfinite(result["composed_target_defect"]))
