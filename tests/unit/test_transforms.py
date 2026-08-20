from __future__ import annotations

import pytest

from gct.config import ExperimentConfig
from gct.data.transforms import Transform, commuting_square
from gct.worlds.toythermo import State, ToyThermo


def test_nuisance_preserves_state_and_oracle(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    state = State("Aquila", 1.0, 0.5)
    transform = Transform("nuisance", "nuisance_rewrite", {"renderer": "bullets"}, True)
    assert transform.apply(state, world) == state
    assert world.oracle(transform.apply(state, world)) == world.oracle(state)


def test_continuous_inverse_restores_state(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    state = State("Boreal", 1.2, 0.8)
    transform = Transform("substantive", "pressure_shift", {"delta": 0.2}, False)
    restored = transform.inverse().apply(transform.apply(state, world), world)
    assert restored.pressure == pytest.approx(state.pressure)
    assert restored.concentration == state.concentration


def test_pressure_and_concentration_commute(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    state = State("Aquila", 1.3, 0.7)
    route_pm, route_mp = commuting_square(state, world, 0.2, -0.2)
    assert route_pm == route_mp
    assert world.oracle(route_pm) == world.oracle(route_mp)


def test_transform_id_is_canonical() -> None:
    left = Transform("control", "x", {"a": 1, "b": 2}, True)
    right = Transform("control", "x", {"b": 2, "a": 1}, True)
    assert left.transform_id == right.transform_id
