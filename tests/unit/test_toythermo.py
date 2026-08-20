from __future__ import annotations

import math

import pytest

from gct.config import ExperimentConfig
from gct.worlds.toythermo import State, ToyThermo


def test_oracle_closed_form(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    state = State("Aquila", 1.0, 2.0, 0.25)
    assert world.oracle(state) == pytest.approx(96.1)


def test_interaction_term_is_present(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    low_m = world.oracle(State("Aquila", 2.0, 0.0)) - world.oracle(State("Aquila", 1.0, 0.0))
    high_m = world.oracle(State("Aquila", 2.0, 2.0)) - world.oracle(State("Aquila", 1.0, 2.0))
    assert high_m - low_m == pytest.approx(0.36 * math.log(2.0))


def test_sensor_round_trip(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    assert world.inverse_sensor(world.sensor(0.73)) == pytest.approx(0.73)


def test_domain_is_enforced(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    with pytest.raises(ValueError, match="pressure"):
        world.oracle(State("Boreal", 0.0, 1.0))


def test_irrelevant_q_has_no_causal_role(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    first = State("Boreal", 1.2, 0.4, 0.2)
    second = State("Boreal", 1.2, 0.4, 2.8)
    assert world.oracle(first) == world.oracle(second)


def test_irrelevant_q_uses_matched_pressure_range(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    with pytest.raises(ValueError, match="irrelevant Q"):
        world.validate(State("Aquila", 1.0, 0.5, 4.0))
