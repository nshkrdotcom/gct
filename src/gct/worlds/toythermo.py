"""Axiomatic ToyThermo world; it is not a real thermodynamic model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from gct.config import WorldConfig


@dataclass(frozen=True, slots=True)
class State:
    fluid: str
    pressure: float
    concentration: float
    irrelevant_q: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToyThermo:
    """Deterministic oracle and observation laws for the configured synthetic world."""

    def __init__(self, config: WorldConfig) -> None:
        self.config = config

    def validate(self, state: State) -> None:
        if state.fluid not in self.config.fluids:
            raise ValueError(f"unknown synthetic fluid: {state.fluid}")
        p_min, p_max = self.config.pressure_range
        m_min, m_max = self.config.concentration_range
        if not p_min <= state.pressure <= p_max:
            raise ValueError(f"pressure {state.pressure} outside [{p_min}, {p_max}]")
        if not m_min <= state.concentration <= m_max:
            raise ValueError(f"concentration {state.concentration} outside [{m_min}, {m_max}]")

    def oracle(self, state: State) -> float:
        self.validate(state)
        c = self.config.fluids[state.fluid]
        log_p = math.log(state.pressure)
        return c.a + c.b * log_p + c.k * state.concentration + c.q * state.concentration * log_p

    def sensor(self, pressure: float) -> float:
        if pressure <= 0:
            raise ValueError("sensor pressure must be strictly positive")
        sensor = self.config.calibration_sensor
        return sensor.r0 + sensor.r1 * math.log(pressure)

    def inverse_sensor(self, reading: float) -> float:
        sensor = self.config.calibration_sensor
        return math.exp((reading - sensor.r0) / sensor.r1)

    def sample_state(
        self, rng: np.random.Generator, allowed_fluids: list[str] | None = None
    ) -> State:
        fluids = allowed_fluids or sorted(self.config.fluids)
        fluid = str(rng.choice(fluids))
        p_min, p_max = self.config.pressure_range
        m_min, m_max = self.config.concentration_range
        pressure = float(np.exp(rng.uniform(math.log(p_min), math.log(p_max))))
        concentration = float(rng.uniform(m_min, m_max))
        q = float(rng.uniform(p_min, p_max))
        return State(fluid, pressure, concentration, q)

    def with_pressure(self, state: State, pressure: float) -> State:
        result = replace(state, pressure=pressure)
        self.validate(result)
        return result

    def with_concentration(self, state: State, concentration: float) -> State:
        result = replace(state, concentration=concentration)
        self.validate(result)
        return result

    def swap_fluid(self, state: State, fluid: str) -> State:
        result = replace(state, fluid=fluid)
        self.validate(result)
        return result

    def label(self, fluid: str, renamed: bool) -> str:
        return self.config.familiar_aliases[fluid] if renamed else fluid
