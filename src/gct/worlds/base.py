"""Interfaces for independently computed synthetic oracles."""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np

StateT = TypeVar("StateT")


class World(Protocol[StateT]):
    def sample_state(
        self, rng: np.random.Generator, allowed_fluids: list[str] | None = None
    ) -> StateT:
        """Sample a state without consulting a language model."""

    def oracle(self, state: StateT) -> float:
        """Return the independent numeric oracle value."""
