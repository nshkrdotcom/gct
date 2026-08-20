"""Structured deterministic context transformations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from gct.worlds.toythermo import State, ToyThermo


@dataclass(frozen=True, slots=True)
class Transform:
    family: Literal["identity", "nuisance", "substantive", "control"]
    name: str
    parameters: dict[str, Any]
    oracle_identity: bool
    inverse_transform_id: str | None = None
    composition: tuple[str, ...] = ()

    @property
    def transform_id(self) -> str:
        payload = {
            "family": self.family,
            "name": self.name,
            "parameters": self.parameters,
            "oracle_identity": self.oracle_identity,
            "composition": self.composition,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "tx-" + hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def apply(self, state: State, world: ToyThermo) -> State:
        if self.family in {"identity", "nuisance", "control"}:
            return state
        if self.name == "pressure_shift":
            return world.with_pressure(state, state.pressure + float(self.parameters["delta"]))
        if self.name == "concentration_shift":
            return world.with_concentration(
                state, state.concentration + float(self.parameters["delta"])
            )
        if self.name == "fluid_swap":
            return world.swap_fluid(state, str(self.parameters["target_fluid"]))
        if self.name == "square_final":
            changed = world.with_pressure(state, state.pressure + float(self.parameters["delta_p"]))
            return world.with_concentration(
                changed, changed.concentration + float(self.parameters["delta_m"])
            )
        raise ValueError(f"unsupported transform: {self.name}")

    def inverse(self) -> Transform:
        if self.family in {"identity", "nuisance", "control"}:
            return Transform(
                family=self.family,
                name=f"{self.name}_inverse" if self.name != "identity" else "identity",
                parameters=self.parameters,
                oracle_identity=self.oracle_identity,
                inverse_transform_id=self.transform_id,
                composition=(self.transform_id,),
            )
        if self.name in {"pressure_shift", "concentration_shift"}:
            return Transform(
                family=self.family,
                name=self.name,
                parameters={"delta": -float(self.parameters["delta"])},
                oracle_identity=False,
                inverse_transform_id=self.transform_id,
                composition=(self.transform_id,),
            )
        raise ValueError(f"no generic inverse for {self.name}")


def commuting_square(
    state: State, world: ToyThermo, delta_p: float, delta_m: float
) -> tuple[State, State]:
    p = Transform("substantive", "pressure_shift", {"delta": delta_p}, False)
    m = Transform("substantive", "concentration_shift", {"delta": delta_m}, False)
    return m.apply(p.apply(state, world), world), p.apply(m.apply(state, world), world)
