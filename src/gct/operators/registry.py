"""Load a safe transport artifact by its recorded model type."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gct.operators.affine import AffineRidgeTransport
from gct.operators.base import load_operator_payload
from gct.operators.baselines import IdentityTransport, MeanShiftTransport
from gct.operators.generator import ContinuousGeneratorTransport
from gct.operators.low_rank import LowRankResidualTransport


def load_transport(path: Path) -> Any:
    model_type, _, _ = load_operator_payload(path)
    classes: dict[str, Any] = {
        "identity": IdentityTransport,
        "mean_shift": MeanShiftTransport,
        "affine_ridge": AffineRidgeTransport,
        "low_rank_residual": LowRankResidualTransport,
        "continuous_generator": ContinuousGeneratorTransport,
    }
    if model_type not in classes:
        raise ValueError(f"unknown transport model type: {model_type}")
    return classes[model_type].load(path)
