"""Exploratory surface-feature baseline: parsing, train-only fitting, and null machinery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gct.config import ExperimentConfig
from gct.data.generate import build_dataset
from gct.probes.surface_baseline import (
    ALPHA_GRID,
    SurfaceProbe,
    build_feature_matrix,
    fit_surface_baseline,
    parse_context_fields,
)

PRIMARY_INFERABLE = """Synthetic-world rules override real-world chemistry.
T(Aquila,P,M) = 95 + 18 ln(P) + 0.55 M + 0.18 M ln(P)
Calibration law: R = 40 + 7 ln(P).

Context:
Fluid: Aquila; Concentration M: 0.71131488; Calibration reading R: 41.65338007.

Task:
Compute the oracle-defined synthetic temperature."""

RENAMED_IRRELEVANT = """Synthetic-world rules override real-world chemistry.
Calibration law: G = 40 + 7 ln(X).

Context:
Entity: Water; Composition Y: 0.71131488; Proxy G: 41.65338007; Nuisance W: 0.91029339.

Task:
Compute the oracle-defined synthetic temperature."""

PRIMARY_UNOBSERVABLE = """Synthetic-world rules override real-world chemistry.

Context:
Fluid: Boreal; Concentration M: 0.31131488.

Task:
Compute the oracle-defined synthetic temperature."""


def test_primary_context_fields_are_parsed_to_canonical_names() -> None:
    assert parse_context_fields(PRIMARY_INFERABLE) == {
        "entity": "Aquila",
        "concentration_literal": "0.71131488",
        "calibration_reading_literal": "41.65338007",
    }


def test_renamed_context_fields_map_to_the_same_canonical_names() -> None:
    assert parse_context_fields(RENAMED_IRRELEVANT) == {
        "entity": "Water",
        "concentration_literal": "0.71131488",
        "calibration_reading_literal": "41.65338007",
        "nuisance_literal": "0.91029339",
    }


def test_unobservable_context_exposes_no_pressure_bearing_literal() -> None:
    fields = parse_context_fields(PRIMARY_UNOBSERVABLE)
    assert "calibration_reading_literal" not in fields
    assert "explicit_coordinate_literal" not in fields


def test_unknown_context_label_fails_closed() -> None:
    prompt = PRIMARY_INFERABLE.replace("Concentration M:", "Molarity:")
    with pytest.raises(ValueError, match="unrecognized"):
        parse_context_fields(prompt)


def test_missing_context_block_fails_closed() -> None:
    with pytest.raises(ValueError, match="context"):
        parse_context_fields("Task:\nCompute something.")


def test_entity_vocabulary_is_fitted_on_train_only() -> None:
    prompts = [PRIMARY_INFERABLE, PRIMARY_UNOBSERVABLE]
    matrix, columns = build_feature_matrix(
        prompts,
        numeric_names=("concentration_literal",),
        entity_values=("Aquila",),
    )
    assert columns == ("concentration_literal", "entity=Aquila")
    assert matrix.shape == (2, 2)
    # The held-out entity is absent from the train vocabulary, so its indicator stays zero.
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(0.0)


def test_surface_probe_recovers_a_linear_target() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(80, 3))
    labels = (2.0 * features[:, 0] - features[:, 2] + 5.0).astype(np.float32)
    probe = SurfaceProbe(alpha=1e-6).fit(features, labels)
    assert float(np.max(np.abs(probe.predict(features) - labels))) < 1e-3


def test_surface_probe_cannot_explain_a_label_from_constant_features() -> None:
    features = np.ones((30, 2))
    labels = np.linspace(0.0, 1.0, 30, dtype=np.float32)
    probe = SurfaceProbe(alpha=1.0).fit(features, labels)
    assert float(np.std(probe.predict(features))) == pytest.approx(0.0, abs=1e-6)


def test_alpha_grid_is_predeclared_and_positive() -> None:
    assert len(ALPHA_GRID) >= 3
    assert all(alpha > 0 for alpha in ALPHA_GRID)
    assert list(ALPHA_GRID) == sorted(ALPHA_GRID)


@pytest.mark.integration
def test_surface_baseline_writes_an_exploratory_artifact(
    ci_config: ExperimentConfig, tmp_path: Path
) -> None:
    run_dir = build_dataset(ci_config, tmp_path)
    output = fit_surface_baseline(run_dir)
    assert output == run_dir / "exploratory" / "surface_baseline"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_role"] == "exploratory_non_confirmatory"
    assert manifest["config_hash"]
    assert manifest["reads_activations"] is False
    results = pd.read_parquet(output / "results.parquet")
    assert set(results["coordinate_condition"]) == {
        "explicit_coordinate",
        "inferable_unnamed_coordinate",
        "irrelevant_coordinate",
        "unobservable_coordinate",
    }
    # Nothing may be written into the confirmatory namespaces.
    assert not (run_dir / "probes" / "surface_baseline").exists()
    assert not (run_dir / "statistics" / "surface_baseline").exists()


@pytest.mark.integration
def test_surface_baseline_separates_arms_that_do_and_do_not_print_the_coordinate(
    repo_root: Path,
) -> None:
    """The baseline's own positive and negative control.

    The explicit arm prints P itself, so a trivial linear read of the prompt must
    recover it almost exactly. The unobservable arm prints no pressure-bearing
    literal, so the same reader must fail. Without both, a number from the
    inferable arm would be uninterpretable.
    """
    output = fit_surface_baseline(repo_root / "runs" / "gct-v0.1-db5a41461117")
    results = pd.read_parquet(output / "results.parquet")
    primary = results[results["world_variant"] == "primary"].set_index("coordinate_condition")
    assert float(primary.loc["explicit_coordinate", "test_r2"]) > 0.99
    assert float(primary.loc["unobservable_coordinate", "test_r2"]) <= 0.0
