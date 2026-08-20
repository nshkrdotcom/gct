"""Deterministic ToyThermo dataset generation and leakage validation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from gct.config import ExperimentConfig
from gct.data.prompts import PromptRenderer
from gct.data.schema import DATASET_COLUMNS, DatasetRow, Split
from gct.data.split import assert_grouped_splits
from gct.data.transforms import Transform, commuting_square
from gct.provenance import initialize_run, update_run_manifest
from gct.storage.hashes import canonical_hash, file_hash
from gct.storage.manifests import artifact_record, write_json_atomic
from gct.worlds.toythermo import State, ToyThermo


def _stable_id(prefix: str, payload: Any) -> str:
    return f"{prefix}-{canonical_hash(payload)[:20]}"


def _number_for_leakage_check(value: float) -> str:
    return format(value, ".8f")


def _safe_state(
    world: ToyThermo,
    rng: np.random.Generator,
    split: Split,
    index: int,
    config: ExperimentConfig,
) -> State:
    p_deltas = config.transforms.pressure_deltas[split]
    m_deltas = config.transforms.concentration_deltas[split]
    p_min, p_max = config.world.pressure_range
    m_min, m_max = config.world.concentration_range
    low_p = max(p_min, p_min - min(p_deltas))
    high_p = min(p_max, p_max - max(p_deltas))
    low_m = max(m_min, m_min - min(m_deltas))
    high_m = min(m_max, m_max - max(m_deltas))
    if low_p >= high_p or low_m >= high_m:
        raise ValueError(f"transformation magnitudes leave no valid sampling domain for {split}")
    all_fluids = sorted(config.world.fluids)
    holdout = config.splits.holdout_fluid_on_secondary_test
    if split != "test" and holdout:
        fluids = [fluid for fluid in all_fluids if fluid != holdout]
    else:
        fluids = all_fluids
    fluid = fluids[index % len(fluids)]
    pressure = float(np.exp(rng.uniform(math.log(low_p), math.log(high_p))))
    concentration = float(rng.uniform(low_m, high_m))
    # Q uses an independent RNG stream keyed before any oracle calculation.
    q = float(rng.uniform(p_min, p_max))
    state = State(fluid, pressure, concentration, q)
    world.validate(state)
    return state


def _condition_specs() -> list[tuple[str, str, bool]]:
    base = [
        "explicit_coordinate",
        "inferable_unnamed_coordinate",
        "unobservable_coordinate",
        "irrelevant_coordinate",
    ]
    return [(condition, condition, False) for condition in base] + [
        ("semantic_renaming", condition, True) for condition in base
    ]


def _renderer_variant(config: ExperimentConfig, split: Split, index: int) -> str:
    holdout = config.splits.holdout_renderer_family
    if split == "test":
        return holdout
    choices = [value for value in config.transforms.nuisance.formats if value != holdout]
    choices.extend(
        f"paraphrase_{value}" for value in range(config.transforms.nuisance.paraphrase_variants)
    )
    if not choices:
        raise ValueError("no nuisance renderer remains after holdout")
    return choices[index % len(choices)]


def _make_row(
    *,
    world: ToyThermo,
    renderer: PromptRenderer,
    state: State,
    base_world_id: str,
    split: Split,
    arm: str,
    condition: str,
    renamed: bool,
    variant: str,
    transform: Transform,
    role: str,
    source_sample_id: str | None,
    square_id: str | None = None,
    cycle_id: str | None = None,
    persona: str = "neutral",
    irrelevant_fact: bool = False,
) -> DatasetRow:
    rendered = renderer.render(
        state,
        condition,
        renamed=renamed,
        variant=variant,
        persona=persona,
        irrelevant_fact=irrelevant_fact,
    )
    identity_payload = {
        "base_world_id": base_world_id,
        "arm": arm,
        "condition": condition,
        "renamed": renamed,
        "role": role,
        "transform_id": transform.transform_id,
        "source": source_sample_id,
    }
    sample_id = _stable_id("sample", identity_payload)
    return DatasetRow(
        sample_id=sample_id,
        base_world_id=base_world_id,
        split=split,
        arm=arm,
        coordinate_condition=condition,
        world_variant="renamed" if renamed else "primary",
        world_version=world.config.version,
        fluid=state.fluid,
        pressure=state.pressure,
        concentration=state.concentration,
        irrelevant_q=state.irrelevant_q,
        observable_json=json.dumps(rendered.observable, sort_keys=True, separators=(",", ":")),
        oracle_target=world.oracle(state),
        prompt=rendered.text,
        prompt_hash=rendered.prompt_hash,
        renderer_variant=variant,
        transform_id=transform.transform_id,
        transform_family=transform.family,
        transform_name=transform.name,
        transform_parameters_json=json.dumps(
            transform.parameters, sort_keys=True, separators=(",", ":")
        ),
        oracle_identity=transform.oracle_identity,
        inverse_transform_id=transform.inverse_transform_id,
        source_sample_id=source_sample_id,
        composition_json=json.dumps(transform.composition, separators=(",", ":")),
        square_id=square_id,
        cycle_id=cycle_id,
        char_count=len(rendered.text),
        secondary_entity_holdout=False,
    )


def _rows_for_base(
    world: ToyThermo,
    renderer: PromptRenderer,
    state: State,
    base_world_id: str,
    split: Split,
    index: int,
    config: ExperimentConfig,
) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    delta_p = config.transforms.pressure_deltas[split][
        index % len(config.transforms.pressure_deltas[split])
    ]
    delta_m = config.transforms.concentration_deltas[split][
        index % len(config.transforms.concentration_deltas[split])
    ]
    nuisance_variant = _renderer_variant(config, split, index)
    persona = config.transforms.nuisance.personas[index % len(config.transforms.nuisance.personas)]
    fluids = sorted(config.world.fluids)
    holdout_fluid = config.splits.holdout_fluid_on_secondary_test
    if split != "test" and holdout_fluid is not None:
        fluids = [fluid for fluid in fluids if fluid != holdout_fluid]
    target_fluid = fluids[(fluids.index(state.fluid) + 1) % len(fluids)]
    for arm, condition, renamed in _condition_specs():
        identity = Transform("identity", "identity", {}, True)
        source = _make_row(
            world=world,
            renderer=renderer,
            state=state,
            base_world_id=base_world_id,
            split=split,
            arm=arm,
            condition=condition,
            renamed=renamed,
            variant="prose",
            transform=identity,
            role="source",
            source_sample_id=None,
        )
        rows.append(source)

        pressure = Transform("substantive", "pressure_shift", {"delta": delta_p}, False)
        pressure_state = pressure.apply(state, world)
        rows.append(
            _make_row(
                world=world,
                renderer=renderer,
                state=pressure_state,
                base_world_id=base_world_id,
                split=split,
                arm=arm,
                condition=condition,
                renamed=renamed,
                variant="prose",
                transform=pressure,
                role="pressure_target",
                source_sample_id=source.sample_id,
            )
        )

        if condition == "explicit_coordinate":
            concentration = Transform(
                "substantive", "concentration_shift", {"delta": delta_m}, False
            )
            rows.append(
                _make_row(
                    world=world,
                    renderer=renderer,
                    state=concentration.apply(state, world),
                    base_world_id=base_world_id,
                    split=split,
                    arm=arm,
                    condition=condition,
                    renamed=renamed,
                    variant="prose",
                    transform=concentration,
                    role="concentration_target",
                    source_sample_id=source.sample_id,
                )
            )
            swap = Transform("substantive", "fluid_swap", {"target_fluid": target_fluid}, False)
            rows.append(
                _make_row(
                    world=world,
                    renderer=renderer,
                    state=swap.apply(state, world),
                    base_world_id=base_world_id,
                    split=split,
                    arm=arm,
                    condition=condition,
                    renamed=renamed,
                    variant="prose",
                    transform=swap,
                    role="fluid_target",
                    source_sample_id=source.sample_id,
                )
            )
            route_pm, route_mp = commuting_square(state, world, delta_p, delta_m)
            if route_pm != route_mp:
                raise AssertionError("configured P/M state transforms do not commute")
            square_id = _stable_id("square", [base_world_id, arm, condition, renamed])
            final = Transform(
                "substantive",
                "square_final",
                {"delta_p": delta_p, "delta_m": delta_m},
                False,
                composition=("pressure_shift", "concentration_shift"),
            )
            rows.append(
                _make_row(
                    world=world,
                    renderer=renderer,
                    state=route_pm,
                    base_world_id=base_world_id,
                    split=split,
                    arm=arm,
                    condition=condition,
                    renamed=renamed,
                    variant="prose",
                    transform=final,
                    role="square_final",
                    source_sample_id=source.sample_id,
                    square_id=square_id,
                )
            )

        if condition in {"explicit_coordinate", "inferable_unnamed_coordinate"}:
            nuisance = Transform(
                "nuisance",
                "nuisance_rewrite",
                {"renderer": nuisance_variant, "persona": persona},
                True,
            )
            cycle_id = _stable_id("cycle", [base_world_id, arm, condition, renamed])
            forward = _make_row(
                world=world,
                renderer=renderer,
                state=state,
                base_world_id=base_world_id,
                split=split,
                arm=arm,
                condition=condition,
                renamed=renamed,
                variant=nuisance_variant,
                transform=nuisance,
                role="nuisance_forward",
                source_sample_id=source.sample_id,
                cycle_id=cycle_id,
                persona=persona,
                irrelevant_fact=config.transforms.nuisance.include_irrelevant_fact,
            )
            rows.append(forward)
            inverse = Transform(
                "nuisance",
                "nuisance_inverse",
                {"renderer": nuisance_variant, "persona": persona},
                True,
                inverse_transform_id=nuisance.transform_id,
                composition=(nuisance.transform_id,),
            )
            rows.append(
                _make_row(
                    world=world,
                    renderer=renderer,
                    state=state,
                    base_world_id=base_world_id,
                    split=split,
                    arm=arm,
                    condition=condition,
                    renamed=renamed,
                    variant="prose",
                    transform=inverse,
                    role="nuisance_inverse",
                    source_sample_id=forward.sample_id,
                    cycle_id=cycle_id,
                )
            )
    is_secondary = (
        split == "test"
        and config.splits.holdout_fluid_on_secondary_test is not None
        and state.fluid == config.splits.holdout_fluid_on_secondary_test
    )
    return [row.model_copy(update={"secondary_entity_holdout": is_secondary}) for row in rows]


def generate_rows(config: ExperimentConfig) -> list[DatasetRow]:
    world = ToyThermo(config.world)
    renderer = PromptRenderer(world)
    seed_sequence = np.random.SeedSequence(config.project.seed)
    split_specs: list[tuple[Split, int]] = [
        ("train", config.splits.train_base_worlds),
        ("validation", config.splits.validation_base_worlds),
        ("test", config.splits.test_base_worlds),
    ]
    children = seed_sequence.spawn(len(split_specs))
    rows: list[DatasetRow] = []
    for (split, count), child in zip(split_specs, children, strict=True):
        rng = np.random.default_rng(child)
        for index in range(count):
            base_world_id = _stable_id(
                "world", {"seed": config.project.seed, "split": split, "index": index}
            )
            state = _safe_state(world, rng, split, index, config)
            rows.extend(_rows_for_base(world, renderer, state, base_world_id, split, index, config))
    return rows


def logical_dataset_hash(rows: Iterable[DatasetRow]) -> str:
    serializable = [row.model_dump(mode="json") for row in rows]
    return canonical_hash(serializable)


def _validated_records(frame: pd.DataFrame) -> list[DatasetRow]:
    optional = {"inverse_transform_id", "source_sample_id", "square_id", "cycle_id"}
    rows: list[DatasetRow] = []
    for raw in frame.to_dict(orient="records"):
        for key in optional:
            if pd.isna(raw[key]):
                raw[key] = None
        rows.append(DatasetRow.model_validate(raw))
    return rows


def build_dataset(config: ExperimentConfig, repo_root: Path) -> Path:
    run_dir = initialize_run(config, repo_root)
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(config)
    frame = pd.DataFrame([row.model_dump(mode="python") for row in rows], columns=DATASET_COLUMNS)
    validate_dataset_frame(frame, config)
    data_path = dataset_dir / "samples.parquet"
    frame.to_parquet(data_path, index=False, engine="pyarrow", compression="zstd")
    logical_hash = logical_dataset_hash(rows)
    counts = {
        "rows": len(frame),
        "base_worlds": int(frame["base_world_id"].nunique()),
        "by_split": {str(k): int(v) for k, v in frame.groupby("split").size().items()},
        "by_arm": {str(k): int(v) for k, v in frame.groupby("arm").size().items()},
        "by_coordinate_condition": {
            str(k): int(v) for k, v in frame.groupby("coordinate_condition").size().items()
        },
        "by_world_variant": {
            str(k): int(v) for k, v in frame.groupby("world_variant").size().items()
        },
        "by_renderer": {
            str(k): int(v) for k, v in frame.groupby("renderer_variant").size().items()
        },
        "by_transform": {str(k): int(v) for k, v in frame.groupby("transform_name").size().items()},
    }
    manifest = {
        "schema_version": "gct-dataset-v1",
        "config_hash": config.config_hash,
        "world_version": config.world.version,
        "seed": config.project.seed,
        "logical_dataset_hash": logical_hash,
        "parquet_sha256": file_hash(data_path),
        "counts": counts,
        "artifact": artifact_record(data_path, run_dir, "dataset_parquet"),
    }
    write_json_atomic(dataset_dir / "manifest.json", manifest)
    update_run_manifest(run_dir, dataset_hash=logical_hash, status="dataset_complete")
    return run_dir


def validate_dataset_frame(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    if list(frame.columns) != DATASET_COLUMNS:
        raise ValueError("dataset columns do not match the versioned row contract")
    _validated_records(frame)
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id values are not unique")
    assert_grouped_splits(frame)
    if set(frame["arm"]) != set(config.arms):
        raise ValueError("dataset does not contain every configured arm")
    by_id = frame.set_index("sample_id", drop=False)
    targets = frame[frame["source_sample_id"].notna()]
    missing = set(targets["source_sample_id"].astype(str)) - set(frame["sample_id"].astype(str))
    if missing:
        raise ValueError(f"target rows reference missing sources: {sorted(missing)[:5]}")
    for row in targets.itertuples(index=False):
        source = cast(pd.Series, by_id.loc[str(row.source_sample_id)])
        if source["split"] != row.split or source["base_world_id"] != row.base_world_id:
            raise ValueError(f"paired row crosses group/split: {row.sample_id}")
        if row.oracle_identity and not math.isclose(
            float(cast(Any, source["oracle_target"])),
            float(cast(Any, row.oracle_target)),
            abs_tol=1e-10,
        ):
            raise ValueError(f"oracle-identity transform changed truth: {row.sample_id}")
    unobservable = targets[
        (targets["coordinate_condition"] == "unobservable_coordinate")
        & (targets["transform_name"] == "pressure_shift")
    ]
    if unobservable.empty:
        raise ValueError("unobservable pressure-pair negative control is absent")
    for row in unobservable.itertuples(index=False):
        source = cast(pd.Series, by_id.loc[str(row.source_sample_id)])
        if source["prompt_hash"] != row.prompt_hash or source["prompt"] != row.prompt:
            raise ValueError("unobservable prompt changed when only hidden pressure changed")
        if _number_for_leakage_check(float(cast(Any, row.pressure))) in str(row.prompt):
            raise ValueError("unobservable prompt contains the literal hidden pressure value")
        forbidden_names = (
            ("Pressure P", "Calibration reading R", "ln(P)")
            if row.world_variant == "primary"
            else ("Control X", "Proxy G", "ln(X)")
        )
        if any(name in str(row.prompt) for name in forbidden_names):
            raise ValueError("unobservable prompt contains a pressure field name or proxy name")
        observable = json.loads(str(row.observable_json))
        if any(key in observable for key in ("pressure", "sensor_reading")):
            raise ValueError("unobservable prompt metadata leaks pressure or its proxy")
    inferable = frame[frame["coordinate_condition"] == "inferable_unnamed_coordinate"]
    sensor = config.world.calibration_sensor
    for row in inferable.itertuples(index=False):
        observable = json.loads(str(row.observable_json))
        if "pressure" in observable or "sensor_reading" not in observable:
            raise ValueError("inferable arm must expose only the deterministic sensor proxy")
        recovered = math.exp((float(observable["sensor_reading"]) - sensor.r0) / sensor.r1)
        if not math.isclose(recovered, float(cast(Any, row.pressure)), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("inferable proxy does not deterministically encode pressure")
    nuisance_rows = frame[frame["transform_name"] == "nuisance_rewrite"]
    holdout_renderer = config.splits.holdout_renderer_family
    if set(nuisance_rows[nuisance_rows["split"] == "test"]["renderer_variant"]) != {
        holdout_renderer
    }:
        raise ValueError("test nuisance rows do not exclusively use the held-out renderer")
    if (
        nuisance_rows[nuisance_rows["split"].isin(["train", "validation"])]["renderer_variant"]
        == holdout_renderer
    ).any():
        raise ValueError("held-out nuisance renderer leaked into train/validation")
    holdout_fluid = config.splits.holdout_fluid_on_secondary_test
    if holdout_fluid is not None:
        if (frame[frame["split"].isin(["train", "validation"])]["fluid"] == holdout_fluid).any():
            raise ValueError("secondary holdout fluid leaked into train/validation states")
        secondary = frame[frame["secondary_entity_holdout"]]
        enough_test_groups = config.splits.test_base_worlds >= len(config.world.fluids)
        if enough_test_groups and (secondary.empty or set(secondary["split"]) != {"test"}):
            raise ValueError("secondary entity subset is missing or is not test-only")
    primary_keys = [
        "base_world_id",
        "coordinate_condition",
        "transform_name",
        "transform_parameters_json",
    ]
    primary = frame[frame["world_variant"] == "primary"].set_index(primary_keys)
    renamed = frame[frame["world_variant"] == "renamed"].set_index(primary_keys)
    if not primary.index.equals(renamed.index):
        raise ValueError("semantic-renaming rows are not isomorphic to primary rows")
    for column in ("pressure", "concentration", "oracle_target"):
        if not np.allclose(
            primary[column].to_numpy(dtype=float), renamed[column].to_numpy(dtype=float)
        ):
            raise ValueError(f"semantic renaming changed oracle-state column {column}")
    if primary["fluid"].astype(str).tolist() != renamed["fluid"].astype(str).tolist():
        raise ValueError("semantic renaming changed internal fluid identities")
    for prompt, digest in zip(frame["prompt"], frame["prompt_hash"], strict=True):
        actual = __import__("hashlib").sha256(str(prompt).encode()).hexdigest()
        if actual != digest:
            raise ValueError("prompt hash mismatch")
    magnitude_sets: dict[str, dict[str, set[float]]] = {}
    for transform_name, configured in (
        ("pressure_shift", config.transforms.pressure_deltas),
        ("concentration_shift", config.transforms.concentration_deltas),
    ):
        magnitude_sets[transform_name] = {}
        subset = frame[frame["transform_name"] == transform_name]
        for split in ("train", "validation", "test"):
            values = {
                float(json.loads(value)["delta"])
                for value in subset[subset["split"] == split]["transform_parameters_json"]
            }
            magnitude_sets[transform_name][split] = values
            if values != set(configured[split]):
                raise ValueError(f"{transform_name} magnitudes for {split} differ from config")
    return {
        "valid": True,
        "rows": len(frame),
        "base_worlds": int(frame["base_world_id"].nunique()),
        "magnitude_sets": {
            transform: {split: sorted(values) for split, values in by_split.items()}
            for transform, by_split in magnitude_sets.items()
        },
        "unobservable_pairs": len(unobservable),
    }


def validate_dataset_path(run_dir: Path) -> dict[str, Any]:
    from gct.config import load_config

    config = load_config(run_dir / "config.yaml")
    data_path = run_dir / "dataset" / "samples.parquet"
    manifest_path = run_dir / "dataset" / "manifest.json"
    if not data_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("dataset artifacts are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != config.config_hash:
        raise ValueError("dataset manifest config mismatch")
    if manifest.get("parquet_sha256") != file_hash(data_path):
        raise ValueError("dataset parquet hash mismatch")
    frame = pd.read_parquet(data_path)
    result = validate_dataset_frame(frame, config)
    rows = _validated_records(frame)
    actual_logical = logical_dataset_hash(rows)
    if actual_logical != manifest.get("logical_dataset_hash"):
        raise ValueError("dataset logical hash mismatch")
    result["logical_dataset_hash"] = actual_logical
    return result
