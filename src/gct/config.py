"""Validated experiment configuration and immutable run resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(StrictModel):
    name: str = "geometry-of-conditional-truth"
    protocol_version: str
    seed: int
    run_root: Path = Path("runs")


class ModelConfig(StrictModel):
    name: str = "Qwen/Qwen3-4B"
    revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: Literal["cuda", "cpu"] = "cuda"
    quantization: Literal["none"] = "none"
    trust_remote_code: bool = False
    chat_template: Literal["official"] = "official"
    deterministic_decoding: bool = True
    max_new_tokens: int = Field(default=32, ge=1, le=512)


class HardwareConfig(StrictModel):
    min_vram_gb: float = 14.0
    allow_cpu_offload: bool = False
    dynamic_batching: bool = True
    initial_batch_size: int = Field(default=8, ge=1)


class FluidCoefficients(StrictModel):
    a: float
    b: float
    k: float
    q: float


class SensorConfig(StrictModel):
    r0: float = 40.0
    r1: float = 7.0


class WorldConfig(StrictModel):
    name: Literal["toythermo"] = "toythermo"
    version: str = "toythermo-v1"
    pressure_range: tuple[float, float] = (0.2, 3.0)
    concentration_range: tuple[float, float] = (0.0, 2.0)
    fluids: dict[str, FluidCoefficients]
    calibration_sensor: SensorConfig = SensorConfig()
    familiar_aliases: dict[str, str]
    renamed_fields: dict[str, str]

    @model_validator(mode="after")
    def validate_world(self) -> WorldConfig:
        if self.pressure_range[0] <= 0 or self.pressure_range[0] >= self.pressure_range[1]:
            raise ValueError("pressure_range must be increasing and strictly positive")
        if self.concentration_range[0] >= self.concentration_range[1]:
            raise ValueError("concentration_range must be increasing")
        if len(self.fluids) != 3 or set(self.fluids) != set(self.familiar_aliases):
            raise ValueError("ToyThermo requires three fluids and one alias for each")
        required_fields = {
            "temperature",
            "pressure",
            "hidden",
            "concentration",
            "sensor",
            "fluid",
            "nuisance",
        }
        if set(self.renamed_fields) != required_fields:
            raise ValueError(f"renamed_fields must define exactly {sorted(required_fields)}")
        if len(set(self.renamed_fields.values())) != len(required_fields):
            raise ValueError("renamed field aliases must be one-to-one")
        return self


class SplitsConfig(StrictModel):
    train_base_worlds: int = Field(ge=1)
    validation_base_worlds: int = Field(ge=1)
    test_base_worlds: int = Field(ge=1)
    group_key: Literal["base_world_id"] = "base_world_id"
    holdout_renderer_family: str = "json_like"
    holdout_fluid_on_secondary_test: str | None = None


class NuisanceConfig(StrictModel):
    paraphrase_variants: int = Field(default=3, ge=1)
    formats: list[str] = Field(default_factory=lambda: ["prose", "bullets", "json_like"])
    personas: list[str] = Field(default_factory=lambda: ["neutral", "engineer", "student"])
    include_irrelevant_fact: bool = True


class TransformConfig(StrictModel):
    nuisance: NuisanceConfig = NuisanceConfig()
    pressure_deltas: dict[str, list[float]]
    concentration_deltas: dict[str, list[float]]
    keep_transformed_states_in_domain: bool = True

    @model_validator(mode="after")
    def disjoint_magnitudes(self) -> TransformConfig:
        for name, values in (
            ("pressure_deltas", self.pressure_deltas),
            ("concentration_deltas", self.concentration_deltas),
        ):
            required = {"train", "validation", "test"}
            if set(values) != required:
                raise ValueError(f"{name} must define exactly {sorted(required)}")
            sets = [set(values[split]) for split in ("train", "validation", "test")]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError(f"{name} must be disjoint across splits")
        return self


class ActivationConfig(StrictModel):
    token_position: Literal["final_common_anchor"] = "final_common_anchor"
    layers: Literal["all"] | list[int] = "all"
    storage_dtype: Literal["float16", "bfloat16"] = "float16"
    shard_size: int = Field(default=256, ge=1)
    save_full_sequence: bool = False
    common_suffix_tokens: int = Field(default=4, ge=1)


class PreprocessingConfig(StrictModel):
    pca_dims: list[int] = Field(default_factory=lambda: [32, 64, 128])
    whitening: bool = True
    fit_split: Literal["train"] = "train"


class LowRankConfig(StrictModel):
    ranks: list[int] = Field(default_factory=lambda: [4, 8, 16, 32])
    ridge_alpha: float = Field(default=1.0, gt=0)
    lr: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 500
    early_stopping_patience: int = 30


class GeneratorConfig(StrictModel):
    enabled: bool = True
    reduced_dims: list[int] = Field(default_factory=lambda: [32, 64, 128])
    regularization: float = Field(default=0.001, gt=0)


class OperatorsConfig(StrictModel):
    baselines: list[str] = Field(default_factory=lambda: ["identity", "mean_shift", "affine_ridge"])
    affine_ridge_alpha: float = Field(default=1.0, gt=0)
    low_rank: LowRankConfig = LowRankConfig()
    generator: GeneratorConfig = GeneratorConfig()


class MetricsConfig(StrictModel):
    distances: list[str] = Field(
        default_factory=lambda: ["cosine", "standardized_l2", "whitened_l2"]
    )
    primary_distance: Literal["whitened_l2"] = "whitened_l2"
    behavior_tolerance_c: float = Field(default=0.5, gt=0)
    normalize_transport_against: Literal["identity"] = "identity"


class StatisticsConfig(StrictModel):
    bootstrap_replicates: int = Field(default=2000, ge=1)
    permutation_replicates: int = Field(default=1000, ge=1)
    fdr_alpha: float = Field(default=0.05, gt=0, lt=1)
    test_alpha: float = Field(default=0.05, gt=0, lt=1)
    group_key: Literal["base_world_id"] = "base_world_id"


class MdlConfig(StrictModel):
    lambda_values: list[float] = Field(default_factory=lambda: [0, 0.01, 0.03, 0.1, 0.3, 1])


class ReportingConfig(StrictModel):
    save_raw_tables: bool = True
    save_figures: bool = True
    include_all_preregistered_hypotheses: bool = True
    include_negative_results: bool = True
    scientific_claims_allowed: bool = True


class PreregisteredHypothesis(StrictModel):
    title: str
    endpoint: str
    decision_rule: str


class ExperimentConfig(StrictModel):
    project: ProjectConfig
    model: ModelConfig
    hardware: HardwareConfig = HardwareConfig()
    world: WorldConfig
    splits: SplitsConfig
    arms: list[str]
    transforms: TransformConfig
    activations: ActivationConfig = ActivationConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    operators: OperatorsConfig = OperatorsConfig()
    metrics: MetricsConfig = MetricsConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    mdl: MdlConfig = MdlConfig()
    reporting: ReportingConfig = ReportingConfig()
    preregistration: dict[str, PreregisteredHypothesis]

    @model_validator(mode="after")
    def validate_protocol(self) -> ExperimentConfig:
        required_arms = {
            "explicit_coordinate",
            "inferable_unnamed_coordinate",
            "unobservable_coordinate",
            "irrelevant_coordinate",
            "semantic_renaming",
        }
        if set(self.arms) != required_arms:
            raise ValueError(f"arms must be exactly {sorted(required_arms)}")
        if set(self.preregistration) != {f"H{i}" for i in range(1, 9)}:
            raise ValueError("preregistration must encode H1 through H8")
        if self.activations.save_full_sequence:
            raise ValueError("full-sequence extraction is outside the confirmatory v0 protocol")
        return self

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @property
    def run_id(self) -> str:
        return f"{self.project.protocol_version}-{self.config_hash[:12]}"

    def run_dir(self, cwd: Path | None = None) -> Path:
        root = self.project.run_root
        if not root.is_absolute():
            root = (cwd or Path.cwd()) / root
        return root / self.run_id


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    return ExperimentConfig.model_validate(raw)


def write_config_snapshot(config: ExperimentConfig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    destination.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
