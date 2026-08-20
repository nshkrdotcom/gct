"""CUDA diagnostics and unquantized Qwen/Hugging Face loading."""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from huggingface_hub import model_info
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from gct.config import ExperimentConfig

PYTORCH_INSTALL_URL = "https://pytorch.org/get-started/locally/"


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    ok: bool
    python: str
    torch: str
    torch_cuda_runtime: str | None
    cuda_available: bool
    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    total_vram_gb: float | None
    free_vram_gb: float | None
    driver_report: str | None
    model: str
    dtype: str
    errors: tuple[str, ...]
    guidance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version_tuple(value: str | None) -> tuple[int, int]:
    if not value:
        return (0, 0)
    match = re.match(r"(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def runtime_report(config: ExperimentConfig) -> RuntimeReport:
    errors: list[str] = []
    guidance: list[str] = []
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    driver_report = driver.stdout.strip().splitlines()[0] if driver.returncode == 0 else None
    cuda = torch.cuda.is_available()
    name: str | None = None
    capability: tuple[int, int] | None = None
    bf16 = False
    total: float | None = None
    free: float | None = None
    if config.model.device == "cuda" and not cuda:
        errors.append("CUDA is required by the config but this PyTorch build cannot access it.")
        guidance.append(f"Install a CUDA-enabled PyTorch build from {PYTORCH_INSTALL_URL}")
    if cuda:
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        bf16 = bool(torch.cuda.is_bf16_supported())
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        total = total_bytes / 1024**3
        free = free_bytes / 1024**3
        if capability[0] >= 12:
            if _version_tuple(torch.__version__) < (2, 7):
                errors.append("Blackwell compute capability requires PyTorch 2.7 or newer.")
            cuda_version = _version_tuple(torch.version.cuda)
            if cuda_version < (12, 8):
                errors.append("Blackwell requires a PyTorch wheel built with CUDA 12.8 or newer.")
            if errors:
                guidance.append(
                    "Use an official CUDA 13.0+ wheel (or a supported CUDA 12.8 wheel for older "
                    f"PyTorch releases): {PYTORCH_INSTALL_URL}"
                )
        if config.model.dtype == "bfloat16" and not bf16:
            errors.append("BF16 is configured but torch.cuda.is_bf16_supported() is false.")
        if total is not None and total < config.hardware.min_vram_gb:
            errors.append(
                f"GPU has {total:.2f} GiB VRAM; config requires {config.hardware.min_vram_gb:.2f}."
            )
    if config.model.device == "cpu" and config.model.name == "Qwen/Qwen3-4B":
        guidance.append(
            "CPU is supported for diagnostics only; the primary full run requires CUDA."
        )
    if config.model.quantization != "none":
        errors.append("The primary loader does not accept quantization.")
    return RuntimeReport(
        ok=not errors,
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda_runtime=torch.version.cuda,
        cuda_available=cuda,
        gpu_name=name,
        compute_capability=capability,
        bf16_supported=bf16,
        total_vram_gb=total,
        free_vram_gb=free,
        driver_report=driver_report,
        model=config.model.name,
        dtype=config.model.dtype,
        errors=tuple(errors),
        guidance=tuple(dict.fromkeys(guidance)),
    )


def require_compatible_runtime(config: ExperimentConfig) -> RuntimeReport:
    report = runtime_report(config)
    if not report.ok:
        details = "\n".join([*report.errors, *report.guidance])
        raise RuntimeError(f"incompatible model runtime:\n{details}")
    return report


def resolve_model_revision(config: ExperimentConfig) -> str:
    info = model_info(config.model.name, revision=config.model.revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {config.model.name}")
    return str(info.sha)


def load_model_and_tokenizer(
    config: ExperimentConfig, resolved_revision: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    require_compatible_runtime(config)
    revision = resolved_revision or resolve_model_revision(config)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.model.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            config.model.name,
            revision=revision,
            dtype=dtype,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ),
    )
    cast(Any, model).to(config.model.device)
    cast(Any, model).eval()
    return model, tokenizer, revision
