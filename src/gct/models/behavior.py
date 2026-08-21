"""Deterministic numeric behavior generation and parsing."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from gct.config import ExperimentConfig
from gct.models.adapters import RESPONSE_PREFILL, get_model_adapter
from gct.models.anchor import tokenize_batch

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
FINAL_PATTERN = re.compile(
    r"(?:^|\n)\s*FINAL\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*[°]?[Cc]?(?=\s*(?:\n|$))",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    status: str
    value: float | None


def parse_numeric_answer(text: str) -> ParsedAnswer:
    cleaned = text.replace(",", "").strip()
    final = FINAL_PATTERN.search(cleaned)
    if final:
        candidate = final.group(1)
    elif re.fullmatch(NUMBER_PATTERN, cleaned):
        candidate = cleaned
    else:
        return ParsedAnswer("missing_final_marker", None)
    try:
        value = float(candidate)
    except ValueError:
        return ParsedAnswer("invalid_numeric_value", None)
    if not math.isfinite(value):
        return ParsedAnswer("non_finite_value", None)
    return ParsedAnswer("parsed", value)


def generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    config: ExperimentConfig,
) -> list[str]:
    if not config.model.deterministic_decoding:
        raise ValueError("confirmatory behavior evaluation requires deterministic decoding")
    encoded = tokenize_batch(tokenizer, prompts, config, "behavior")
    input_width = encoded["input_ids"].shape[1]
    eos_token_id = get_model_adapter(config).generation_eos_token_id(tokenizer)
    with torch.inference_mode():
        generated: Any = cast(Any, model).generate(
            **encoded,
            do_sample=False,
            max_new_tokens=config.model.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
    decoded = [tokenizer.decode(row[input_width:], skip_special_tokens=True) for row in generated]
    return [RESPONSE_PREFILL + value.strip() for value in decoded]
