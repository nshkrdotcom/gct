"""Deterministic numeric behavior generation and parsing."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from gct.config import ExperimentConfig
from gct.models.anchor import tokenize_batch

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
FINAL_PATTERN = re.compile(
    r"(?:^|\n)\s*FINAL\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*[°]?[Cc]?(?=\s*(?:\n|$))",
    re.IGNORECASE,
)
RESPONSE_PREFILL = "FINAL="


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
    encoded = tokenize_batch(tokenizer, prompts, config.model.device)
    # The prompt asks for this exact response form. Prefilling only its fixed
    # prefix prevents verbose arithmetic from consuming the answer budget while
    # leaving the numeric prediction entirely model-generated.
    prefix_ids = tokenizer.encode(RESPONSE_PREFILL, add_special_tokens=False)
    if not prefix_ids:
        raise RuntimeError("behavior response prefill tokenized to an empty sequence")
    prefix = torch.tensor(prefix_ids, dtype=encoded["input_ids"].dtype, device=config.model.device)
    prefix = prefix.unsqueeze(0).expand(len(prompts), -1)
    encoded["input_ids"] = torch.cat((encoded["input_ids"], prefix), dim=1)
    prefix_mask = torch.ones_like(prefix, dtype=encoded["attention_mask"].dtype)
    encoded["attention_mask"] = torch.cat((encoded["attention_mask"], prefix_mask), dim=1)
    input_width = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        generated: Any = cast(Any, model).generate(
            **encoded,
            do_sample=False,
            max_new_tokens=config.model.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    decoded = [tokenizer.decode(row[input_width:], skip_special_tokens=True) for row in generated]
    return [RESPONSE_PREFILL + cast(str, value).strip() for value in decoded]
