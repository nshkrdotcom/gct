"""Common final-anchor positioning and adapter-aware batch tokenization."""

from __future__ import annotations

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from gct.config import ExperimentConfig
from gct.models.adapters import TokenizationPurpose, tokenize_for_model


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    config: ExperimentConfig,
    purpose: TokenizationPurpose,
) -> dict[str, torch.Tensor]:
    return tokenize_for_model(tokenizer, prompts, config, purpose)


def anchor_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    return (positions * attention_mask).max(dim=1).values
