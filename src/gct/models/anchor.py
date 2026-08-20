"""Common chat-template anchor tokenization."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from gct.data.prompts import PromptRenderer


def chat_text(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    rendered = tokenizer.apply_chat_template(
        PromptRenderer.chat_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("chat template did not return text")
    return rendered


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase, prompts: list[str], device: str
) -> dict[str, torch.Tensor]:
    texts = [chat_text(tokenizer, prompt) for prompt in prompts]
    encoded: Any = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
    return {
        key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }


def anchor_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    return (positions * attention_mask).max(dim=1).values
