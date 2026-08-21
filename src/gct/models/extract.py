"""All-layer final-anchor extraction from a real Hugging Face causal LM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from gct.config import ExperimentConfig
from gct.models.anchor import anchor_positions, tokenize_batch


@dataclass(frozen=True, slots=True)
class ActivationBatch:
    activations: torch.Tensor
    embeddings: torch.Tensor
    layer_numbers: tuple[int, ...]
    token_counts: torch.Tensor
    anchor_token_ids: torch.Tensor


def configured_layers(config: ExperimentConfig, num_hidden_layers: int) -> tuple[int, ...]:
    if config.activations.layers == "all":
        return tuple(range(num_hidden_layers))
    layers = tuple(config.activations.layers)
    if any(layer < 0 or layer >= num_hidden_layers for layer in layers):
        raise ValueError(f"configured layer outside [0, {num_hidden_layers - 1}]")
    if len(set(layers)) != len(layers):
        raise ValueError("configured layers contain duplicates")
    return layers


def extract_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    config: ExperimentConfig,
) -> ActivationBatch:
    encoded = tokenize_batch(tokenizer, prompts, config, "activation")
    attention_mask = encoded["attention_mask"]
    positions = anchor_positions(attention_mask)
    batch_indices = torch.arange(len(prompts), device=positions.device)
    with torch.inference_mode():
        outputs: Any = cast(Any, model)(
            **encoded, output_hidden_states=True, use_cache=False, return_dict=True
        )
    hidden_states = outputs.hidden_states
    num_hidden_layers = int(cast(Any, model).config.num_hidden_layers)
    if hidden_states is None or len(hidden_states) != num_hidden_layers + 1:
        found = None if hidden_states is None else len(hidden_states)
        raise RuntimeError(
            f"expected embedding plus {num_hidden_layers} layer states, received {found}"
        )
    layers = configured_layers(config, num_hidden_layers)
    selected = torch.stack(
        [hidden_states[layer + 1][batch_indices, positions] for layer in layers], dim=1
    )
    embeddings = hidden_states[0][batch_indices, positions]
    anchor_ids = encoded["input_ids"][batch_indices, positions]
    storage_dtype = (
        torch.float16 if config.activations.storage_dtype == "float16" else torch.bfloat16
    )
    return ActivationBatch(
        activations=selected.to(device="cpu", dtype=storage_dtype),
        embeddings=embeddings.to(device="cpu", dtype=storage_dtype),
        layer_numbers=layers,
        token_counts=attention_mask.sum(dim=1).to(device="cpu"),
        anchor_token_ids=anchor_ids.to(device="cpu"),
    )
