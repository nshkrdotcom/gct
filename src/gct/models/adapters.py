"""Pinned model-family adapters for chat anchoring and architecture discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from transformers.modeling_utils import PreTrainedModel

from gct.config import ExperimentConfig
from gct.data.prompts import PromptRenderer

QWEN3_MODEL_ID = "Qwen/Qwen3-4B"
PHI4_MINI_MODEL_ID = "microsoft/Phi-4-mini-instruct"
PHI4_MINI_REVISION = "4b00ec8714b0cb224e4fb33380cbf0919f177f3e"
RESPONSE_PREFILL = "FINAL="
TokenizationPurpose = Literal["activation", "behavior"]


class ChatTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None
    padding_side: str

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Any: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ModelAdapter:
    name: str
    model_id: str
    activation_prefill: str
    chat_template_kwargs: dict[str, Any]
    protocol_version: str

    def input_ids(
        self,
        tokenizer: ChatTokenizer,
        prompt: str,
        purpose: TokenizationPurpose,
    ) -> list[int]:
        values: Any = tokenizer.apply_chat_template(
            PromptRenderer.chat_messages(prompt),
            tokenize=True,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        if isinstance(values, dict):
            values = values["input_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        result = [int(value) for value in values]
        prefill = RESPONSE_PREFILL if purpose == "behavior" else self.activation_prefill
        if prefill:
            prefix_ids = tokenizer.encode(prefill, add_special_tokens=False)
            if not prefix_ids:
                raise RuntimeError("response prefill tokenized to an empty sequence")
            result.extend(int(value) for value in prefix_ids)
        if not result:
            raise RuntimeError("official chat template tokenized to an empty sequence")
        return result

    def generation_eos_token_id(self, tokenizer: ChatTokenizer) -> int | list[int]:
        if self.name != "phi4_mini_instruct":
            if tokenizer.eos_token_id is None:
                raise RuntimeError("tokenizer has no EOS token ID")
            return int(tokenizer.eos_token_id)
        end_id = int(tokenizer.convert_tokens_to_ids("<|end|>"))
        if tokenizer.eos_token_id is None:
            return [end_id]
        return list(dict.fromkeys([end_id, int(tokenizer.eos_token_id)]))


QWEN3_ADAPTER = ModelAdapter(
    name="qwen3",
    model_id=QWEN3_MODEL_ID,
    activation_prefill="",
    chat_template_kwargs={"enable_thinking": False},
    protocol_version="qwen3-v1",
)
PHI4_MINI_ADAPTER = ModelAdapter(
    name="phi4_mini_instruct",
    model_id=PHI4_MINI_MODEL_ID,
    activation_prefill=RESPONSE_PREFILL,
    chat_template_kwargs={},
    protocol_version="phi4mini-v2",
)


def get_model_adapter(config: ExperimentConfig) -> ModelAdapter:
    if config.model.name == QWEN3_MODEL_ID:
        return QWEN3_ADAPTER
    if config.model.name == PHI4_MINI_MODEL_ID:
        if config.model.revision != PHI4_MINI_REVISION or not config.model.trust_remote_code:
            raise ValueError("Phi adapter requires its pinned revision and trust_remote_code=true")
        if config.model.adapter_protocol != PHI4_MINI_ADAPTER.protocol_version:
            raise ValueError("Phi config adapter protocol does not match the implementation")
        return PHI4_MINI_ADAPTER
    raise ValueError(f"no GCT model adapter is registered for {config.model.name}")


def tokenize_for_model(
    tokenizer: ChatTokenizer,
    prompts: list[str],
    config: ExperimentConfig,
    purpose: TokenizationPurpose,
) -> dict[str, torch.Tensor]:
    adapter = get_model_adapter(config)
    rows = [adapter.input_ids(tokenizer, prompt, purpose) for prompt in prompts]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer defines neither pad_token_id nor eos_token_id")
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        values = torch.tensor(row, dtype=torch.long)
        if tokenizer.padding_side == "right":
            input_ids[index, : len(row)] = values
            attention_mask[index, : len(row)] = 1
        else:
            input_ids[index, width - len(row) :] = values
            attention_mask[index, width - len(row) :] = 1
    return {
        "input_ids": input_ids.to(config.model.device),
        "attention_mask": attention_mask.to(config.model.device),
    }


def anchor_suffix_ids(
    tokenizer: ChatTokenizer,
    prompts: list[str],
    config: ExperimentConfig,
    suffix_tokens: int,
) -> tuple[int, ...]:
    if not prompts:
        raise ValueError("at least one prompt is required for anchor validation")
    adapter = get_model_adapter(config)
    suffixes = [
        tuple(adapter.input_ids(tokenizer, prompt, "activation")[-suffix_tokens:])
        for prompt in prompts
    ]
    if any(suffix != suffixes[0] for suffix in suffixes[1:]):
        raise ValueError("prompt variants do not share the configured model-adapter anchor suffix")
    return suffixes[0]


def model_architecture_metadata(model: PreTrainedModel) -> dict[str, int | str | None]:
    model_config = cast(Any, model).config
    return {
        "model_type": getattr(model_config, "model_type", None),
        "hidden_size": int(model_config.hidden_size),
        "num_hidden_layers": int(model_config.num_hidden_layers),
        "num_attention_heads": int(model_config.num_attention_heads),
        "num_key_value_heads": int(model_config.num_key_value_heads),
        "vocab_size": int(model_config.vocab_size),
    }


def validate_loaded_architecture(model: PreTrainedModel, config: ExperimentConfig) -> None:
    metadata = model_architecture_metadata(model)
    if config.model.name != PHI4_MINI_MODEL_ID:
        return
    expected: dict[str, int | str] = {
        "model_type": "phi3",
        "hidden_size": 3072,
        "num_hidden_layers": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "vocab_size": 200064,
    }
    differences = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if differences:
        raise RuntimeError(f"pinned Phi architecture differs from the protocol: {differences}")
