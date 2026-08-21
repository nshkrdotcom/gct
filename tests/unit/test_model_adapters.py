from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from gct.config import ExperimentConfig
from gct.models.adapters import (
    PHI4_MINI_MODEL_ID,
    PHI4_MINI_REVISION,
    anchor_suffix_ids,
    get_model_adapter,
    model_architecture_metadata,
)
from gct.models.extract import extract_batch


class FakeChatTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    padding_side = "left"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Sequence[int]:
        assert tokenize and add_generation_prompt
        assert "enable_thinking" not in kwargs
        body = [10 + (ord(char) % 7) for char in conversation[-1]["content"]]
        return [1, *body, 200, 201]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        assert text == "FINAL="
        return [300, 301]

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|end|>"
        return 200


class FakePhiModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="phi3",
            hidden_size=3,
            num_hidden_layers=32,
            num_attention_heads=24,
            num_key_value_heads=8,
            vocab_size=200064,
        )

    def __call__(self, **encoded: Any) -> Any:
        batch, width = encoded["input_ids"].shape
        states = tuple(
            torch.full((batch, width, 3), float(index), dtype=torch.float32) for index in range(33)
        )
        return SimpleNamespace(hidden_states=states)


def _phi_config(ci_config: ExperimentConfig) -> ExperimentConfig:
    return ci_config.model_copy(
        update={
            "model": ci_config.model.model_copy(
                update={
                    "name": PHI4_MINI_MODEL_ID,
                    "revision": PHI4_MINI_REVISION,
                    "trust_remote_code": True,
                    "device": "cpu",
                    "adapter_protocol": "phi4mini-v2",
                }
            ),
            "activations": ci_config.activations.model_copy(update={"layers": "all"}),
        }
    )


def test_phi_adapter_is_revision_pinned_and_remote_code_scoped(
    ci_config: ExperimentConfig,
) -> None:
    config = _phi_config(ci_config)
    adapter = get_model_adapter(config)
    assert adapter.name == "phi4_mini_instruct"
    assert adapter.activation_prefill == "FINAL="
    assert adapter.chat_template_kwargs == {}
    assert adapter.generation_eos_token_id(FakeChatTokenizer()) == [200, 99]

    unsafe = ci_config.model_dump(mode="python")
    unsafe["model"]["trust_remote_code"] = True
    with pytest.raises(ValueError, match="trust_remote_code"):
        ExperimentConfig.model_validate(unsafe)


def test_phi_anchor_suffix_includes_final_prefill_and_is_prompt_invariant(
    ci_config: ExperimentConfig,
) -> None:
    config = _phi_config(ci_config)
    tokenizer = FakeChatTokenizer()
    suffix = anchor_suffix_ids(tokenizer, ["short", "a different prompt"], config, 4)
    assert suffix == (200, 201, 300, 301)


def test_phi_hidden_state_indexing_is_embedding_plus_32_layers(
    ci_config: ExperimentConfig,
) -> None:
    config = _phi_config(ci_config)
    model = FakePhiModel()
    batch = extract_batch(model, FakeChatTokenizer(), ["one", "two"], config)  # type: ignore[arg-type]
    assert batch.activations.shape == (2, 32, 3)
    assert batch.embeddings.shape == (2, 3)
    assert batch.layer_numbers == tuple(range(32))
    assert torch.equal(batch.embeddings.float(), torch.zeros((2, 3)))
    assert torch.equal(batch.activations[:, 0].float(), torch.ones((2, 3)))
    assert torch.equal(batch.activations[:, 31].float(), torch.full((2, 3), 32.0))
    assert batch.anchor_token_ids.tolist() == [301, 301]


def test_phi_pinned_config_metadata_is_discovered_not_hard_coded(
    ci_config: ExperimentConfig,
) -> None:
    metadata = model_architecture_metadata(FakePhiModel())  # type: ignore[arg-type]
    assert metadata == {
        "model_type": "phi3",
        "hidden_size": 3,
        "num_hidden_layers": 32,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "vocab_size": 200064,
    }
