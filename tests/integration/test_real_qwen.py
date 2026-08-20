from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gct.config import ExperimentConfig
from gct.data.prompts import PromptRenderer
from gct.models.behavior import generate_batch, parse_numeric_answer
from gct.models.extract import extract_batch
from gct.models.loader import load_model_and_tokenizer, require_compatible_runtime
from gct.worlds.toythermo import State, ToyThermo


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("GCT_RUN_REAL_MODEL_TEST") != "1",
    reason="set GCT_RUN_REAL_MODEL_TEST=1 to execute the real Qwen/CUDA integration",
)
def test_real_qwen_all_layers_behavior_and_shard(
    ci_config: ExperimentConfig, tmp_path: Path
) -> None:
    require_compatible_runtime(ci_config)
    all_layer_config = ci_config.model_copy(
        update={
            "activations": ci_config.activations.model_copy(update={"layers": "all"}),
            "model": ci_config.model.model_copy(update={"max_new_tokens": 8}),
        }
    )
    model, tokenizer, revision = load_model_and_tokenizer(all_layer_config)
    renderer = PromptRenderer(ToyThermo(all_layer_config.world))
    state = State("Aquila", 1.1, 0.5, 0.8)
    prompts = [
        renderer.render(state, condition, renamed=False, variant="prose").text
        for condition in ("explicit_coordinate", "inferable_unnamed_coordinate")
    ]
    suffix = renderer.assert_common_anchor(
        tokenizer, prompts, all_layer_config.activations.common_suffix_tokens
    )
    assert len(suffix) == all_layer_config.activations.common_suffix_tokens
    batch = extract_batch(model, tokenizer, prompts, all_layer_config)
    assert batch.activations.shape == (
        2,
        int(model.config.num_hidden_layers),
        int(model.config.hidden_size),
    )
    assert batch.embeddings.shape == (2, int(model.config.hidden_size))
    assert batch.layer_numbers == tuple(range(int(model.config.num_hidden_layers)))
    repeated_batch = extract_batch(model, tokenizer, prompts, all_layer_config)
    assert torch.equal(repeated_batch.activations, batch.activations)
    assert torch.equal(repeated_batch.embeddings, batch.embeddings)
    first = generate_batch(model, tokenizer, prompts[:1], all_layer_config)
    second = generate_batch(model, tokenizer, prompts[:1], all_layer_config)
    assert first == second
    assert parse_numeric_answer(first[0]).status == "parsed"
    shard = tmp_path / "real-qwen.safetensors"
    save_file(
        {"activations": batch.activations, "embeddings": batch.embeddings},
        shard,
        metadata={"model_revision": revision},
    )
    restored = load_file(shard)
    assert torch.equal(restored["activations"], batch.activations)
