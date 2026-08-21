from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gct.config import load_config
from gct.data.prompts import PromptRenderer
from gct.models.adapters import anchor_suffix_ids, model_architecture_metadata
from gct.models.behavior import generate_batch, parse_numeric_answer
from gct.models.extract import extract_batch
from gct.models.loader import load_model_and_tokenizer, require_compatible_runtime
from gct.worlds.toythermo import State, ToyThermo


@pytest.mark.integration
@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("GCT_RUN_REAL_MODEL_TEST") != "1",
    reason="set GCT_RUN_REAL_MODEL_TEST=1 to execute the real Phi/CUDA integration",
)
def test_real_phi_bf16_all_layers_anchor_behavior_and_shard(
    repo_root: Path, tmp_path: Path
) -> None:
    config = load_config(repo_root / "configs" / "experiment_model2_phi4mini_ci.yaml")
    config = config.model_copy(
        update={
            "activations": config.activations.model_copy(update={"layers": "all"}),
            "model": config.model.model_copy(update={"max_new_tokens": 8}),
        }
    )
    require_compatible_runtime(config)
    model, tokenizer, revision = load_model_and_tokenizer(config)
    metadata = model_architecture_metadata(model)
    assert revision == config.model.revision
    assert metadata["model_type"] == "phi3"
    assert metadata["num_hidden_layers"] == 32
    assert metadata["hidden_size"] == 3072
    assert next(model.parameters()).dtype == torch.bfloat16
    assert next(model.parameters()).device.type == "cuda"

    renderer = PromptRenderer(ToyThermo(config.world))
    state = State("Aquila", 1.1, 0.5, 0.8)
    prompts = [
        renderer.render(state, condition, renamed=False, variant="prose").text
        for condition in ("explicit_coordinate", "inferable_unnamed_coordinate")
    ]
    suffix = anchor_suffix_ids(tokenizer, prompts, config, config.activations.common_suffix_tokens)
    assert suffix
    batch = extract_batch(model, tokenizer, prompts, config)
    assert batch.activations.shape == (2, 32, 3072)
    assert batch.embeddings.shape == (2, 3072)
    assert batch.layer_numbers == tuple(range(32))
    repeated = extract_batch(model, tokenizer, prompts, config)
    assert torch.equal(repeated.activations, batch.activations)
    assert torch.equal(repeated.embeddings, batch.embeddings)
    first = generate_batch(model, tokenizer, prompts[:1], config)
    second = generate_batch(model, tokenizer, prompts[:1], config)
    assert first == second
    assert parse_numeric_answer(first[0]).status == "parsed"

    shard = tmp_path / "real-phi.safetensors"
    save_file({"activations": batch.activations, "embeddings": batch.embeddings}, shard)
    restored = load_file(shard)
    assert torch.equal(restored["activations"], batch.activations)
