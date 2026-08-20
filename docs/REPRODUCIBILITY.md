# Reproducibility

The tested host is Linux on an NVIDIA GeForce RTX 5060 Ti 16 GB (compute capability 12.0), NVIDIA
driver 610.88, Python 3.12, and PyTorch 2.12.0 with CUDA 13.0. `uv.lock` and `.python-version` define
the environment. Run:

```bash
uv sync --extra dev
uv run gct doctor
uv run gct run --config configs/experiment_full.yaml --resume
```

The full run ID is a protocol name plus the canonical config hash. A run stores its config snapshot,
dependency-lock hash, repository commit, model and tokenizer revision, seeds, dataset logical and file
hashes, activation and behavior shard hashes, frozen selection, operator/probe artifacts, metrics,
statistics, figures, and report.

Resume accepts only artifacts with matching config, dataset, model revision, upstream-manifest hash,
and recursively checked content hashes. Completed valid activation shards are not recomputed. GPU
kernels are deterministic to the extent
supported by the pinned runtime, but cross-driver bitwise equality is not promised; semantic
determinism and artifact hashes are recorded. Within a run, repeated identical prompt hashes are
canonicalized to their first dataset occurrence after raw inference. The activation and behavior
manifests preserve the pre-canonicalization mismatch counts, and the final audit checks exact equality
for the embedding, all transformer layers, and raw responses.

Audit commands:

```bash
uv run gct dataset validate --run runs/<run-id>
uv run gct verify runs/<run-id>
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src/gct
GCT_RUN_REAL_MODEL_TEST=1 uv run pytest -q tests/integration/test_real_qwen.py -m real_model
```

The engineering CI config has tiny samples and reduced layers/replicates. It validates code paths only.
