# Reproducibility

The tested host is Linux on an NVIDIA GeForce RTX 5060 Ti 16 GB (compute capability 12.0), NVIDIA
driver 610.88, Python 3.12, and PyTorch 2.12.0 with CUDA 13.0. `uv.lock` and `.python-version` define
the environment. Run:

```bash
uv sync --extra dev
uv run gct doctor
uv run gct run --config configs/experiment_full.yaml --resume
```

Model #2 uses pinned Phi remote code only at immutable revision
`4b00ec8714b0cb224e4fb33380cbf0919f177f3e` and a Transformers 4.53-compatible lock:

```bash
uv run gct doctor --config configs/experiment_model2_phi4mini_full.yaml
uv run gct model audit --config configs/experiment_model2_phi4mini_full.yaml
uv run gct model probe-batch --config configs/experiment_model2_phi4mini_full.yaml
uv run gct preregister freeze --config configs/experiment_model2_phi4mini_full.yaml
uv run gct run --config configs/experiment_model2_phi4mini_full.yaml --resume
uv run gct compare models \
  --baseline-run gct-v0.1-db5a41461117 \
  --replication-run gct-v0.2-phi4mini-7a87777ac843
uv run gct verify gct-v0.2-phi4mini-7a87777ac843
```

The canonical replication run is `gct-v0.2-phi4mini-7a87777ac843`. Its dataset builder copies the
Model #1 Parquet file byte-for-byte, then validates logical hash
`dd44cbc000df7322f45cce1b7faef9cd0cc22290bcac5bb9d76fb95d6f2fd84f`, all 12,600 stable row IDs,
420 group IDs, and split membership. Model source/config/tokenizer/remote-code files are hashed in the
activation manifest. The response/activation suffix audit covers every full-dataset prompt and records
token IDs `[200020, 200019, 176019, 28]`.

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
GCT_RUN_REAL_MODEL_TEST=1 uv run pytest -q tests/integration/test_real_phi.py -m real_model
```

The engineering CI configs have tiny samples and reduced layers/replicates. They validate code paths
only, have distinct protocol/config identities, and cannot generate scientific root reports.
