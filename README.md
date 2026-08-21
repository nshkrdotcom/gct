# Geometry of Conditional Truth

GCT is a reproducible experiment testing whether real language-model residual states carry
reusable transport laws across controlled context transformations. The oracle is an axiomatic
synthetic world; latent coherence is never treated as truth.

The original target is unquantized `Qwen/Qwen3-4B`; the completed second-family replication uses
pinned `microsoft/Phi-4-mini-instruct` in BF16 on an NVIDIA RTX 5060 Ti 16 GB. Model #2 reuses the
exact Model #1 scientific rows and frozen H1–H8 rules. Synthetic data are intentional experimental
data. Reported activation results must come from real checkpoints; the package contains no
production fake backend.

## Setup

Python 3.11+ and a Blackwell-capable NVIDIA driver are required for the default run. The lock uses
the official PyTorch CUDA 13.0 wheel index.

```bash
uv sync --extra dev
uv run gct doctor
```

`gct doctor` fails with actionable guidance when CUDA, BF16, or Blackwell support is incompatible;
it never silently sends a full model run to CPU.

## Reproduce

The engineering configuration exercises code paths and is not scientific validation:

```bash
uv run gct run --config configs/experiment_ci.yaml --resume
```

The preregistered experiment is:

```bash
uv run gct run --config configs/experiment_full.yaml --resume
```

The completed Model #2 replication and paired comparison are reproduced with:

```bash
uv run gct run --config configs/experiment_model2_phi4mini_full.yaml --resume
uv run gct compare models \
  --baseline-run gct-v0.1-db5a41461117 \
  --replication-run gct-v0.2-phi4mini-7a87777ac843
uv run gct verify gct-v0.2-phi4mini-7a87777ac843
```

`configs/experiment_model2_phi4mini_ci.yaml` is an engineering-only configuration and is rejected
as scientific evidence.

Individual and audit commands follow the same immutable run resolution:

```bash
uv run gct dataset build --config configs/experiment_full.yaml
uv run gct dataset validate --run runs/<run-id>
uv run gct activations extract --config configs/experiment_full.yaml --resume
uv run gct behavior evaluate --config configs/experiment_full.yaml --resume
uv run gct transport fit --config configs/experiment_full.yaml
uv run gct probes fit --config configs/experiment_full.yaml
uv run gct metrics evaluate --config configs/experiment_full.yaml
uv run gct stats run --config configs/experiment_full.yaml
uv run gct report build --config configs/experiment_full.yaml
uv run gct inspect run runs/<run-id>
uv run gct inspect sample <sample-id>
uv run gct verify runs/<run-id>
```

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [METHODS.md](docs/METHODS.md), and the
machine-readable preregistration in the full config. Generated evidence is summarized in `REPORT.md`
for Model #1, `REPORT_MODEL2.md` for Phi, and `REPORT_CROSS_MODEL.md` for the paired stable-ID/base-
world comparison. The broad reusable-transport null replicated; Phi alone supported H5 residual-
coordinate decodability while both H6 negative controls passed. H7/H8 remained unsupported, so the
result is not evidence of causal use, ontology, or universal truth geometry.

## Quality gates

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src/gct
GCT_RUN_REAL_MODEL_TEST=1 uv run pytest -q tests/integration/test_real_qwen.py -m real_model
GCT_RUN_REAL_MODEL_TEST=1 uv run pytest -q tests/integration/test_real_phi.py -m real_model
```
