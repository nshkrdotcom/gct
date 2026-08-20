# Geometry of Conditional Truth

GCT is a reproducible experiment testing whether real language-model residual states carry
reusable transport laws across controlled context transformations. The oracle is an axiomatic
synthetic world; latent coherence is never treated as truth.

The default target is unquantized `Qwen/Qwen3-4B` in BF16 on an NVIDIA RTX 5060 Ti 16 GB.
Synthetic data are intentional experimental data. Reported activation results must come from the
real checkpoint; the package contains no production fake backend.

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
machine-readable preregistration in the full config. Generated evidence is summarized in
`REPORT.md`; limitations and incomplete hardware-dependent work are stated explicitly.

## Quality gates

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src/gct
```
