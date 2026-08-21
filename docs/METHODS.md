# Methods

## World and controls

ToyThermo is an axiomatic synthetic world:

```text
T(S,P,M) = a[S] + b[S] ln(P) + k[S] M + q[S] M ln(P)
```

The coefficients and domains are versioned in each run config. The interaction term prevents pressure
and concentration from reducing to independent translations while their state updates still commute.
The oracle is evaluated only by Python.

Each grouped base world is rendered into four coordinate conditions and a renamed replication:

- explicit P;
- P omitted but recoverable from `R = r0 + r1 ln(P)`;
- P's value, name, and every proxy absent behind opaque symbol U (V after renaming), with
  pressure-shift pairs rendering byte-identical prompts;
- the inferable condition plus causally irrelevant, independently sampled Q;
- all four repeated with familiar entity aliases, one-to-one renamed field symbols, and an explicit
  synthetic-law override.

Every base world also has held-out nuisance renderings, pressure/concentration magnitudes, fluid swaps,
nuisance cycles, and pressure/concentration commuting squares. Nuisance renderers distinctly cover
paraphrase, active/passive phrasing, clause order, formatting, persona framing, irrelevant facts, and
a reversible field-label alias map. Derived rows never cross the `base_world_id` split. Cyrene is
absent from train/validation base states in the full protocol and is marked as a secondary test subset.

## Prompt anchor and model adapters

Model #1 is unquantized `Qwen/Qwen3-4B` BF16. Model #2 is unquantized
`microsoft/Phi-4-mini-instruct` BF16 at immutable revision
`4b00ec8714b0cb224e4fb33380cbf0919f177f3e`. A shared adapter boundary handles official chat
templates, Qwen's `enable_thinking=False`, Phi's pinned `trust_remote_code=True`, assistant headers,
official EOS IDs, and the fixed response prefill without changing semantic system/user content.
Before extraction, every full-dataset prompt is audited for the configured invariant token suffix.
`hidden_states[0]` is saved as the embedding output; transformer layer `l` is explicitly read from
`hidden_states[l+1]`. Phi runtime discovery verified 32 layers, hidden size 3072, 24 attention heads,
8 KV heads, BF16 parameters, and all embedding-plus-layer shapes.

Only the final anchor vector is retained. Activation tensors are sharded safetensors with Parquet row
indexes and SHA-256 hashes. Deterministic greedy answers are stored separately. Generation prefills
the fixed response prefix already requested by the prompt (`FINAL=`) and leaves the numeric
continuation greedy and unconstrained. A complete `FINAL=<number>` line or an all-numeric response is
parsed; copied context numerals, arithmetic expressions, and missing markers are failures rather than
silently coerced answers. The manifest versions this response protocol.

Identical prompt hashes are canonicalized to their first dataset occurrence after shard extraction and
generation. This removes rare BF16/SDPA batch-boundary variation as a hidden-label pathway. Manifests
record how many raw mismatches were found and which shards were rewritten; per-row oracle errors are
recomputed after response canonicalization.

Model #2 reuses the exact Model #1 sample Parquet rather than a statistically equivalent regeneration.
The copy, logical hash, stable IDs, group IDs, and split IDs are all verified before model inference.
Cross-model analysis joins stable IDs and resamples whole paired base worlds; row order is never a key.
Where no additive paired endpoint artifact exists (aggregate probe R² and the H8 joint gate), the
cross-model difference is explicitly descriptive.

## Frozen analysis

PCA, standardization, whitening, operators, and probes fit training rows only. Layer, PCA dimension,
rank, generator dimension, and residual-probe dimension are chosen on validation. The resulting
`selection_frozen.json` is hashed before the test metrics stage.

Operators are identity, mean shift, PCA-affine ridge, factored low-rank residual transport, and a
PCA-space continuous generator. The low-rank forward pass is
`z + (z @ B.T) @ A.T + b`; no full `d x d` product is formed. Generator prediction uses a matrix
exponential and held-out magnitudes. Train PCA uses a fixed-seed randomized SVD when truncated;
candidate dimensions slice one train-fit basis, and nested rank candidates slice one residual SVD.
These mechanical caches change neither the candidate sets nor validation scores.

Distances are raw cosine, train-standardized RMS L2, and train-PCA-whitened RMS L2. The latter is
primary. The experiment reports transport, cycle, commuting-square, matching/descent, and generator
composition proxies, raw values, and identity-normalized values where the denominator is nonzero.

Residual pressure probes report held-out R2 and MAE over preregistered PCA dimensions. Each grouped
permutation replicate refits coefficients and repeats validation dimension selection. Bootstrap
confidence intervals resample base worlds. Exploratory test layer p-values receive Benjamini-Hochberg
FDR correction. Behavior-link regressions compare defect features against character/token length,
activation norm, and oracle-delta controls.

The descriptive MDL control is `normalized held-out whitened defect + lambda * complexity`, where
complexity is the sum of an added-coordinate indicator and the fitted operator's effective parameter
count divided by the maximum candidate effective-parameter count. The predeclared lambda values are
0, 0.01, 0.03, 0.1, 0.3, and 1. The sweep is reported in full and never replaces the held-out
prediction, irrelevant-coordinate, or unobservable-coordinate decision rules.
