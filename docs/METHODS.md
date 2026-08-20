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
- P and every proxy absent, with pressure-shift pairs rendering byte-identical prompts;
- the inferable condition plus causally irrelevant, independently sampled Q;
- all four repeated with familiar-label aliases and an explicit synthetic-law override.

Every base world also has held-out nuisance renderings, pressure/concentration magnitudes, fluid swaps,
nuisance cycles, and pressure/concentration commuting squares. Derived rows never cross the
`base_world_id` split. Cyrene is absent from train/validation base states in the full protocol and is
marked as a secondary test subset.

## Prompt anchor and model

The model is unquantized `Qwen/Qwen3-4B` BF16. Prompts use the checkpoint's official chat template,
a fixed system message, `enable_thinking=False`, and a fixed assistant-generation anchor. Before
extraction, representative rows from every coordinate/world/renderer combination must share the
configured final token suffix. `hidden_states[0]` is saved as the embedding output; transformer layer
`l` is explicitly read from `hidden_states[l+1]`.

Only the final anchor vector is retained. Activation tensors are sharded safetensors with Parquet row
indexes and SHA-256 hashes. Deterministic greedy answers are stored separately. An answer must end in
`FINAL=<number>` or contain only a number; copied context numerals and missing markers are parse
failures, not silently coerced.

## Frozen analysis

PCA, standardization, whitening, operators, and probes fit training rows only. Layer, PCA dimension,
rank, generator dimension, and residual-probe dimension are chosen on validation. The resulting
`selection_frozen.json` is hashed before the test metrics stage.

Operators are identity, mean shift, PCA-affine ridge, factored low-rank residual transport, and a
PCA-space continuous generator. The low-rank forward pass is
`z + (z @ B.T) @ A.T + b`; no full `d x d` product is formed. Generator prediction uses a matrix
exponential and held-out magnitudes.

Distances are raw cosine, train-standardized RMS L2, and train-PCA-whitened RMS L2. The latter is
primary. The experiment reports transport, cycle, commuting-square, matching/descent, and generator
composition proxies, raw values, and identity-normalized values where the denominator is nonzero.

Residual pressure probes report held-out R2 and MAE over preregistered PCA dimensions. Each grouped
permutation replicate refits coefficients and repeats validation dimension selection. Bootstrap
confidence intervals resample base worlds. Exploratory test layer p-values receive Benjamini-Hochberg
FDR correction. Behavior-link regressions compare defect features against character/token length,
activation norm, and oracle-delta controls.
