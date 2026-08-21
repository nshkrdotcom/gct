# Limitations

- ToyThermo is axiomatic and arithmetic; it is not real thermodynamics or a broad factual benchmark.
- The protocol now has one checkpoint from each of two approximately 4B instruction-model families,
  still at one final prompt anchor. The broad transport null is more family-robust than before, but it
  does not rule out scale-, post-training-, nonlinear-, trajectory-, circuit-, or causal-intervention
  structure.
- Vector distances depend on coordinates and train-fit preprocessing. Three metrics are reported, but
  none is an intrinsic semantic metric.
- Linear, affine, low-rank, and matrix-generator failure is a valid confirmatory null. It does not rule
  out nonlinear structure; nonlinear operators are intentionally outside the confirmatory analysis.
- Probing establishes decodability, not causal use. No activation intervention is part of v0.
- Phi's supported H5 is family-dependent residual decodability. Because H7 and H8 remain
  unsupported, it is not evidence that a base lift repairs transport, that the signal is
  semantic-label invariant, or that the model uses the decoded coordinate causally. H6 passing
  alongside it adds nothing to that reading; see the identical-prompt entry below.
- Explicit P and inferable R differ in computational accessibility as well as base coordinates. The Q
  arm controls added prompt capacity, but not every possible cognitive-load confound.
- **The inferable arm prints R, and P is decodable from R alone without a model.** An exploratory
  post-hoc baseline fits P from the numeric literals rendered into the prompt, on the same held-out
  rows and with the residual probe's estimator, null, and bootstrap. It reaches R² 0.8874
  (95% CI [0.8618, 0.9105]; MAE 0.1700; permutation p 0.000999) in the primary inferable arm,
  against the supporting residual probe's R² 0.2878 and MAE 0.4573. The quantity H5 recovers is
  therefore arithmetic on a number written in the input rather than hidden information, and H5 is
  materially weaker than it reads: it is not evidence that the model inferred and represented an
  otherwise inaccessible coordinate. That baseline is exploratory and non-confirmatory; it changes
  no endpoint status, and it does not establish that the residual carries nothing beyond the
  printed reading, which would need a conditional test that was not run. See
  `EXPLORATORY_SURFACE_BASELINE.md`.
- Familiar chemical names can activate pretrained priors despite explicit override text. That is the
  purpose of the stress test, not proof that the aliases are neutral.
- Finite base worlds, grouped bootstrap replicates, and permutation replicates limit interval and
  p-value resolution.
- The full single-GPU protocol uses 240/60/120 rather than the recommended 2,000/500/1,000 grouped
  worlds. It preserves controls but has materially lower power, especially for behavior and Cyrene.
- **The H6 identical-prompt control has no power and could not have failed.** Identical prompts
  force identical deterministic activations, so the transport residual in that arm is the zero
  matrix, so the probe collapses to intercept-only and its held-out prediction is the training-label
  mean. The endpoint is then a function of the shared pressure labels alone: it is invariant to
  model, layer, and prompt world, and the preregistered rule `test_r2 <= null_95th` reduces to
  `0 <= 0`. The two completed families confirm this bit-for-bit — Qwen3-4B (hidden 2560, layer 22)
  and Phi-4-mini (hidden 3072, layer 13) both record `test_r2` −0.03576857159757152, `null_95th`
  −0.03576843143210984, `p` 0.8471528471528471, and `test_mae` 0.5390305964897076, in the primary
  and the renamed world alike, while the inferable and irrelevant arms are non-degenerate and
  model-specific. H6 is a pipeline sanity check that the probe cannot manufacture signal from a
  zero residual. It carries no information about model-side leakage in the inferable arm, and it
  must not be cited as a control that H5 survived. `gct verify` now asserts the degeneracy itself
  (all-zero coefficient over all-zero residual variance), which is the fact a prompt-rendering
  regression would actually break.
- The canonical-first duplicate policy deliberately removes rare batch-dependent BF16/SDPA variation.
  It protects identifiability controls but treats such numerical sensitivity as instrumentation noise;
  a separate systems study could analyze that sensitivity rather than erase it.
- Observational correlation between latent defects and errors does not establish causal relevance.
- Both models have low numeric-task correctness (Qwen 4.42%, Phi 2.64% over all held-out prompts).
  This limits H4/H7 behavioral interpretation but does not remove representational H1–H3/H5/H6.
- No result equates latent coherence with oracle truth or establishes a sheaf, bifibration, ontology,
  consciousness, or universal truth manifold.
