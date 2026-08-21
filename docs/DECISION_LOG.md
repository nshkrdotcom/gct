# Decision log

- Use raw Hugging Face hidden states, not NNsight, in the critical path. Patching is exploratory.
- Store transformer layers separately from the embedding output to eliminate indexing ambiguity.
- Use byte-identical unobservable pressure-shift prompts. This makes above-null pressure decoding an
  immediate contamination signal.
- Treat semantic renaming as one first-class arm with all four coordinate subconditions so H2/H5/H7
  can actually be repeated, rather than renaming only one prompt type.
- Use factored reduced-rank regression rather than an optimizer-dependent neural fit. Forward
  prediction never constructs a full square matrix.
- Fit continuous generators in train PCA space using finite-difference derivatives, then evaluate
  matrix exponentials on unseen magnitudes. Nonlinear operators remain outside v0.
- Deduplicate identical prompt hashes inside each model shard, then expand results back to every
  dataset row. This preserves per-row artifacts while avoiding redundant deterministic inference.
- Reduce the handoff's recommended 2,000/500/1,000 grouped counts to 240/60/120 before the frozen run.
  The unchanged 30-row control design would otherwise require about 105,000 long-form generations on
  one GPU. The reduced run retains all controls, 120 held-out groups, roughly 40 secondary entity
  groups, 2,000 bootstraps, and 1,000 refit permutations; reduced power is reported as a limitation.
- Require a terminal `FINAL=<number>` marker (or a response containing only a number) for behavioral
  parsing. Copied context numbers and truncated workings are parsing failures, not model answers.
- During the pre-analysis behavior stage, the first 512 outputs all exhausted the 128-token budget
  while narrating arithmetic and emitted no numeric answer. This was classified as a measurement-
  protocol failure, not a model error or scientific result. Before fitting operators or viewing any
  test metric, the invalid behavior shards were discarded and answer collection was repaired by
  prefilling the already-requested fixed assistant prefix `FINAL=`. The numeric continuation remains
  greedy and unconstrained, and the manifest records this as `deterministic-greedy-prefill-v1`.
- A subsequent pre-analysis acceptance audit found that voice/clause-order and reversible lexical-
  alias nuisance families were not distinct renderers. The partial run was abandoned before any
  operator fit or test statistic, these renderers were added to the frozen config, and the scientific
  run restarted under the resulting new config hash.
- The same audit found that generator composition was unit-tested but not serialized as empirical
  output. Before resuming extraction, a held-out table was added for `T_(a+b)` versus `T_b T_a`,
  together with direct-route and composed-route errors to observed targets.
- The semantic-replication audit then found that entity labels were renamed but field symbols were
  not. Before any operator fitting or test statistic, the run was superseded with a versioned,
  one-to-one `T/P/M/R/Q/U -> Z/X/Y/G/W/V` field map in addition to the familiar entity aliases.
- The scientific-QC checklist additionally required the unobservable arm to omit the pressure field
  name, not only its value and proxy. Its sealed field is therefore opaque `U` in the primary world
  and isomorphically renamed `V`; neither `P` nor `X` appears in its state description or law.
- Final leakage QC found three byte-identical unobservable pairs with different stored states when the
  rows crossed BF16/SDPA shard batches, even though the probe control passed. The affected run was
  superseded. A config-hashed `canonical_first_occurrence` policy now makes activations and greedy
  responses exact functions of prompt hash across shards before any operator fit, records raw mismatch
  counts, and recomputes row-specific behavioral errors after response canonicalization.
- Pin Python 3.12 because the available Python 3.14 free-threaded interpreter lacked binary wheels for
  critical tensor dependencies. The package source remains Python 3.11+ compatible.
- Use PyTorch 2.12.0+cu130 following current Blackwell guidance and record the full lock hash.
- Add a shared model-adapter boundary for the frozen Model #2 replication. Phi uses its official chat
  template and pinned remote code at revision `4b00ec8714b0cb224e4fb33380cbf0919f177f3e`;
  Qwen behavior remains unchanged. Phi extraction appends literal `FINAL=` after the official assistant
  header so behavior and activation use one prompt-invariant semantic answer anchor.
- The pinned Phi remote code imports `LossKwargs`, which is exported by Transformers 4.53.3 but not by
  the initially resolved 5.15.1 or tested 4.57.6 installs. Before any Model #2 output existed, constrain
  Transformers to `>=4.51,<4.54` and lock 4.53.3. This is a model-adapter compatibility change, not a
  scientific protocol change.
- A pre-test generation audit found that the inherited single tokenizer-EOS override discarded Phi's
  official two-token EOS list and allowed continuations after `<|end|>`. Before the first full behavior
  shard and before any full test metric, version the corrected adapter as `phi4mini-v2`, preserve the
  official EOS list, supersede the incomplete run identities, and restart under config hash
  `7a87777ac8437e32b5adf586924a53914ca313aa2d061defccdd0f28c82687be`.
- Cross-model endpoint differences use paired whole-base-world bootstrap only for persisted additive
  group effects (H1–H4 and H7). H5/H6 have aggregate probe R² artifacts without persisted row-level
  predictions and H8 is a joint gate, so their differences are descriptive rather than pseudo-paired.
- The frozen Phi result is Level 1: H1 is distinguishable but wrong-sign; H2/H3/H4/H7/H8 are not
  supported; H5 is supported; and H6 passes. The permitted interpretation is a broad second-family
  simple-transport null plus family-dependent residual decodability without a uniquely useful base lift,
  causal-use evidence, or universal ontology.
