# Geometry of Conditional Truth — Model #2 Replication Report

## 1. Executive result

This report is generated from run `gct-v0.2-phi4mini-7a87777ac843` and recorded artifact hashes. The validation-selected transformer layer was 13; test data were evaluated only after the selection artifact was frozen. The run tested all five coordinate/control arms and both arbitrary and familiar-label versions. Across H1–H8, the status counts were {'not_supported': 6, 'supported': 1, 'control_pass': 1}. The conservative interpretation is Level 1 of 6. The broad v0 transport null replicates across a second model family, while H5 shows a control-safe family difference in residual decodability. This is evidence about empirical transport proxies in one synthetic task and one model, not evidence that coherence proves truth or that the model literally contains a sheaf, bifibration, or universal truth manifold.

## 2. Environment

- Python: `3.12.13 (main, Mar  3 2026, 14:59:34) [Clang 21.1.4 ]`
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- GPU: `NVIDIA GeForce RTX 5060 Ti`; compute capability `(12, 0)`
- NVIDIA driver / PyTorch CUDA runtime: `610.88` / `13.0`
- PyTorch: `2.12.0+cu130`
- Model: `microsoft/Phi-4-mini-instruct` at revision `4b00ec8714b0cb224e4fb33380cbf0919f177f3e`
- Model adapter protocol / remote code: `phi4mini-v2` / `True` (immutable revision only)
- Dtype/storage: `bfloat16` / `float16`
- Layers/hidden size: 32 / 3072
- Runtime-discovered parameters/checkpoint bytes: `3836021760` / `7672066216`
- Hashed checkpoint/config/tokenizer/code files: `23`
- Repository commit at report build: `4742559aab9f82d01d7aa43b00f832559dd4ebbc`
- Config hash: `7a87777ac8437e32b5adf586924a53914ca313aa2d061defccdd0f28c82687be`
- Dependency lock hash: `f5d6ff10b891fbb7246e548626cbafe03b971b494abfb102e10f2fab8e0f051c`

## 3. Completed implementation

The run includes deterministic ToyThermo oracle data; explicit, inferable, identical-prompt unobservable, irrelevant-Q, and semantic-renaming arms; grouped and held-out transformations; real all-layer anchor activations; deterministic answers; train-fit preprocessing; identity, mean-shift, PCA-affine, low-rank, and continuous-generator operators; transport, cycle, commuting-square, and matching proxies; residual probes/permutations; MDL sensitivity; grouped bootstrap uncertainty; behavior links; figures; and hash verification.

## 4. QC status

- Dataset logical hash: `dd44cbc000df7322f45cce1b7faef9cd0cc22290bcac5bb9d76fb95d6f2fd84f`
- Activation shards: 50, status `complete`
- Behavior parse failures: 235
- Common anchor token IDs: `[200020, 200019, 176019, 28]`
- Frozen selection certifies test data used: `False`
- Probe permutation replicates: 1000
- Model-adapter anchor audit: `a37bc57cc578bfe16936c646812752d14a493c895ea7772a026facc3be7f2584`; token suffix `[200020, 200019, 176019, 28]`
- Pre-test preregistration freeze: `d651077214ff4d6762f580594e1d33107736370c48caf108fdd727ba5e11390e`
- Tests: `72 passed, 2 opt-in real-model tests skipped in 11.46s` (`uv run pytest -q`)
- Lint: `All checks passed`
- Format: `97 files already formatted`
- Type checking: `No issues in 59 source files`
- Real-model integration: `Qwen: 1 passed in 6.29s; Phi: 1 passed in 6.32s; CUDA BF16, every layer, deterministic repeated generation`
- Split validation: `valid: 12,600 rows, 420 disjoint base-world groups, 840 unobservable pairs, exact frozen split/magnitude holdouts`
- Deterministic regeneration: `two regenerations matched dd44cbc...84f; Model #1/Model #2 Parquet SHA-256 both 59bd8a17...c3f4 and stable IDs/groups/splits matched exactly`
- Prompt-anchor audit: `12,600 rows/10,080 unique prompts; suffix [200020, 200019, 176019, 28]; all arms/worlds/renderers/transforms; 0 token-control mismatches`
- Exact duplicate-prompt audit: `0/2,520 activation-row mismatches across 83,160 embedding/layer comparisons; 0/2,520 response mismatches; unobservable subsets 0/840 and 0/840`
- Artifact verification: `Model #2 valid=true/scientifically_complete=true, 7 stages, 0 errors/warnings; Model #1 valid and fingerprint unchanged at 9ae67526...7dd0e`
- Figure reproducibility: `five Model #2 and seven cross-model regenerated PNG hashes matched exactly across consecutive builds`
- Placeholder audit: `no unfinished implementation marker, production mock/dummy backend, whitespace error, or endpoint placeholder remains`

Held-out primary behavior metrics (grouped bootstrap by base world):

- Mean absolute oracle error among parsed answers: 10.79 (95% CI [10.17, 11.41])
- Within-tolerance correctness over all prompts: 0.02639 (95% CI [0.01667, 0.03751])
- Nuisance answer-flip rate among parsed pairs: 0.5168 (95% CI [0.4475, 0.5867])
- Substantive correction rate among parsed pairs: 0.1821 (95% CI [0.1566, 0.2078])

## 5. Dataset

The dataset contains 12600 prompt rows from 420 grouped base worlds. Split counts are `{'test': 3600, 'train': 7200, 'validation': 1800}`, arm counts are `{'explicit_coordinate': 2940, 'inferable_unnamed_coordinate': 1680, 'irrelevant_coordinate': 840, 'semantic_renaming': 6300, 'unobservable_coordinate': 840}`, and transformation counts are `{'concentration_shift': 840, 'fluid_swap': 840, 'identity': 3360, 'nuisance_inverse': 1680, 'nuisance_rewrite': 1680, 'pressure_shift': 3360, 'square_final': 840}`. The oracle is `toythermo-v1-fields-v1`, computed exclusively in Python. Transformation magnitudes, the JSON-like renderer, and Cyrene entity evaluation follow the frozen held-out design.
Model #2 reused the Model #1 Parquet sample byte-for-byte from the immutable `gct-v0.1-db5a41461117` evidence run. Stable row, group, and split IDs were checked for exact equality before inference; no sample was regenerated or re-randomized.

## 6. Preregistered hypotheses

| Hypothesis | Title | Status | Primary evidence |
|---|---|---|---|
| H1 | Nuisance versus substantive separability | not_supported | effect=0.8897; 95% CI [0.8654, 0.914] |
| H2 | Reusable transport | not_supported | effect=-0.2115; 95% CI [-0.3047, -0.121] |
| H3 | Continuous composition | not_supported | effect=-1.562; 95% CI [-1.994, -1.189]; composition=4.628e-07; square=0.5133; composed-to-target=0.837 |
| H4 | Structural defects predict behavior | not_supported | R² gain=-0.0715; prediction-error gain=-0.09896 [95% CI -0.2424, 0.05096] |
| H5 | Inferable omitted coordinate in residuals | supported | test R²=0.2878; MAE=0.4573; permutation p=0.000999; R² 95% CI [0.1523, 0.3869] |
| H6 | Unobservable-coordinate negative control | control_pass | test R²=-0.03577; MAE=0.539; permutation p=0.8472; R² 95% CI [-0.1559, -0.0005554] |
| H7 | Informative base lift | not_supported | effect=0.1273; 95% CI [0.09863, 0.1568]; Q structural=0.05492; behavior=explicit=-5.878 [95% CI -7.244, -4.559]; Q=0.4737 [95% CI -0.6132, 1.501] |
| H8 | Semantic-renaming robustness | not_supported | renamed statuses=H2:not_supported, H5:supported, H6:control_pass, H7:not_supported; renamed H2 effect=-0.04428; H5 R²=0.1626, p=0.001998; H6 R²=-0.03577, p=0.8561; H7 structural effect=0.1408 |

### Frozen Model #1 comparison

| Endpoint | Model #1 effect | Model #1 status | Model #2 effect | Model #2 status |
|---|---:|---|---:|---|
| H1 | 1.521 | not_supported | 0.8897 | not_supported |
| H2 | -0.2899 | not_supported | -0.2115 | not_supported |
| H3 | -0.7617 | not_supported | -1.562 | not_supported |
| H4 | -0.3428 | not_supported | -0.09896 | not_supported |
| H5 | -0.214 | not_supported | 0.2878 | supported |
| H6 | -0.03577 | control_pass | -0.03577 | control_pass |
| H7 | -0.06991 | not_supported | 0.1273 | not_supported |
| H8 | — | not_supported | — | not_supported |

H1 is wrong-sign in both families and H2/H3/H4/H7/H8 remain unsupported. H5 is the sole status divergence: Phi supports residual hidden-pressure decoding while Qwen does not; H6 passes for both. The paired confidence intervals and behavior/resource contrasts are in `REPORT_CROSS_MODEL.md`.

Effect signs were fixed in advance: H1 is nuisance minus substantive displacement (support requires a wholly negative interval); H2/H3 are one minus candidate-to-baseline defect ratios (positive favors learned transport); H4 prediction-error gain is confounds-only error minus confounds-plus-defect error; and H7 gains are inferable-arm loss minus lifted-arm loss (positive favors the lift). Thus the negative H2/H3/H4/H7 values are evidence against, not for, their hypotheses. H3's near-zero operator-composition defect does not rescue its substantially worse prediction to observed targets.

For H7, the explicit-P structural 95% CI was [0.09863, 0.1568], versus [0.03366, 0.07682] for Q. Behavioral gains were explicit=-5.878 [95% CI -7.244, -4.559]; Q=0.4737 [95% CI -0.6132, 1.501]; the preregistered non-overlap/superiority rule failed.

For H8, renamed H2 had 95% CI [-0.101, 0.008675]. Renamed H5's R² interval was [-0.0539, 0.3195]; p=0.001998, with recorded status `supported`. Renamed H7's explicit structural interval was [0.1142, 0.1674], so replication remained unsupported.

The complete nested effects, null thresholds, behavior baselines, and H8 replication results are in `runs/gct-v0.2-phi4mini-7a87777ac843/statistics/hypotheses.json`; no endpoint was removed because of its sign.

## 7. Key figures/tables

- [Validation layer selection](runs/gct-v0.2-phi4mini-7a87777ac843/figures/validation_layer_selection.png)
- [Held-out transport operators](runs/gct-v0.2-phi4mini-7a87777ac843/figures/test_transport_models.png)
- [Held-out hidden-pressure probes](runs/gct-v0.2-phi4mini-7a87777ac843/figures/test_hidden_pressure_probes.png)
- [Held-out base-lift comparison](runs/gct-v0.2-phi4mini-7a87777ac843/figures/test_base_lift.png)
- [Exploratory test layer scan with BH-FDR](runs/gct-v0.2-phi4mini-7a87777ac843/figures/test_exploratory_layer_scan.png)
- Machine-readable metric tables: `runs/gct-v0.2-phi4mini-7a87777ac843/metrics/*.parquet`
- Generator composition proxy: `runs/gct-v0.2-phi4mini-7a87777ac843/metrics/generator_composition.parquet`
- MDL lambda sweep: `runs/gct-v0.2-phi4mini-7a87777ac843/statistics/mdl_sensitivity.parquet`

Across every preregistered lambda from 0 to 1, the MDL proxy's minimum was the byte-identical unobservable condition in both worlds. This is the expected zero-defect degeneracy of the negative control, not evidence for a discovered ontology; the full sweep is reported rather than used to override H5/H7.

## 8. Negative controls

The identical-prompt unobservable control status was `control_pass` (test R² -0.03577; null 95th percentile -0.03577). H7's explicit-P structural gain was 0.1273, versus 0.05492 for irrelevant Q; the preregistered superiority rule was not met. In the familiar-label world, nested statuses were H2=`not_supported`, H5=`supported`, H6=`control_pass`, and H7=`not_supported`; the joint H8 gate remained unsupported. A failed H6 would invalidate positive hidden-coordinate interpretation until leakage was resolved.

Raw batched extraction showed numerical batch-boundary sensitivity in 34 of 2520 repeated activation rows (maximum stored difference 2); generation differed in 6 repeated rows. Because identical token sequences cannot contain a row-specific hidden coordinate, the preregistered `canonical_first_occurrence` policy rewrote each duplicate prompt from its first dataset occurrence before analysis. The final exact audit is recorded above. An earlier adapter-generation run with an incomplete EOS configuration was superseded before any full behavior shard or test metric existed.

## 9. Interpretation level

**Level 1.** The level follows the preregistered evidence ladder mechanically. H1's interval excludes zero, so nuisance and substantive transformations are distinguishable on held-out groups, but the sign is opposite the preregistered invariance expectation: nuisance displacement is larger. H2 fails, so no reusable held-out transport law is established and no higher level is claimed. This does not generalize beyond this model, prompt protocol, representation anchor, or synthetic world.

## 10. Prior-art update

The nearest work includes context-conditioned truth-vector geometry, transformation-equivariant representation learning, causal representation identifiability, continuous latent reasoning, activation patching, and sheaf-theoretic contextuality. `docs/LITERATURE_MAP.md` and `docs/PRIOR_ART_DIFF.md` distinguish this controlled residual-transport/base-lift protocol. No universal novelty claim is made.

## 11. Limitations

Two approximately 4B instruction-model families now share the protocol, but Model #2 still uses one checkpoint per family, one anchor, and a synthetic arithmetic world; linear/reduced-rank operators, representation-dependent distances, observational activations, finite permutation/bootstrap precision, and possible prompt-computation confounds limit inference. Familiar labels may invoke pretraining priors even though the prompt overrides chemistry. See `docs/LIMITATIONS.md`.

## 12. Next experiment

The single most informative follow-up is a new v0.3 preregistration that changes the representational object—such as trajectories or causal activation effects—while preserving the identical-prompt negative control and frozen train/validation/test discipline.

## 13. Git state

- Commit at report build: `4742559aab9f82d01d7aa43b00f832559dd4ebbc`
- Remote configured: `True` (`origin (n:nshkrdotcom/gct.git)`)
- Push attempted: `True`
- Push verified: `True` at `4742559aab9f82d01d7aa43b00f832559dd4ebbc`
