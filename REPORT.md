# Geometry of Conditional Truth — Run Report

## 1. Executive result

This report is generated from run `gct-v0.1-db5a41461117` and recorded artifact hashes. The validation-selected transformer layer was 22; test data were evaluated only after the selection artifact was frozen. The run tested all five coordinate/control arms and both arbitrary and familiar-label versions. Across H1–H8, the status counts were {'not_supported': 7, 'control_pass': 1}. The conservative interpretation is Level 1 of 6. This is evidence about empirical transport proxies in one synthetic task and one model, not evidence that coherence proves truth or that the model literally contains a sheaf, bifibration, or universal truth manifold.

## 2. Environment

- Python: `3.12.13 (main, Mar  3 2026, 14:59:34) [Clang 21.1.4 ]`
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- GPU: `NVIDIA GeForce RTX 5060 Ti`; compute capability `(12, 0)`
- NVIDIA driver / PyTorch CUDA runtime: `610.88` / `13.0`
- PyTorch: `2.12.0+cu130`
- Model: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`
- Dtype/storage: `bfloat16` / `float16`
- Layers/hidden size: 36 / 2560
- Repository commit at report build: `62eaf2be781361694530f8d5d98e366bf87182ad`
- Config hash: `db5a414611170ba43e29ab33a3e2a614056b423ef072ab8e594f038a0c231018`
- Dependency lock hash: `a2b3ea8d47104ad3ecabb28b77c58305847059e5776e4550a287298834c5b6b0`

## 3. Completed implementation

The run includes deterministic ToyThermo oracle data; explicit, inferable, identical-prompt unobservable, irrelevant-Q, and semantic-renaming arms; grouped and held-out transformations; real all-layer anchor activations; deterministic answers; train-fit preprocessing; identity, mean-shift, PCA-affine, low-rank, and continuous-generator operators; transport, cycle, commuting-square, and matching proxies; residual probes/permutations; MDL sensitivity; grouped bootstrap uncertainty; behavior links; figures; and hash verification.

## 4. QC status

- Dataset logical hash: `dd44cbc000df7322f45cce1b7faef9cd0cc22290bcac5bb9d76fb95d6f2fd84f`
- Activation shards: 50, status `complete`
- Behavior parse failures: 887
- Common anchor token IDs: `[151667, 271, 151668, 271]`
- Frozen selection certifies test data used: `False`
- Probe permutation replicates: 1000
- Tests: `53 passed, 1 opt-in real-model test skipped in 3.93s` (`uv run pytest -q`)
- Lint: `All checks passed`
- Format: `84 files already formatted`
- Type checking: `No issues in 55 source files`
- Real-model integration: `1 passed in 7.21s; real Qwen3-4B, CUDA BF16, all 36 layers, deterministic generation`
- Split validation: `valid: 12,600 rows, 420 disjoint base-world groups, 840 unobservable pairs, disjoint magnitude holdouts`
- Deterministic regeneration: `two regenerations exactly matched dd44cbc000df7322f45cce1b7faef9cd0cc22290bcac5bb9d76fb95d6f2fd84f`
- Prompt-anchor audit: `10,080 unique prompts share suffix [151667, 271, 151668, 271]`
- Exact duplicate-prompt audit: `0/2,520 activation mismatches across embedding plus 36 layers; 0/2,520 response mismatches; unobservable subsets 0/840 and 0/840`
- Artifact verification: `valid=true, scientifically_complete=true, 7 stages verified, 0 errors, 0 warnings`
- Figure reproducibility: `five regenerated PNG hashes matched exactly across consecutive builds`
- Placeholder audit: `no TODO, FIXME, pass-body, mock-backend, dummy, or placeholder implementation remains`

Held-out primary behavior metrics (grouped bootstrap by base world):

- Mean absolute oracle error among parsed answers: 7.978 (95% CI [7.284, 8.726])
- Within-tolerance correctness over all prompts: 0.04417 (95% CI [0.02916, 0.06028])
- Nuisance answer-flip rate among parsed pairs: 0.3075 (95% CI [0.2659, 0.346])
- Substantive correction rate among parsed pairs: 0.3015 (95% CI [0.276, 0.328])

## 5. Dataset

The dataset contains 12600 prompt rows from 420 grouped base worlds. Split counts are `{'test': 3600, 'train': 7200, 'validation': 1800}`, arm counts are `{'explicit_coordinate': 2940, 'inferable_unnamed_coordinate': 1680, 'irrelevant_coordinate': 840, 'semantic_renaming': 6300, 'unobservable_coordinate': 840}`, and transformation counts are `{'concentration_shift': 840, 'fluid_swap': 840, 'identity': 3360, 'nuisance_inverse': 1680, 'nuisance_rewrite': 1680, 'pressure_shift': 3360, 'square_final': 840}`. The oracle is `toythermo-v1-fields-v1`, computed exclusively in Python. Transformation magnitudes, the JSON-like renderer, and Cyrene entity evaluation follow the frozen held-out design.

## 6. Preregistered hypotheses

| Hypothesis | Title | Status | Primary evidence |
|---|---|---|---|
| H1 | Nuisance versus substantive separability | not_supported | effect=1.521; 95% CI [1.494, 1.548] |
| H2 | Reusable transport | not_supported | effect=-0.2899; 95% CI [-0.3899, -0.1917] |
| H3 | Continuous composition | not_supported | effect=-0.7617; 95% CI [-0.943, -0.5924]; composition=2.334e-07; square=0.4581; composed-to-target=0.5453 |
| H4 | Structural defects predict behavior | not_supported | R² gain=-0.08419; prediction-error gain=-0.3428 [95% CI -0.4536, -0.228] |
| H5 | Inferable omitted coordinate in residuals | not_supported | test R²=-0.214; MAE=0.5545; permutation p=0.5634; R² 95% CI [-0.5605, 0.02679] |
| H6 | Unobservable-coordinate negative control | control_pass | test R²=-0.03577; MAE=0.539; permutation p=0.8472; R² 95% CI [-0.1559, -0.0005554] |
| H7 | Informative base lift | not_supported | effect=-0.06991; 95% CI [-0.08709, -0.05283]; Q structural=-0.03184; explicit behavior=3.359; Q behavior=2.262 |
| H8 | Semantic-renaming robustness | not_supported | renamed statuses=H2:not_supported, H5:not_supported, H6:control_pass, H7:not_supported; renamed H2 effect=-0.1252; H5 R²=-0.06671, p=0.04695; H6 R²=-0.03577, p=0.8561; H7 structural effect=-0.02544 |

Effect signs were fixed in advance: H1 is nuisance minus substantive displacement (support requires a wholly negative interval); H2/H3 are one minus candidate-to-baseline defect ratios (positive favors learned transport); H4 prediction-error gain is confounds-only error minus confounds-plus-defect error; and H7 gains are inferable-arm loss minus lifted-arm loss (positive favors the lift). Thus the negative H2/H3/H4/H7 values are evidence against, not for, their hypotheses. H3's near-zero operator-composition defect does not rescue its substantially worse prediction to observed targets.

For H7, the explicit-P structural 95% CI was [-0.08709, -0.05283], versus [-0.04721, -0.01794] for Q. Behavioral gains were 3.359 [2.349, 4.4] for explicit P and 2.262 [1.339, 3.198] for Q; the preregistered non-overlap/superiority rule failed.

For H8, renamed H2 had 95% CI [-0.2027, -0.0541]. Renamed H5's R² interval was [-0.3245, 0.1184]; despite p=0.04695, its negative point R² failed the joint decision rule. Renamed H7's explicit structural interval was [-0.05396, 0.003724], so replication remained unsupported.

The complete nested effects, null thresholds, behavior baselines, and H8 replication results are in `runs/gct-v0.1-db5a41461117/statistics/hypotheses.json`; no endpoint was removed because of its sign.

## 7. Key figures/tables

- [Validation layer selection](runs/gct-v0.1-db5a41461117/figures/validation_layer_selection.png)
- [Held-out transport operators](runs/gct-v0.1-db5a41461117/figures/test_transport_models.png)
- [Held-out hidden-pressure probes](runs/gct-v0.1-db5a41461117/figures/test_hidden_pressure_probes.png)
- [Held-out base-lift comparison](runs/gct-v0.1-db5a41461117/figures/test_base_lift.png)
- [Exploratory test layer scan with BH-FDR](runs/gct-v0.1-db5a41461117/figures/test_exploratory_layer_scan.png)
- Machine-readable metric tables: `runs/gct-v0.1-db5a41461117/metrics/*.parquet`
- Generator composition proxy: `runs/gct-v0.1-db5a41461117/metrics/generator_composition.parquet`
- MDL lambda sweep: `runs/gct-v0.1-db5a41461117/statistics/mdl_sensitivity.parquet`

Across every preregistered lambda from 0 to 1, the MDL proxy's minimum was the byte-identical unobservable condition in both worlds. This is the expected zero-defect degeneracy of the negative control, not evidence for a discovered ontology; the full sweep is reported rather than used to override H5/H7.

## 8. Negative controls

The identical-prompt unobservable control status was `control_pass` (test R² -0.03577; null 95th percentile -0.03577). H7's explicit-P structural gain was -0.06991, versus -0.03184 for irrelevant Q; the preregistered superiority rule was not met. In the familiar-label world, H2, H5, and H7 remained unsupported while H6 again passed. A failed H6 would invalidate positive hidden-coordinate interpretation until leakage was resolved.

Raw batched extraction showed numerical batch-boundary sensitivity in 34 of 2520 repeated activation rows (maximum stored difference 4); generation differed in 9 repeated rows. Because identical token sequences cannot contain a row-specific hidden coordinate, the preregistered `canonical_first_occurrence` policy rewrote each duplicate prompt from its first dataset occurrence before analysis. The final exact audit is recorded above. An earlier batch-sensitive run was superseded rather than reported.

## 9. Interpretation level

**Level 1.** The level follows the preregistered evidence ladder mechanically. H1's interval excludes zero, so nuisance and substantive transformations are distinguishable on held-out groups, but the sign is opposite the preregistered invariance expectation: nuisance displacement is larger. H2 fails, so no reusable held-out transport law is established and no higher level is claimed. This does not generalize beyond this model, prompt protocol, representation anchor, or synthetic world.

## 10. Prior-art update

The nearest work includes context-conditioned truth-vector geometry, transformation-equivariant representation learning, causal representation identifiability, continuous latent reasoning, activation patching, and sheaf-theoretic contextuality. `docs/LITERATURE_MAP.md` and `docs/PRIOR_ART_DIFF.md` distinguish this controlled residual-transport/base-lift protocol. No universal novelty claim is made.

## 11. Limitations

One 4B instruction model, one anchor, a synthetic arithmetic world, linear/reduced-rank operators, representation-dependent distances, observational activations, finite permutation/bootstrap precision, and possible prompt-computation confounds limit inference. Familiar labels may invoke pretraining priors even though the prompt overrides chemistry. See `docs/LIMITATIONS.md`.

## 12. Next experiment

The single most informative follow-up is a preregistered replication on a second model family at matched data and compute, preserving the identical-prompt negative control and frozen analysis.

## 13. Git state

- Commit at report build: `62eaf2be781361694530f8d5d98e366bf87182ad`
- Remote configured: `True` (`origin (n:nshkrdotcom/gct.git)`)
- Push attempted: `True`
- Push verified: `True` at `62eaf2be781361694530f8d5d98e366bf87182ad`
