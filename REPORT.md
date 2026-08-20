# Geometry of Conditional Truth — Run Report

## 1. Executive result

This report is generated from run `gct-v0.1-ci-cb7d9a13fa85` and recorded artifact hashes. The validation-selected transformer layer was 24; test data were evaluated only after the selection artifact was frozen. The run tested all five coordinate/control arms and both arbitrary and familiar-label versions. Across H1–H8, the status counts were {'development_only': 8}. The conservative interpretation is Level 0 of 6. This is evidence about empirical transport proxies in one synthetic task and one model, not evidence that coherence proves truth or that the model literally contains a sheaf, bifibration, or universal truth manifold.

## 2. Environment

- Python: `3.12.13 (main, Mar  3 2026, 14:59:34) [Clang 21.1.4 ]`
- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- Model: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`
- Dtype/storage: `bfloat16` / `float16`
- Layers/hidden size: 36 / 2560
- Repository commit at report build: `7a3922abd5af329e4c4eb02e39e0518ed00ab550`
- Config hash: `cb7d9a13fa859f42cf48ee8cb1710854f888400923e315fec308b0f801436964`
- Dependency lock hash: `a2b3ea8d47104ad3ecabb28b77c58305847059e5776e4550a287298834c5b6b0`

## 3. Completed implementation

The run includes deterministic ToyThermo oracle data; explicit, inferable, identical-prompt unobservable, irrelevant-Q, and semantic-renaming arms; grouped and held-out transformations; real all-layer anchor activations; deterministic answers; train-fit preprocessing; identity, mean-shift, PCA-affine, low-rank, and continuous-generator operators; transport, cycle, commuting-square, and matching proxies; residual probes/permutations; MDL sensitivity; grouped bootstrap uncertainty; behavior links; figures; and hash verification.

## 4. QC status

- Dataset logical hash: `e5b9a0274c482455091abbc292d783ffd19a05c51134006854e9c6d8b2862f3a`
- Activation shards: 30, status `complete`
- Behavior parse failures: 227
- Common anchor token IDs: `[151667, 271, 151668, 271]`
- Frozen selection certifies test data used: `False`
- Probe permutation replicates: 10
- Unit/lint/type gate results are recorded in the repository final handoff and may be reproduced with the README commands.

## 5. Dataset

The dataset contains 240 prompt rows from 8 grouped base worlds. Split counts are `{'test': 60, 'train': 120, 'validation': 60}` and arm counts are `{'explicit_coordinate': 56, 'inferable_unnamed_coordinate': 32, 'irrelevant_coordinate': 16, 'semantic_renaming': 120, 'unobservable_coordinate': 16}`. The oracle is ToyThermo v1, computed exclusively in Python. Transformation magnitudes, the JSON-like renderer, and Cyrene entity evaluation follow the frozen held-out design.

## 6. Preregistered hypotheses

| Hypothesis | Title | Status | Primary evidence |
|---|---|---|---|
| H1 | Nuisance versus substantive separability | development_only | effect=0.2375; 95% CI [0.2374, 0.2377] |
| H2 | Reusable transport | development_only | effect=-1.26; 95% CI [-2.067, -0.4519] |
| H3 | Continuous composition | development_only | effect=-0.3908; 95% CI [-0.7269, -0.05474] |
| H4 | Structural defects predict behavior | development_only | held-out absolute-error prediction versus trivial confounds |
| H5 | Inferable omitted coordinate in residuals | development_only | test R²=-1.778; MAE=0.7398; permutation p=0.09091 |
| H6 | Unobservable-coordinate negative control | development_only | test R²=-1.374; MAE=0.6703; permutation p=0.1818 |
| H7 | Informative base lift | development_only | explicit-coordinate gain versus inferable omission and irrelevant-Q lift |
| H8 | Semantic-renaming robustness | development_only | renamed-world repetition of H2, H5, and H7 |

The complete nested effects, null thresholds, behavior baselines, and H8 replication results are in `statistics/hypotheses.json`; no endpoint was removed because of its sign.

## 7. Key figures/tables

- [Validation layer selection](figures/validation_layer_selection.png)
- [Held-out transport operators](figures/test_transport_models.png)
- [Held-out hidden-pressure probes](figures/test_hidden_pressure_probes.png)
- [Held-out base-lift comparison](figures/test_base_lift.png)
- [Exploratory test layer scan with BH-FDR](figures/test_exploratory_layer_scan.png)
- Machine-readable metric tables: `metrics/*.parquet`
- MDL lambda sweep: `statistics/mdl_sensitivity.parquet`

## 8. Negative controls

The identical-prompt unobservable control status was `development_only` (test R² -1.374; null 95th percentile -1.377). The irrelevant-Q comparison is reported inside H7, and familiar-label semantic renaming inside H8. A failed H6 invalidates positive hidden-coordinate interpretation until leakage is resolved.

## 9. Interpretation level

**Level 0.** The level follows the preregistered evidence ladder mechanically. It does not generalize beyond this model, prompt protocol, representation anchor, or synthetic world.

## 10. Prior-art update

The nearest work includes context-conditioned truth-vector geometry, transformation-equivariant representation learning, causal representation identifiability, continuous latent reasoning, activation patching, and sheaf-theoretic contextuality. `docs/LITERATURE_MAP.md` and `docs/PRIOR_ART_DIFF.md` distinguish this controlled residual-transport/base-lift protocol. No universal novelty claim is made.

## 11. Limitations

One 4B instruction model, one anchor, a synthetic arithmetic world, linear/reduced-rank operators, representation-dependent distances, observational activations, finite permutation/bootstrap precision, and possible prompt-computation confounds limit inference. Familiar labels may invoke pretraining priors even though the prompt overrides chemistry. See `docs/LIMITATIONS.md`.

## 12. Next experiment

The single most informative follow-up is a preregistered replication on a second model family at matched data and compute, preserving the identical-prompt negative control and frozen analysis.

## 13. Git state

- Commit at report build: `7a3922abd5af329e4c4eb02e39e0518ed00ab550`
- Remote/push status is reported in the final handoff; report generation does not mutate remotes.
