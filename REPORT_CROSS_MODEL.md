# Geometry of Conditional Truth — Cross-model report

## Executive result

Phi reproduces the broad Qwen v0 simple-state-transport null (H1 wrong-sign; H2/H3/H4/H7/H8 unsupported), while Phi alone supports H5: model-family-dependent latent residual decodability without evidence that an explicit base lift uniquely repairs structure. H6 is bit-identical in both families because its identical-prompt arm has a zero residual by construction, so it qualifies neither result. Phi behavior remains near floor, limiting behavioral endpoints; no universal truth geometry, causal use, or ontology is established.

Both models used the exact same 12,600 stable sample IDs, 420 base-world groups, split assignments, prompts, controls, metrics, and H1–H8 rules. Endpoint decisions remain model-specific; the paired contrasts below are secondary and do not revise either preregistered decision.

## H1–H8 comparison

| Endpoint | Qwen effect (95% CI) | Qwen status | Phi effect (95% CI) | Phi status | Phi − Qwen (paired 95% CI) | Sign/status |
|---|---:|---|---:|---|---:|---|
| H1 | 1.521 [1.494, 1.548] | not_supported | 0.8897 [0.8654, 0.914] | not_supported | -0.6315 [-0.6636, -0.5993] | sign=True; status=True |
| H2 | -0.2899 [-0.3899, -0.1917] | not_supported | -0.2115 [-0.3047, -0.121] | not_supported | 0.07846 [-0.01582, 0.1828] | sign=True; status=True |
| H3 | -0.7617 [-0.943, -0.5924] | not_supported | -1.562 [-1.994, -1.189] | not_supported | -0.8001 [-1.129, -0.4968] | sign=True; status=True |
| H4 | -0.3428 [-0.4536, -0.228] | not_supported | -0.09896 [-0.2424, 0.05096] | not_supported | 0.2438 [-0.02436, 0.5013] | sign=True; status=True |
| H5 | -0.214 [-0.5605, 0.02679] | not_supported | 0.2878 [0.1523, 0.3869] | supported | — — | sign=False; status=False |
| H6 | -0.03577 [-0.1559, -0.0005554] | control_pass | -0.03577 [-0.1559, -0.0005554] | control_pass | — — | sign=True; status=True |
| H7 | -0.06991 [-0.08709, -0.05283] | not_supported | 0.1273 [0.09863, 0.1568] | not_supported | 0.1972 [0.1622, 0.2317] | sign=False; status=True |
| H8 | — — | not_supported | — — | not_supported | — — | sign=None; status=True |

H1 uses nuisance minus substantive displacement, so its wholly positive interval is opposite the preregistered theory in both models. H2/H3/H4/H7 positive effects favor the theory; negative values do not. H4's paired contrast uses the persisted grouped absolute-prediction-error gain. H5/H6 are persisted aggregate probe R² endpoints without per-row prediction artifacts, and H8 is a joint gate; their cross-model differences are therefore descriptive (`—`) rather than pseudo-paired.

Phi's H5 result is the only endpoint-status divergence: inferable hidden-pressure residual decoding was supported with R² 0.2878 while Qwen's R² was −0.2140. H6 is bit-identical in both models because its arm renders byte-identical prompts, so its residual is identically zero and its statistic depends on the shared labels alone; it is a pipeline check, not an independent leakage test that this divergence survived. This supports family-dependent residual association/decodability, not causal use or ontology discovery. H7 and H8 failed, so the signal did not establish a uniquely useful explicit-coordinate lift or semantic-robust transport structure.

## Behavior

| Metric | Qwen | Phi | Phi − Qwen (paired 95% CI) |
|---|---:|---:|---:|
| parse_failure_rate_all_prompts | 0.04806 | 0.01222 | -0.03583 [-0.05111, -0.02139] |
| mean_absolute_oracle_error_parsed | 7.978 | 10.79 | 2.808 [2.189, 3.398] |
| within_tolerance_rate_all_prompts | 0.04417 | 0.02639 | -0.01778 [-0.03389, -0.003333] |
| nuisance_answer_flip_rate_parsed_pairs | 0.3075 | 0.5168 | 0.2093 [0.1358, 0.2803] |
| substantive_correction_rate_parsed_pairs | 0.3015 | 0.1821 | -0.1194 [-0.1553, -0.08484] |

Phi parsed more outputs but was less accurate: its parsed-answer MAE was higher and its all-prompt correctness and substantive correction rates were lower. Both models are behaviorally weak on this numeric protocol, so H4/H7 behavioral interpretation is limited. Representation-only H1–H3/H5/H6 decisions remain reportable under the frozen matrix.

## Selected depth and resources

| Model | Selected layer | Layers | Normalized depth | Hidden size | Activation shards/bytes | Full-run parse failures |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 22 | 36 | 0.6286 | 2560 | 50 / 2386972400 | 887 |
| Phi-4-mini | 13 | 32 | 0.4194 | 3072 | 50 / 2554702800 | 235 |

Phi's unquantized BF16 checkpoint has 3836021760 parameters and 7672066216 checkpoint weight bytes. Its frozen batch probe used batch 4 with peak CUDA allocation 8398130688 bytes. The historical Qwen manifests did not record comparable checkpoint-byte or peak-memory fields, so those cells are intentionally not estimated after the fact.

Absolute layer indices are not treated as anatomically equivalent. Qwen selected 22/35 (normalized 0.6286); Phi selected 13/31 (0.4194). The normalized all-layer overlay is exploratory test analysis, not confirmatory selection.

## Metric/control details

Nuisance/substantive displacement summaries (means; full distribution summaries are machine-readable):

| Metric | Model | Nuisance | Substantive |
|---|---|---:|---:|
| cosine | Qwen3-4B | 0.02051 | 0.001603 |
| cosine | Phi-4-mini | 0.006285 | 0.003558 |
| standardized_l2 | Qwen3-4B | 1.515 | 0.3742 |
| standardized_l2 | Phi-4-mini | 1.068 | 0.5714 |
| whitened_l2 | Qwen3-4B | 2.011 | 0.4902 |
| whitened_l2 | Phi-4-mini | 1.611 | 0.7213 |

Held-out low-rank candidate / validation-selected baseline defect ratios (<1 favors the candidate):

| Metric | Qwen ratio (baseline) | Phi ratio (baseline) |
|---|---:|---:|
| cosine | 1.19 (identity) | 1.114 (mean_shift) |
| standardized_l2 | 1.093 (identity) | 1.062 (mean_shift) |
| whitened_l2 | 1.2 (identity) | 1.066 (mean_shift) |

The cross-model paired endpoint contrasts use the frozen primary whitened metric. H5/H6 probe results, H7 explicit-versus-Q base lifts, and H8 renamed replication appear in the endpoint table and model-specific reports. The identical-prompt control is degenerate by construction in both primary and renamed worlds — zero residual, intercept-only probe, an endpoint invariant to model and layer — so its agreement across families is arithmetic rather than evidential. The exact post-canonicalization duplicate audit found zero mismatches for Phi, and `gct verify` asserts the stored control probes are exactly degenerate.

## Figures

- [H1 Displacement By Model](runs/gct-v0.2-phi4mini-7a87777ac843/cross_model/figures/h1_displacement_by_model.png)
- [H2 Transport Ratio By Model](runs/gct-v0.2-phi4mini-7a87777ac843/cross_model/figures/h2_transport_ratio_by_model.png)
- [H5 Probe Curves By Model](runs/gct-v0.2-phi4mini-7a87777ac843/cross_model/figures/h5_probe_curves_by_model.png)
- [H7 Base Lift By Model](runs/gct-v0.2-phi4mini-7a87777ac843/cross_model/figures/h7_base_lift_by_model.png)
- [Exploratory Normalized Depth](runs/gct-v0.2-phi4mini-7a87777ac843/cross_model/figures/exploratory_normalized_depth.png)
- [Behavior By Model](runs/gct-v0.2-phi4mini-7a87777ac843/cross_model/figures/behavior_by_model.png)
- [Endpoint Status Matrix](runs/gct-v0.2-phi4mini-7a87777ac843/cross_model/figures/endpoint_status_matrix.png)

## Interpretation matrix

The applicable frozen matrix rows are: (1) H1 wrong-sign with H2+ null, a broad second-family replication of the simple v0 state-transport null; (2) H5 positive with H7 failing, latent residual decodability without evidence that explicit base lift uniquely repairs structure; and (3) Phi behavior near floor, which limits behavioral endpoints but does not license prompt redesign or remove representational tests. The frozen matrix names H6 alongside that second row, but H6 could not have failed and adds no support to it.

The result does not disprove GCT broadly, prove universal truth geometry, or show causal use. A future v0.3 would require a new preregistration before testing a changed representational object such as trajectories, nonlinear local transports, circuits, Jacobians, or interventions.

## Reproducibility

Baseline run: `gct-v0.1-db5a41461117`. Replication run: `gct-v0.2-phi4mini-7a87777ac843`. Dataset logical hash: `dd44cbc000df7322f45cce1b7faef9cd0cc22290bcac5bb9d76fb95d6f2fd84f`. The machine-readable summary and paired base-world tables are in `runs/gct-v0.2-phi4mini-7a87777ac843/cross_model/`; joins are by stable ID/base-world ID, never row order.
