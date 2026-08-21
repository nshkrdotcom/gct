# Exploratory surface-feature baseline

**This analysis is exploratory, post-hoc, and non-confirmatory.** It is not in the frozen
`preregistration` mapping, it was added after the test metrics were known, and it changes no
H1–H8 status, no decision rule, and no Level assessment. It is recorded here so the H5 endpoint
can be read with a reference point it previously lacked.

## The gap it fills

In the inferable arm the prompt prints the calibration reading verbatim:

```text
Calibration law: R = 40 + 7 ln(P).
...
Fluid: Aquila; Concentration M: 0.71131488; Calibration reading R: 41.65338007.
```

`P = exp((R - 40) / 7)` is an exact inverse, so `P` is not withheld from the prompt in any
information-theoretic sense; only its name and its decimal expansion are. H5 reports that `P` is
decodable from held-out transport residuals. Nothing in the protocol reported what a trivial
reader of the prompt's own numerals achieves on the same held-out rows, so there was no way to
tell whether that R² described latent structure or the linear readability of a number written in
the input.

## What is measured

```bash
uv run gct probes baseline --run gct-v0.1-db5a41461117
uv run gct probes baseline --run gct-v0.2-phi4mini-7a87777ac843
```

The command reads `dataset/samples.parquet` and `config.yaml` only. It loads no activations and
no model.

Features are the numeric literals rendered into each prompt's `Context:` block, which is the only
place a per-sample numeral appears; the law block above it is constant within an arm. Each
rendered label maps to one canonical feature across the primary and renamed worlds
(`Fluid`/`Entity`, `Concentration M`/`Composition Y`, `Pressure P`/`Control X`, `Calibration
reading R`/`Proxy G`, `Nuisance Q`/`Nuisance W`). An unrecognized label raises rather than being
dropped. Only the literals an arm actually renders are used, so the arms differ exactly as the
prompts do:

| Arm | Features used |
|---|---|
| explicit | explicit coordinate, concentration, entity one-hot |
| inferable | calibration reading, concentration, entity one-hot |
| irrelevant | calibration reading, concentration, nuisance, entity one-hot |
| unobservable | concentration, entity one-hot |

The estimator and the uncertainty machinery are the residual probe's, so the numbers are
comparable: `ridge_coefficients` over train-standardized features, one predeclared penalty
selected on validation rows alone, the same grouped `base_world_id` splits, the same
`_group_permute` grouped permutation null with a refit and reselection on every replicate, and
the same grouped bootstrap over base worlds. The entity vocabulary is fitted on training rows, so
the held-out Cyrene entity leaves its indicators at zero. Row counts match the residual probe
exactly in all eight groups of both completed runs (240 train / 60 validation / 120 test).

Both sides of the comparison are linear. Because `R` inverts to `P` exactly, a nonlinear surface
reader would do strictly better, so the figures below are a **lower bound** on trivial surface
recoverability.

## Results

Held-out test rows, primary world. Surface columns are from
`runs/<id>/exploratory/surface_baseline/results.parquet`; residual columns are the recorded
confirmatory probe R² at each run's validation-selected layer (Qwen 22, Phi 13).

| Arm | Surface R² | 95% CI | Surface MAE | Surface p | Residual R² (Qwen) | Residual R² (Phi) |
|---|---:|---:|---:|---:|---:|---:|
| explicit | 1.000000 | [1.0000, 1.0000] | 0.000002 | 0.000999 | −0.0990 | 0.3247 |
| inferable | 0.887360 | [0.8618, 0.9105] | 0.169950 | 0.000999 | −0.2140 | **0.2878** |
| irrelevant | 0.887616 | [0.8632, 0.9108] | 0.169792 | 0.000999 | −0.0959 | 0.2354 |
| unobservable | −0.043488 | [−0.1540, 0.0106] | 0.539559 | 0.610390 | −0.0358 | −0.0358 |

Renamed world:

| Arm | Surface R² | 95% CI | Surface MAE | Surface p | Residual R² (Qwen) | Residual R² (Phi) |
|---|---:|---:|---:|---:|---:|---:|
| explicit | 1.000000 | [1.0000, 1.0000] | 0.000002 | 0.000999 | −0.2128 | 0.3795 |
| inferable | 0.887360 | [0.8631, 0.9098] | 0.169950 | 0.000999 | −0.0667 | 0.1626 |
| irrelevant | 0.887616 | [0.8636, 0.9098] | 0.169792 | 0.000999 | −0.0684 | 0.2276 |
| unobservable | −0.043488 | [−0.1558, 0.0113] | 0.539559 | 0.614386 | −0.0358 | −0.0358 |

`p = 0.000999` is the 1/1001 floor at 1,000 permutation replicates.

The surface results are byte-identical between the two runs — the two `results.parquet` files
share a SHA-256 — because the baseline is a function of the dataset alone and Model #2 reuses the
Model #1 sample Parquet byte-for-byte. That is the dataset-reuse invariant, not an agreement
between models.

## Reading

The baseline is not near zero and it is not merely comparable to the residual probe. In the arm
that carries H5, a linear read of the prompt's own numerals reaches R² 0.8874 and MAE 0.1700 with
no model at all, against the supporting residual probe's R² 0.2878 and MAE 0.4573 on the same 120
held-out rows.

What this supports:

- The quantity H5 recovers is not hidden information. It is arithmetic on a number printed in the
  input, and it is already linearly readable there roughly three times better than from the
  transport residual.
- H5's supported status therefore cannot be read as evidence that the model inferred and
  represented an omitted coordinate that would otherwise have been inaccessible. **H5 is
  materially weaker than it currently reads.**
- The arm contrasts behave exactly as the prompts imply: the explicit arm, which prints `P`, is
  recovered to R² 1.0; the unobservable arm, which prints no pressure-bearing literal, fails at
  R² −0.043 with p 0.61.

What this does not support:

- It does not show that the residual signal *is* the surface feature. The two probes read
  different objects. Establishing that the residual carries nothing about `P` beyond what the
  printed reading already carries would need a conditional or nested test — decoding `P` from
  residuals after partialling out the rendered literals. That test is not preregistered, was not
  run, and would itself be exploratory.
- It does not invalidate the H5 measurement. H5's permutation null uses real residuals and
  remains a valid measurement of what it measures. No recorded endpoint value, status, or the
  Level 1 of 6 assessment changes on the basis of this analysis.
- It says nothing about causal use, which no v0 analysis addresses.

## A control that can fail

This baseline's unobservable arm is a negative control with power, and it is a useful contrast
with H6. Its probe is non-degenerate — it still has the concentration and entity features and a
real design matrix — and it fails on the merits, at R² −0.043 with p 0.61, because no
pressure-bearing literal is rendered. H6's arm, by contrast, has an identically zero residual, so
its probe is intercept-only and its statistic could not have moved for any model, layer, or
prompt world (`LIMITATIONS.md`). The difference is exactly the one the amendment in
`PRE_REGISTRATION.md` asks future controls to declare in advance.

## Artifacts

`runs/<run-id>/exploratory/surface_baseline/` — `manifest.json` (schema
`gct-exploratory-surface-baseline-v1`, recording the run's `config_hash`, the predeclared penalty
grid, the parsed label map, and `analysis_role: exploratory_non_confirmatory`),
`results.parquet`, and `permutation_nulls.parquet`. Nothing is written into `probes/` or
`statistics/`, which are confirmatory namespaces.
