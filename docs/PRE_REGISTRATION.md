# Preregistration

The machine-readable authority is the `preregistration` mapping in
`configs/experiment_full.yaml`. It is hashed into the run ID before final test evaluation.

The eight endpoints are:

1. nuisance versus substantive held-out displacement;
2. low-rank transport improvement over validation-selected identity/mean shift;
3. continuous-generator held-out prediction and commuting-route consistency;
4. behavior prediction beyond trivial prompt/norm confounds;
5. inferable-arm pressure recovery from held-out residuals;
6. failure of pressure recovery in byte-identical unobservable pairs;
7. structural and behavioral gain from explicit P beyond irrelevant Q;
8. repetition of H2, H5, and H7 under entity-and-field semantic renaming, gated by the renamed
   identical-prompt unobservable control.

The full config fixes grouped counts, transformation magnitudes, renderer and entity holdouts, all
candidate dimensions/ranks, the primary distance, tolerance, bootstrap/permutation counts, alpha, and
MDL lambda sweep. Test data do not select preprocessing or models. Every endpoint is reported even
when negative. The CI config is explicitly development-only and cannot support scientific claims.
For H7, “beyond irrelevant Q” is operationalized conservatively: the lower 95% confidence bound for
the explicit-coordinate gain must exceed both zero and the upper bound for the Q gain, separately for
structural and behavioral endpoints.

---

## Amendment 1 — negative controls must declare their power (2026-08-21)

**This is an amendment, not a revision.** The machine-readable authority is still the
`preregistration` mapping in `configs/experiment_full.yaml`, which is hashed into the run ID and is
unchanged. Nothing below alters a recorded endpoint value, a status, a decision rule, or the
Level 1 of 6 assessment for either completed run. The eight endpoints above are reproduced exactly
as they were frozen; this section records what was learned about one of them after the fact.

### The finding

H6, the identical-prompt unobservable control, cannot fail. In that arm the prompt contains neither
the hidden pressure `P` nor any field depending on it, so the pressure-shift source and target
prompts are byte-identical — enforced by a raise in `src/gct/data/generate.py`. Deterministic
activations are therefore identical, the transport residual is the zero matrix, the fitted probe
collapses to intercept-only, and the endpoint becomes a function of the shared pressure labels
alone. The preregistered rule "passes when recovery does not exceed the 95th percentile of its
permutation null" reduces to `0 <= 0`.

The two completed families confirm it bit-for-bit. Qwen3-4B (hidden 2560, layer 22) and
Phi-4-mini (hidden 3072, layer 13) both record `test_r2` −0.03576857159757152, `null_95th`
−0.03576843143210984, `p` 0.8471528471528471, and `test_mae` 0.5390305964897076, in the primary and
the renamed world alike, while the inferable and irrelevant arms are non-degenerate and
model-specific. H6 is a pipeline sanity check that the probe cannot manufacture signal from a zero
residual. It has no power against model-side leakage and must not be cited as a control that H5
survived.

### The generalizable rule

**Every negative control must declare its power, not only its failure action.** The H6 section
above specifies what to do if the control fires — treat the experiment as contaminated until
leakage is ruled out — but never states what would have to be true for it to fire. A control whose
honest answer to that question is "nothing could make this fire" must be rejected at design time or
reclassified as a pipeline sanity check before any result is reported against it.

### Clause for future preregistrations

For each negative control, state:

1. **(a) The failing quantity.** The exact quantity that would have to change for the control to
   fail, named concretely enough to be measured.
2. **(b) A positive-control demonstration.** Evidence that the control *can* fail when that
   quantity is perturbed — a deliberate perturbation that turns it red, run and recorded before the
   confirmatory analysis.
3. **(c) Invariance disclosure.** Whether the control's statistic is invariant to model, layer, or
   prompt world. A statistic invariant to all three is a property of the design, not a measurement
   of the run, and must be labeled as such wherever it is reported.

A control that cannot satisfy (a) and (b) is not a negative control. It may still be worth running
as a pipeline check, under that name, with its invariance stated.

### What was done about H6

The frozen endpoint and its recorded values are left exactly as they are. The interpretation
surfaces were corrected (`LIMITATIONS.md`, `FAILURE_INTERPRETATION_MATRIX.md`, `METHODS.md`,
`THEORY.md`, `PRIOR_ART_DIFF.md`, `CROSS_MODEL_ANALYSIS.md`, `DECISION_LOG.md`,
`ACCEPTANCE_CRITERIA.md`, `README.md`, and the report generators), and the dead
`verify.py` gate on H6's status was replaced by an assertion with content: that the stored
identical-prompt probes are exactly degenerate — all-zero coefficient over all-zero residual
variance. That is clause (a) for this control, and a deliberate perturbation of the stored
coefficient was confirmed to turn it red, which is clause (b). Clause (c) is now stated wherever
H6 is reported.

The same amendment is recorded against the originating handoff bundle as
`AMENDMENT_0001_NEGATIVE_CONTROL_POWER.md`.
