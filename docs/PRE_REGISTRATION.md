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
