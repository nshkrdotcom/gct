# Acceptance criteria audit

This checklist is updated from actual artifacts during final review. “Implemented” is distinct from
“completed scientific run.”

- [x] Installable typed package and deterministic uv lock.
- [x] ToyThermo oracle, all arms, transforms, squares/cycles, grouped splits, and leakage tests.
- [x] Qwen BF16 loader, anchor alignment, explicit all-layer indexing, deterministic behavior parser.
- [x] Hash-verified safetensor/Parquet shards and resume.
- [x] Identity, mean, PCA-affine, low-rank, and continuous-generator operators.
- [x] Transport, cycle, commuting-square, matching/descent, and composition machinery.
- [x] Residual pressure probes, grouped permutation nulls, base lift, Q, unobservable, and renaming.
- [x] Grouped bootstrap, FDR, behavior confounds, MDL sweep, plots, and generated reports.
- [x] CLI contract and one-command resolver.
- [x] Full preregistered real-model run verified.
- [x] Final report populated from the full run rather than development artifacts.
- [x] Final tests, lint, format, mypy, deterministic regeneration, and artifact verification recorded.
- [x] Final TODO/placeholder and test-contamination audit complete.
- [x] Logical commits and verified remote push complete.

## Model #2 replication

- [x] Shared Qwen/Phi adapter boundary with immutable Phi revision and hashed remote-code sources.
- [x] Official Phi template, invariant `FINAL=` suffix audit, embedding + 32-layer indexing, BF16 CUDA,
  deterministic generation, and real-model integration test.
- [x] Exact byte-identical Model #1 dataset reuse with logical hash/stable-ID/split equality checks.
- [x] Distinct scientific/CI configs; CI artifacts cannot emit scientific root reports.
- [x] Full 12,600-row Phi activation and behavior run, raw mismatch preservation, and exact
  post-canonicalization audit.
- [x] Validation-only layer/operator/probe selection frozen before test evaluation.
- [x] All H1–H8 endpoints and controls evaluated; H6 mandatory and passing in primary/renamed worlds.
- [x] Stable-ID/base-world paired cross-model analysis, seven required figures, and schema-validated JSON.
- [x] Actual-evidence `REPORT_MODEL2.md` and `REPORT_CROSS_MODEL.md` generated.
- [x] Final Model #2 tests/lint/format/typecheck/real integration/hash/regeneration gates recorded.
- [x] Model #2 implementation/reports committed, pushed, remote-verified, and worktree clean.
