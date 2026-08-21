# Cross-model analysis

Model #1 (`gct-v0.1-db5a41461117`) and Model #2
(`gct-v0.2-phi4mini-7a87777ac843`) use the same 12,600 stable sample IDs and 420 base-world groups.
The deterministic command is:

```bash
uv run gct compare models \
  --baseline-run gct-v0.1-db5a41461117 \
  --replication-run gct-v0.2-phi4mini-7a87777ac843
```

All dataset equality checks precede analysis. H1–H4 and H7 have persisted additive group effects and
receive paired whole-base-world bootstrap intervals. H5/H6 are aggregate probe R² endpoints without
persisted row-level predictions, and H8 is a joint gate, so these cross-model differences remain
descriptive. Behavior resampling carries each sampled base world's model-specific parsed rows together.
No join uses row order.

The generated summary, paired tables, and seven preregistered figures live under the Model #2 run's
`cross_model/` directory. `REPORT_CROSS_MODEL.md` applies the frozen failure interpretation matrix and
cannot alter either model's endpoint decisions.
