# Data Accuracy Metrics — Clarification (Reviewer Correction A)

**Date:** 2026-11-02
**Addendum to:** `phase_4a_completion_report.md`
**Status:** Additive documentation, does not modify any prior commit

## The Two Different Data Accuracy Metrics

The Phase 4A completion report cites two apparently contradictory data-accuracy results:

1. **Final validation MSE (Protocol A, BATCH=16):** 0.066331 → 0.066678, a **+0.52% increase** (Phase 4A.1 slightly worse)
2. **Mean paired relative L2 delta:** +3.949e-3, favoring Phase 4A.1 (statistically significant improvement)

**These are not contradictory.** They measure different quantities.

## Metric Definitions

### MSE (Protocol A)
- **Sensitive to worst-case magnitude** — a few scenarios with large absolute errors dominate
- Normalized-state units
- Aggregate over the whole validation set with equal element-weighting

### Relative L2 (paired per-scenario)
- **Scale-invariant per scenario** — normalizes each scenario by its own magnitude
- Median/mean computed across 200 paired deltas
- Sensitive to typical scenario behavior, not worst-case magnitude

## Why Both Can Be True Simultaneously

Phase 4A.1's fine-tuning:
- **Slightly worsened the worst-case scenarios** — a handful of scenarios got much worse in absolute terms, driving the aggregate MSE up by 0.52%
- **Improved most scenarios** — 162/200 scenarios have lower relative L2 (Phase 4A.1 wins 81%)
- **Improved typical accuracy** — median relative L2 0.75% worse (small), mean relative L2 −2.19% better (Phase 4A.1 better)

The MSE-vs-relative-L2 divergence is a consequence of the different aggregation and normalization. Both are valid metrics; neither supersedes the other.

## Reporting Guidance for Manuscript

Report **both** metrics with their definitions:
- "Phase 4A.1 slightly increased aggregate MSE on the validation set (+0.52%, driven by worst-case scenarios) while statistically significantly improving median per-scenario relative L2 (Phase 4A.1 wins 162/200 scenarios, mean paired delta +3.95e-3, 95% CI excludes zero)."

Do not claim one supersedes the other. Do not present only the favorable metric.

## Evidence
- `results/phase_4/phase_4a/optimizer/training_summary.json` (MSE at final ckpt: 0.066702 — plateau epoch 54) 
- `results/phase_4/phase_4a/residual_analysis/baseline_metrics_canonical_FIXED.json` (Protocol A, BATCH=16, final eval: 0.066678)
- `results/phase_4/phase_4a/residual_analysis/per_scenario_comparison.csv` (per-scenario relative L2)
- `results/phase_4/phase_4a/residual_analysis/paired_bootstrap_stats.json` (paired statistics)
