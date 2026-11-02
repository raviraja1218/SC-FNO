# Statistical Reversal — Full Reproducibility Audit (Reviewer Correction C)

**Date:** 2026-11-02
**Status:** Independent verification of commit `f7474c0`

## Purpose

The Step 4A.3 report (`3a04b3d`) and Step 4A.3.R.2 report (`f7474c0`) gave contradictory win/loss labels. This document records the full reproducible audit trail that establishes `f7474c0` as authoritative.

## 1. Source Data

**File:** `results/phase_4/phase_4a/residual_analysis/per_scenario_comparison.csv`
**SHA-256:** (see `statistical_reversal_reproducibility.json` → `source_sha256`)
**Rows:** 200 (one per validation scenario)
**Columns:** `scenario_id, family, phase3_rel_l2, phase4a1_rel_l2, delta, pct_change`

## 2. Delta Definition


- **Positive delta**: Phase 3 has higher error → Phase 4A.1 better
- **Negative delta**: Phase 3 has lower error → Phase 3 better

## 3. Bootstrap Procedure

| Parameter | Value |
|-----------|-------|
| Resample count | 10,000 |
| Random seed | 42 |
| RNG | `numpy.random.default_rng(seed=42)` |
| Resample scheme | With replacement, size = 200 per resample |
| CI method | Percentile (2.5%, 97.5%) |
| Wins definition | `delta > 0` |
| Losses definition | `delta < 0` |
| Ties definition | `delta == 0` |

## 4. Win/Loss/Tie Counts (Recomputed)

| Group | Wins (4A.1) | Losses (P3) | Ties | Win Rate (4A.1) |
|-------|-------------|-------------|------|-----------------|
| Overall | 162 | 38 | 0 | 81.0% |
| Ramp | 49 | 1 | 0 | 98.0% |
| Sinusoidal | 42 | 8 | 0 | 84.0% |
| Step | 49 | 1 | 0 | 98.0% |
| GRF | 22 | 28 | 0 | 44.0% |

## 5. Confidence Intervals (Recomputed)

| Group | Mean delta | 95% CI | Median delta | 95% CI |
|-------|-----------|--------|--------------|--------|
| Overall | +3.949e-3 | [+2.100e-3, +5.765e-3] | +4.296e-3 | [+4.524e-4, +6.204e-3] |
| Ramp | +1.050e-2 | [+8.273e-3, +1.324e-2] | +8.681e-3 | [+7.027e-3, +9.986e-3] |
| Sinusoidal | +6.723e-3 | [+1.627e-3, +1.123e-2] | +8.324e-3 | [+6.236e-3, +1.086e-2] |
| Step | +3.474e-6 | [−4.691e-4, +3.573e-4] | +5.494e-5 | [+2.918e-5, +7.363e-5] |
| GRF | −1.429e-3 | [−5.726e-3, +2.828e-3] | −1.253e-3 | [−7.597e-3, +4.819e-3] |

## 6. Verification Statement

An independent recomputation of the full bootstrap analysis from `per_scenario_comparison.csv` (using the parameters in §3) reproduces all published values in `paired_bootstrap_stats.json` to numerical precision. **Commit `f7474c0` is authoritative.**

See `statistical_reversal_reproducibility.json` for the full machine-readable record. See `statistical_reversal_reproducibility.log` for the verification execution trace.

## 7. Why the Earlier Report Was Wrong

The original Step 4A.3 report (`3a04b3d`) stated "Phase 3 wins 162/200." This was a **sign-convention error** in the report's interpretation of the same data. The underlying CSV data was always correct. Only the report's verbal label was flipped.

**No data was ever changed.** Only the interpretation was corrected.

## 8. Manuscript Guidance

Cite the corrected interpretation:
> "Phase 4A.1 wins 162 of 200 scenarios (81%) on paired relative L2, with mean paired delta +3.95e-3 (95% CI [+2.10e-3, +5.77e-3], statistically significant)."

Do NOT cite "Phase 3 wins 162/200" — that interpretation is superseded.
