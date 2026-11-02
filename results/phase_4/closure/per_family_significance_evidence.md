# Per-Family Significance — Full Evidence (Reviewer Correction B)

**Date:** 2026-11-02
**Addendum to:** `phase_4a_completion_report.md`
**Status:** Additive documentation

## Overview

The original completion report cited family-level improvements without per-family confidence intervals. This document provides the full statistical evidence.

## Bootstrap Procedure

- 10,000 resamples with replacement
- 200 scenarios per resample
- Random seed: 42 (numpy default_rng)
- 95% CI via percentile method (2.5%, 97.5%)
- Source: `paired_bootstrap_stats.json` (from commit `f7474c0`)

## Per-Family Paired Bootstrap Results

### Ramp (n=50)

| Metric | Value |
|--------|-------|
| Mean paired delta | **+1.050e-2** |
| Mean 95% CI | **[+8.27e-3, +1.32e-2]** — **excludes zero, statistically significant** |
| Median paired delta | +8.681e-3 |
| Median 95% CI | [+7.03e-3, +9.99e-3] — excludes zero |
| Phase 4A.1 wins | 49/50 (98%) |
| Phase 3 wins | 1/50 (2%) |

**Verdict:** **Strongly statistically significant improvement.**

### Sinusoidal (n=50)

| Metric | Value |
|--------|-------|
| Mean paired delta | **+6.723e-3** |
| Mean 95% CI | **[+1.63e-3, +1.12e-2]** — **excludes zero, statistically significant** |
| Median paired delta | +8.324e-3 |
| Median 95% CI | [+6.24e-3, +1.09e-2] — excludes zero |
| Phase 4A.1 wins | 42/50 (84%) |
| Phase 3 wins | 8/50 (16%) |

**Verdict:** **Statistically significant improvement.**

### Step (n=50)

| Metric | Value |
|--------|-------|
| Mean paired delta | +3.474e-6 |
| Mean 95% CI | [−4.69e-4, +3.57e-4] — **crosses zero, not statistically significant on mean** |
| Median paired delta | +5.494e-5 |
| Median 95% CI | [+2.92e-5, +7.36e-5] — excludes zero on median |
| Phase 4A.1 wins | 49/50 (98%) |

**Verdict:** **Small but consistent improvement. Mean not significant (CI crosses zero). Median significant. Report as "small consistent descriptive improvement; mean CI crosses zero."**

### GRF (n=50)

| Metric | Value |
|--------|-------|
| Mean paired delta | −1.429e-3 |
| Mean 95% CI | [−5.73e-3, +2.83e-3] — **crosses zero, NOT statistically significant** |
| Median paired delta | −1.253e-3 |
| Median 95% CI | [−7.60e-3, +4.82e-3] — crosses zero |
| Phase 4A.1 wins | 22/50 (44%) |
| Phase 3 wins | 28/50 (56%) |

**Verdict:** **Not statistically significant. Descriptive pattern suggests slight degradation but CIs cross zero. Do NOT claim degradation as statistically established.**

## Manuscript Language Correction

**Original claim (superseded):**
> "Ramp and sinusoidal improvements statistically significant" — implied by 98% / 84% win rates

**Corrected claim (statistically defensible):**
> "Paired bootstrap analysis on the 200 validation scenarios shows statistically significant improvements on the ramp (mean paired delta +1.05e-2, 95% CI [+8.27e-3, +1.32e-2], n=50) and sinusoidal (mean paired delta +6.72e-3, 95% CI [+1.63e-3, +1.12e-2], n=50) families. The step family shows small consistent improvements (98% win rate) but the mean paired delta's 95% CI crosses zero. The GRF family is not statistically distinguishable from neutral (mean delta −1.43e-3, 95% CI [−5.73e-3, +2.83e-3])."

## Evidence
- `results/phase_4/phase_4a/residual_analysis/paired_bootstrap_stats.json`
- `results/phase_4/phase_4a/residual_analysis/paired_bootstrap.log`
- `results/phase_4/phase_4a/residual_analysis/per_scenario_comparison.csv`
- `scripts/phase_4/step_4a_3_r_2_paired_bootstrap.py`
