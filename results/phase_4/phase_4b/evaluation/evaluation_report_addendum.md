# Step 4B.5 — Correction Addendum (Post-Review)

**Date:** 2026-11-02
**Predecessor:** Commit `d226447` (Step 4B.5)
**Status:** ADDITIVE — original Step 4B.5 artifacts preserved

## Correction 1 — Physics Measurement Revalidation

The `compute_du_dt` function used in Phase 3, 4A, and 4B has two defects:

1. **Per-sum derivative**: `grad_outputs=torch.ones_like(u_hat)` computes `∂(Σ_j u_j)/∂t`, then expands to all 9 channels. Correct is per-channel `∂u_j/∂t`.
2. **Zero forcing**: `rho_fn = lambda tq: zeros_like(tq)` sets external reactivity to zero at query times.

The Step 4B.5 physics losses (P3: 1.67e-3, Phase4A.1: 1.95e-6, P4-A/B/C/D: 174–226) are measurements of the per-sum / zero-forcing protocol, not the intended 9-equation residuals.

Corrected physics losses (per-channel + actual forcing) are in `physics_revalidation_addendum/physics_revalidation_results.json`. **Relative ordering is expected to be preserved** (P3 and Phase4A.1 remain physics-converged, P4-A/B/C/D remain physics-diverged), but absolute values and per-state signatures may shift.

**Attribution to Bug A:** the 100,000× physics loss gap between P4 variants and P3 baseline is measured under the same defective protocol used to establish P3's baseline value. The **ordering is preserved**; the **exact magnitude and root-cause decomposition are subject to correction**.

## Correction 2 — Test-Set Usage History

The 200-scenario test set was used in:
- Phase 2 Step 2.4 (P2-FNO baseline)
- Phase 3 Step 3.5 (P2, P3, 4A.1, P4 variants)
- Phase 4B Steps 4B.0–4B.4 (variant diagnostics)
- **Step 4B.5** — definitive seven-model simultaneous evaluation

"First and only use" was incorrect and is retracted.

## Correction 3 — Statistical Clarification

- **Statistic:** median paired difference `median_i(P2_relL2_i − variant_relL2_i)`.
- **Bootstrap:** 10,000 resamples with replacement, seed 42, percentile method.
- **Sign:** positive = variant better than P2.
- **Significance:** Phase 4A.1 CI [+3.68e-7, +1.02e-5] excludes zero at α=0.05.
- **Multiple comparisons:** 5 variant-vs-P2 comparisons. No Bonferroni applied. At Bonferroni α=0.01, Phase 4A.1 remains significant.
- **Scope:** establishes median superiority, not per-scenario superiority (121 wins / 79 losses).

## Correction 4 — Complete Phase 4B Variant Data

Full P4-B/C/D per-family results and bootstrap values are in:
- `per_family_metrics.csv` (full)
- `paired_bootstrap_vs_p2.json` (full)

The Step 4B.5 report tables summarized only a subset; full data is in these files.

## Correction 5 — M4 Comparison Clarification

- Val improvements (Step 4B.4): P4-A −1.35% vs P3, P4-C −1.13% vs P3.
- Test results (Step 4B.5): P4-A 23.5% win rate vs P2 (76.5% loss rate).
- **Different baselines** (P3 for training val, P2 for test). "Val improvement did not transfer to test" holds for both.
