# Phase 4A.2.R — Response to Reviewer

**Date:** 2026-11-01
**Predecessor commit:** `bf1e6d5` (original Phase 4A.2 measurement)
**Status:** All 6 reviewer corrections addressed

## Summary

The reviewer's six corrections are all accepted. The most important — C1 — was
resolved by discovering a **diagnostic bug in the original per-state CSV**, not
a bug in the aggregate. The aggregate `compute_physics_loss = 2.03e-6` was
always correct; the per-state table was wrong.

**Corrected finding:** Phase 4A.1 achieved a genuine **828× reduction in
9-equation physics loss** (1.68e-3 → 2.03e-6), with **all 9 per-state residuals
improving 177×–5,672×**. The 10⁻⁵ target is achieved with 4.9× margin.

The R_n "bottleneck" claimed in the original report was a **diagnostic artifact**,
not a real physics or architecture limitation. It is retracted.

---

## C1 — Aggregate Reconciliation (RESOLVED)

**Reviewer's math:** 9-equation mean cannot be 2.03e-6 if R_n = 1897.

**Root cause:** The original 4A.2 diagnostic computed per-state residuals via a
**manual denormalization** of `du_hat_dt` (`du_phys = du_hat_dt × norm_std`).
But `compute_physics_loss` passes `du_hat_dt` **directly** (already in physical
units per `compute_du_dt`). The manual denorm was wrong, producing a spurious
R_n ≈ 1897.

**Verification:** Recomputed per-state with the exact `compute_physics_loss` path:
**Both models reconcile to machine precision.** R_n's true Phase 4A.1 value is
**1.87e-6**, not 1897.

## C2 — R_n Independent Audit (RESOLVED — NO BOTTLENECK)

**Reviewer:** R_n structural explanation needs verification.

**Outcome:** No structural issue. R_n's correct value:
- Phase 3: 4.02e-4 (normalized)
- Phase 4A.1: 1.87e-6
- **Improvement: 215×** (not unchanged at 1897)

R_n improved normally, like every other state. The "bottleneck" was a bug in my
diagnostic script.

## C3 — Timescale Explanation (WITHDRAWN)

The Λ vs τ timescale analysis in the original report was addressing a phantom
value (R_n = 1897). Since the correct R_n = 1.87e-6 is tiny, no timescale
argument is needed. **The entire "prompt kinetics vs query resolution" analysis
is withdrawn as explanatory text.**

## C4 — Numerical Corrections

### C4.A — C1 ratio
Correct ratio is **5,672×** (not 1,587× as reviewer calculated from my buggy CSV,
and not 1,030× as I originally claimed).

The correct per-state ratios (from corrected computation) are:

| State | Ratio |
|-------|-------|
| n | 215× |
| C1 | **5,672×** |
| C2 | 369× |
| C3 | 710× |
| C4 | 1,116× |
| C5 | 1,158× |
| C6 | 3,297× |
| Tf | 271× |
| Tc | **177×** (minimum) |

**Correct improvement range: 177× – 5,672×.**

### C4.B — Data loss reconciliation
- 0.066702 = Phase 4A.1 `training_summary.json` (plateau epoch 54)
- 0.066678 = fresh eval of `phase_4a_final.pt` (post-L-BFGS)
- Difference is real: L-BFGS slightly improved val_data_loss
- **Comparison uses final ckpt: +0.52% vs Phase 3**

### C4.C — Boundary behavior
Corrected statement: **5 of 9 states slightly worsened**:
- T_c +0.45, C_5 +0.34, T_f +0.30, C_6 +0.10, n +0.07
- C_1–C_4 unchanged (±0.01)

Original report's "unchanged" claim was too generous. The boundary artifact
persists; adaptive fine-tuning did not address it.

### C4.D — Improvement range
Updated to **177×–5,672×** per C4.A.

## C5 — Scientific Interpretation (REVISED)

**Original (incorrect):** "Partial physics success with R_n bottleneck."

**Corrected:** "Phase 4A.1 achieves a genuine **828× reduction in 9-equation
physics loss**, satisfying the 10⁻⁵ target at 4.9× margin, at 0.52% data cost.
**Every per-state residual improved 177×–5,672×.** No structural bottleneck."

## C6 — Phase 4B Recommendations (UPDATED)

**Remove:** "R_n-targeted interventions" — not needed, R_n is not a bottleneck.

**Retain as hypotheses:**
- PCGrad for gradient conflict
- Adaptive λ_phys for convergence dynamics
- Boundary-aware kernels for FFT wraparound

**No Phase 4B change should assume R_n is architecturally limited** — it is not.

## Files Committed (new)

- `residual_comparison_per_state_CORRECTED.csv`
- `residual_comparison_CORRECTED.json`
- `residual_comparison_CORRECTED.md`
- `residual_comparison_per_state_CORRECTED.png`
- `physics_loss_comparison_CORRECTED.png`
- `boundary_error_comparison_CORRECTED.png`
- `step_4a_2_r_manifest.json`
- `phase_4a_2_reviewer_response.md` (this document)
- `corrected_per_state.log`
- `finalize_corrections.log`
- `scripts/phase_4/step_4a_2_r1_correct_per_state.py`
- `scripts/phase_4/step_4a_2_r2_finalize_corrections.py`

## What Was Not Changed

- Original commit `bf1e6d5` — preserved as measurement record
- Phase 3 lock `bc80727` — untouched
- Phase 4A.1 lock — untouched
- Both model checkpoints — untouched (SHAs verified)
- No retraining
- No test-set use

## Final Correction of Phase 4A.2 Claims

| Original Claim | Corrected |
|----------------|-----------|
| 828× reduction, 8-of-9 states improved | **828× reduction, all 9 states improved** |
| R_n bottleneck at 1897 | **R_n improved 215× to 1.87e-6** |
| Improvement range 181×–1,030× | **177×–5,672×** |
| Boundary unchanged | **5 states slightly worsened** |
| 10⁻⁵ target achieved (provisional) | **10⁻⁵ target genuinely achieved (9-equation)** |

**Phase 4A.1 is a stronger positive result than originally claimed.**
