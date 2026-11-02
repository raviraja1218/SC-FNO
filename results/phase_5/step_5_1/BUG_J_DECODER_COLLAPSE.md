# Bug J — Phase 3 Decoder Temporal Collapse

**Date:** 2026-11-02
**Severity:** CRITICAL — invalidates Phase 4 physics claims
**Status:** Discovered during Step 5.1 pre-training verification

## Observation

The Phase 3 PI-FNO decoder produces nearly constant output across query times.

### Phase 3 baseline (unpadded, direct from checkpoint)
- `u_hat` across 50 query times: **variance = 7.1e-5**
- `du_dt` across 50 query times: **std = 1.4e-3, mean = -3.0e-4**
- Latent `z2`: shape `[B, 32, 200]`, std = 1.41 (latent is NOT constant)
- **Interpretation:** decoder reads latent but produces constant output regardless of query time

### P5-B (padded UnifiedLatentFNO, best checkpoint)
- `u_hat` variance across 50 query times: **6.0e-6** (12× worse than Phase 3)
- `du_dt` std: **1.0e-4** (essentially zero)
- **Interpretation:** padded wrapper slightly worsened an already-collapsed decoder

## Root Cause

The decoder receives gradient from **only two sources** in the training loop:

1. **Physics loss (λ_phys × L_phys)** — but λ_phys started at 1e-4, and du_dt≈0 makes residual ≈ 0, so physics gradient ≈ 0
2. **IC loss (L_ic)** — only at t=0

**No supervised data loss is applied to decoder output.** The data loss `L_data = MSE(model.fno(xb), yb)` uses only the projection MLP path — the decoder is not in that path.

The decoder therefore collapses to:
- At t=0: matching `y0` (IC loss anchors this)
- At t>0: nearly the same value (no gradient to change it, physics residual ≈ 0 for any constant output)

## Consequence for Phase 4

Phase 4A/4B reported substantial "physics loss" values (1.67e-3 for Phase 3, 1.95e-6 for Phase 4A.1, 174–226 for Phase 4B variants).

**All of these are measuring a degenerate quantity.**

With `du_dt ≈ 0`, the 9-equation residual R_j reduces to a function of `u_hat` alone:
- `R_n ≈ 0 - [(ρ(u) - β_eff)/Λ · n + Σ λᵢCᵢ]` — the algebraic term
- The "physics loss" measures algebraic consistency of `u_hat`, NOT dynamic consistency

**What Phase 4A.1 actually improved:** algebraic residual reduction by tuning the FNO latent to produce less-algebraically-inconsistent constant outputs. **Not physics-dynamic consistency.**

**What Phase 4B variants failed at:** they were optimizing against the same degenerate metric, so their 100,000× worse residual is meaningful (worse algebraic consistency), but their 174–226 "physics loss" does not measure dynamic physics.

## Why This Wasn't Caught

- Phase 3's 1.68e-3 physics loss was measured with per-state p95 scales that were themselves computed from the degenerate model
- Small absolute physics loss values looked good
- The per-sum derivative bug (Bug E) further obscured the issue
- No test ever verified `du_dt` magnitude against ground truth derivative

## Required Correction

**Option A — Reopen Phase 3 (controlled correction, requires approval):**
1. Add `L_data_decoder = MSE(u_hat_interpolated, y_at_tq)` to composite loss
2. Retrain Phase 3 PI-FNO with this supervised decoder path
3. Verify `du_dt` std > 0.05 after retraining
4. Retrain Phase 4A.1 and Phase 4B variants
5. Retrain Phase 5.1
6. Document as "Phase 3 structural correction"

**Option B — Accept Phase 4 as degenerate-physics negative result:**
1. Report in Phase 5 that Phase 4's "physics improvement" was measuring algebraic consistency of a collapsed decoder
2. Do NOT pursue Phase 5.1's intervention comparison under false premise
3. Move to Phase 5.3B (causal boundary) with fresh architecture and supervised decoder
4. Phase 5 final report: "The decoder-derivative pathway was architecturally degenerate; all Phase 4 physics results are documented as such."

## Recommendation

**Option B is scientifically honest and preserves Phase 0-4 locks.** Option A requires breaking Phase 3 lock (governance violation) but produces valid physics results.

**This decision must be made by the reviewer before Phase 5.1 continues.**

## Status

- Step 5.1 pre-flight: PASS (all locks intact)
- Step 5.1 training run 1: INVALID (3 API bugs + Bug J)
- Step 5.1 training run 2: BLOCKED pending Bug J decision
- Phase 4 physics claims: UNDER REVIEW
