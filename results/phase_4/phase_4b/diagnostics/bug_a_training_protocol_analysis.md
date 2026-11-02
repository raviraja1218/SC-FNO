# Bug A — Phase 4B Training Protocol Analysis

**Date:** 2026-11-02
**Severity:** HIGH — Invalidates all 4 Phase 4B models

## Observation

All 4 Phase 4B variants (P4-A, P4-B, P4-C, P4-D) exhibit **physics loss 173–224** on validation data, compared to:
- P3-PI-FNO: 1.68e-3
- Phase 4A.1: 2.03e-6

The Phase 4B models are ~100,000× worse in physics loss than Phase 3 baseline.

## Root Cause

`src/training/train_phase_4b.py`:
- Line 50: `PATIENCE = 30`
- Line 61: `phase3_default_ramp(epoch, init=1e-4, target=1.0, ramp_epochs=100)`

**Training trajectory analysis (P4-A):**
- Epoch 20 (best val): λ_phys = 1e-4 × 10^(20/100) ≈ 4e-4
- Epoch 51 (early stop, patience=30): λ_phys = 1e-4 × 10^(51/100) ≈ 1.1e-2
- Target 1.0 never reached before early stop

**λ_phys at early-stop time is only 1.1% of its target.** The physics penalty contributes ~1% to total loss at stop. Physics residual pressure is 100× too weak.

## Consequence

The models did not learn PKE+TH residuals — they trained primarily on data loss. Physics loss is 100,000× worse than P3. Phase 4B "interventions" (PCGrad, adaptive λ, unified padding) were applied on top of a training protocol that never gave physics loss sufficient weight.

**The interventions are scientifically untested by this experiment.**

## Options

### Option R1 — Retrain with Corrected Protocol (Recommended, Requires Approval)

Change `src/training/train_phase_4b.py`:
- Line 50: `PATIENCE = 30` → `PATIENCE = 100`
- Line 61: `init=1e-4` → `init=0.002754`

The 0.002754 is Phase 4A's observed λ_phys at plateau epoch 54 — this is the empirical starting point that succeeded in Phase 4A.

**Cost:** ~50-90 min GPU retraining
**Requires:** Reviewer approval (protocol deviation from frozen pre-registration)

### Option R2 — Document as Negative Result

Keep invalid models, report Phase 4B as:
> "Training protocol flaw, interventions untested"

**Cost:** No retraining
**Benefit:** Immediate honest closure
**Cost:** Phase 4B interventions never evaluated

## Recommendation

Document Bug A (this file). Request reviewer approval for Option R1. If approved, retrain with corrected protocol, then run Step 4B.5 on valid models.

If not approved, close Phase 4B with Option R2 as negative result.

**In either case, Phase 4A.1 remains the scientifically valid physics champion (828× reduction, 10⁻⁵ target achieved).**
