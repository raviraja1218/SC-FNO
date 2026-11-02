# Phase 4A — Completion Report (Final)

**Phase:** 4A — Adaptive Optimization (Adam → L-BFGS)
**Milestone:** `PHASE_4A_LOCKED — OPTIMIZATION SUCCESSFUL`
**Closure date:** 2026-11-01
**Starting checkpoint:** `models/sc_fno/phase_3/pi_fno/best_val.pt` (SHA `a0522e57...`)
**Predecessor:** Phase 3 lock `bc80727` (verified intact)

---

## Section 1 — Executive Summary

Phase 4A investigated whether adaptive Adam → L-BFGS fine-tuning of the locked Phase 3 PI-FNO could reduce physical residuals while preserving or improving data accuracy.

**Result: YES.** The adaptive optimizer achieved:

- **828.4× reduction in 9-equation physics loss** (1.68e-3 → 2.03e-6)
- **10⁻⁵ target achieved** with 4.9× margin
- **40.5× reduction in IC residual**
- **Statistically significant data accuracy improvement** (mean paired delta +3.9488e-03, 95% CI [+2.0999e-03, +5.7646e-03])
- **Phase 4A.1 wins 162/200 scenarios (81%)** on data accuracy

### Summary metrics

| Metric | Phase 3 PI-FNO | Phase 4A.1 | Δ |
|--------|----------------|------------|---|
| L_physics (9-eq) | 1.6807e-03 | **2.0289e-06** | **−99.88% (828.4×)** |
| L_IC | 5.0996e-07 | **1.2592e-08** | **−97.53% (40.5×)** |
| L_data (final ckpt) | 0.066331 | 0.066678 | +0.52% |
| Scenario win rate | — | **81%** (162/200) | — |
| Peak VRAM | — | 78.2 MB | 38× under 3 GB |

**All 5 operational gates (4A.0–4A.4) PASS. Phase 4A gate: PASS.**

---

## Section 2 — Adaptive Optimization Algorithm

**Adam:** lr=1e-3, betas=(0.9, 0.999), CosineAnnealingLR (T_max=100), fresh state
**λ_phys:** ramp 0.002754 → 1.0 over 64 epochs, transition policy B (inherit at plateau)
**Plateau detection:** EMA α=0.1, ref_window=20, δ=1e-4, patience=3, min_epochs=20, max=100
**L-BFGS:** history=50, max_iter=50,000, tolerances (1e-7, 1e-9), strong_wolfe, fixed 16-scenario batch

---

## Section 3 — Training Execution (Reviewer-Corrected Wording)

**Adam trajectory (Correction 1 applied):**
The Adam stage reached its **best validation loss (0.066085) at epoch 28**. Plateau detection triggered **26 epochs later at epoch 54** (val_data = 0.066702). Training thus **continued past the best-validation point** before transitioning to L-BFGS. The final checkpoint corresponds to the **plateau-trigger epoch, not the best-val epoch**. No monotonic convergence is claimed — the val_data trajectory reached its minimum at epoch 28, rose slightly, and remained roughly flat until the EMA-based plateau rule fired at epoch 54.

**L-BFGS stage (Correction 2 applied):**
The L-BFGS stage **terminated after 17 closure evaluations** on the fixed 16-scenario batch because `tolerance_change=1e-9` was satisfied. This establishes **local loss stability on that specific batch only** — it does not establish convergence to a global optimum, convergence over the full training set, or convergence over the 200 validation scenarios.

### Training metrics

| Metric | Value |
|--------|-------|
| Adam epochs | 55 (plateau at 54) |
| Adam wall time | 269.8 s |
| Best val_data (epoch 28) | 0.066085 |
| Plateau val_data (epoch 54) | 0.066702 |
| Transition λ_phys | 0.3981 |
| L-BFGS closures | 17 |
| L-BFGS final loss | 0.599 |
| L-BFGS wall time | 0.887 s |
| Total wall time | 270.7 s |
| Peak VRAM (Adam) | 69.3 MB |
| Peak VRAM (L-BFGS) | 78.2 MB |

---

## Section 4 — Per-State Residuals

| State | Phase 3 | Phase 4A.1 | Ratio | Δ% |
|-------|---------|------------|-------|-----|
| n | 4.0223e-04 | 1.8688e-06 | 215.2× | -99.54% |
| C1 | 6.1002e-03 | 1.0756e-06 | 5671.5× | -99.98% |
| C2 | 7.5691e-04 | 2.0495e-06 | 369.3× | -99.73% |
| C3 | 7.5053e-04 | 1.0570e-06 | 710.0× | -99.86% |
| C4 | 7.7924e-04 | 6.9819e-07 | 1116.1× | -99.91% |
| C5 | 7.2083e-04 | 6.2260e-07 | 1157.8× | -99.91% |
| C6 | 3.7424e-03 | 1.1351e-06 | 3296.9× | -99.97% |
| Tf | 4.2847e-04 | 1.5836e-06 | 270.6× | -99.63% |
| Tc | 1.4455e-03 | 8.1695e-06 | 176.9× | -99.43% |

**All 9 states improved by 176.9×–5671.5×.**

### Aggregate physics loss
| Model | L_physics | Target ratio |
|-------|-----------|--------------|
| Phase 3 PI-FNO | 1.6807e-03 | 168.07× above 10⁻⁵ |
| **Phase 4A.1** | **2.0289e-06** | **4.9× below 10⁻⁵** |

### IC residual
| Model | L_IC | Improvement |
|-------|------|-------------|
| Phase 3 PI-FNO | 5.0996e-07 | — |
| **Phase 4A.1** | **1.2592e-08** | **40.5×** |

### Per-family accuracy (median rel L2)
| Family | Phase 3 | Phase 4A.1 | Δ% | Significant? |
|--------|---------|------------|-----|--------------|
| ramp | 2.198e-02 | 1.211e-02 | -44.90% | ✅ yes |
| sinusoidal | 1.158e-01 | 1.039e-01 | -10.28% | ✅ yes |
| step | 3.113e-01 | 3.113e-01 | -0.01% | ⚪ not significant |
| grf | 2.296e-01 | 2.456e-01 | +6.96% | ⚪ not significant |

---

## Section 5 — Statistical Reversal (Correction 3 Applied)

**Authoritative interpretation (commit `f7474c0`):**
- **Phase 4A.1 wins 162/200 scenarios (81%)**
- Phase 3 wins 38/200 scenarios (19%)
- Mean paired delta +3.9488e-03, 95% CI [+2.0999e-03, +5.7646e-03] — excludes zero
- Median paired delta +4.2955e-03, 95% CI [+4.5245e-04, +6.2038e-03]

**Supersedes:** Commit `3a04b3d` had the win/loss labels flipped. Data unchanged; only interpretation was corrected. See `statistical_reversal_reconciliation.md`.

---

## Section 6 — Optimization Visualization

**6 figures produced (Step 4A.4, commit `fa28591`):** F1 Adam loss, F2 L-BFGS loss, F3 combined trace, F4 transition marker, F6 data-vs-physics trade-off, F8 GPU memory.

**2 figures honestly marked unavailable:** F5 (per-equation residual convergence — not logged per epoch), F7 (gradient norm evolution — not logged).

**No fabricated curves.**

---

## Section 7 — Gate Checklist

| Step | Gate | Status |
|------|------|--------|
| 4A.0 | Baseline + preflight (14/14) | ✅ PASS |
| 4A.1 | Implement optimizer (16/16) | ✅ PASS |
| 4A.2 | Execute fine-tuning (18/18) | ✅ PASS |
| 4A.3 | Residual convergence (9/9) | ✅ PASS |
| 4A.4 | Optimization analysis (23/23) | ✅ PASS |
| 4A.5 | Phase 4A closure | ✅ PASS |

---

## Section 8 — Final Declaration

**`PHASE_4A_LOCKED — OPTIMIZATION SUCCESSFUL`**

### Supported
- 828.4× physics reduction, 10⁻⁵ target achieved (4.9× margin)
- 40.5× IC improvement
- All 9 per-state residuals improved 176.9×–5671.5×
- Data accuracy statistically improved (Phase 4A.1 wins 81%)
- No numerical instability, VRAM 38× under gate, wall time 4.5 min

### Not claimed
- Individual causal contribution of duration/λ/optimizer sequence (combined only)
- Boundary artifact resolution (persists)
- L-BFGS global convergence (terminated on tolerance, fixed batch)
- Monotonic val_data convergence (best epoch 28, plateau epoch 54)

### Reviewer-approved causal wording
> *"The tested adaptive optimization procedure achieved a substantial reduction in the normalized nine-equation physics residual. The result demonstrates that the Phase 3 level of physics convergence was not an inherent limitation of the architecture. However, the individual contributions of training duration, physics-weight scheduling, and optimizer sequencing remain to be isolated."*

### Handoff to Phase 4B
- Start from **Phase 3 PI-FNO** (`a0522e57`), NOT Phase 4A.1
- Use Phase 3 Adam protocol
- Priority: recover data accuracy while maintaining physics < 1e-5
- No R_n-specific interventions

---

**Milestone:** `PHASE_4A_LOCKED — OPTIMIZATION SUCCESSFUL`
**Phase 1/2/3 locks:** Intact (see `lock_integrity_verification.json`)

---

# Post-Review Addendum (Step 4A.5.R)

**Date:** 2026-11-02
**Trigger:** Reviewer corrections A–E from Step 4A.5 review

All five reviewer corrections are accepted. Corrections are **additive documentation** — the tag `phase-4a-locked` remains at `dea218a`. No training, no checkpoints, no test-set use.

## Correction A — Data-Accuracy Metrics Clarification

The 0.52% MSE increase and the statistically significant relative L2 improvement are **different metrics**, not contradictory. See `data_accuracy_metrics_clarification.md` for full detail.

- **MSE (Protocol A):** sensitive to worst-case magnitude. Rose 0.52% because a few scenarios worsened substantially in absolute terms.
- **Relative L2 (paired per-scenario):** scale-invariant, dominated by typical scenario behavior. Improved (162/200 wins, mean paired delta +3.95e-3, 95% CI excludes zero).

Both are reported. Neither supersedes the other.

## Correction B — Per-Family Significance with CIs

The completion report cited family win rates but did not include confidence intervals. Full bootstrap CIs are now documented in `per_family_significance_evidence.md`:

| Family | Mean Δ | 95% CI | Median Δ | 95% CI | Verdict |
|--------|--------|--------|----------|--------|---------|
| ramp | +1.050e-2 | [+8.27e-3, +1.32e-2] | +8.68e-3 | [+7.03e-3, +9.99e-3] | ✅ Significant |
| sinusoidal | +6.72e-3 | [+1.63e-3, +1.12e-2] | +8.32e-3 | [+6.24e-3, +1.09e-2] | ✅ Significant |
| step | +3.47e-6 | [−4.69e-4, +3.57e-4] | +5.49e-5 | [+2.92e-5, +7.36e-5] | ⚪ Mean CI crosses zero |
| grf | −1.43e-3 | [−5.73e-3, +2.83e-3] | −1.25e-3 | [−7.60e-3, +4.82e-3] | ⚪ Not significant |

**Manuscript guidance:** claim ramp and sinusoidal improvements as statistically significant. Do NOT claim step improvement as significant on mean. Do NOT claim GRF degradation as significant. The GRF family's direction is descriptively negative but the CIs cross zero.

## Correction C — Statistical Reversal Reproducibility

Full audit trail in `statistical_reversal_reproducibility.md` and `statistical_reversal_reproducibility.json`. Independent recomputation from `per_scenario_comparison.csv` using the same procedure (10,000 resamples, seed=42, percentile CI) reproduces all published values. Commit `f7474c0` is authoritative.

Source data SHA-256 recorded. Full bootstrap parameters documented. Win/loss/tie counts per group verified. Both mean and median CIs reproduced.

## Correction D — Causal Scope

The reviewer-approved causal wording is preserved verbatim in Section 8:

> *"The tested adaptive optimization procedure achieved a substantial reduction in the normalized nine-equation physics residual. The result demonstrates that the Phase 3 level of physics convergence was not an inherent limitation of the architecture. However, the individual contributions of training duration, physics-weight scheduling, and optimizer sequencing remain to be isolated."*

See `causal_scope_note.md`. Do not attribute the 828× improvement to any single component in the manuscript.

## Correction E — Commit Count

The completion report stated "19 commits" but the lineage includes 18 Phase 4A commits on top of the Phase 3 lock `bc80727`. Corrected in `commit_count_correction.md`:

**18 Phase 4A commits on top of the Phase 3 lock `bc80727`.**

## Corrected Figures

The six figures produced in Step 4A.4 (commit `fa28591`) remain valid without modification. Both F5 and F7 remain honestly documented as unavailable (see the `.md` files in the 4A.4 evidence). No new figures need to be generated.

## Status After Corrections

| Aspect | Before Correction | After Correction |
|--------|-------------------|------------------|
| Data accuracy (MSE) | "+0.52% cost" | +0.52% MSE increase (worst-case sensitive) |
| Data accuracy (rel L2) | "+3.95e-3 improvement" | +3.95e-3 (95% CI excludes zero) — different metric |
| Ramp significance | "45% better" | +1.05e-2 (95% CI [+8.27e-3, +1.32e-2], statistically significant) |
| Sinusoidal significance | "10% better" | +6.72e-3 (95% CI [+1.63e-3, +1.12e-2], statistically significant) |
| Step significance | "neutral (98% win)" | Descriptive only — mean CI crosses zero |
| GRF significance | "not significant" | Confirmed not significant (CIs cross zero) |
| Statistical reversal | "Phase 4A.1 wins 162/200" | Fully reproducible — see audit document |
| Causal scope | "combined procedure only" | Verbatim reviewer-approved wording preserved |
| Commit count | "19 Phase 4A commits" | 18 Phase 4A commits + Phase 3 lock as predecessor |

**`phase-4a-locked` tag at `dea218a` remains unchanged.** All corrections are additive documentation committed after the tag.
