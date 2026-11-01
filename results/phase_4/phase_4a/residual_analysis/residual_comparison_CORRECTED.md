# Step 4A.2 — Residual Convergence Assessment (CORRECTED)

**Phase 4A.1 v2 vs Phase 3 PI-FNO — 200 validation scenarios**

- Phase 3 PI-FNO SHA: `models/sc_fno/phase_3/pi_fno/best_val.pt`
- Phase 4A.1 final SHA: `models/sc_fno/phase_4/phase_4a/phase_4a_final.pt`

## Correction Note

The original Phase 4A.2 per-state diagnostic (`bf1e6d5`) contained a bug: it
manually denormalized `du_hat_dt` by `norm_std`, while `compute_physics_loss`
passes `du_hat_dt` directly (already in physical units from `compute_du_dt`).
This produced an anomalous R_n = 1897 in the original CSV. **This correction
recomputes per-state residuals using the exact same path as `compute_physics_loss`**, 
verifying aggregate = mean(9 per-state contributions) to machine precision.

## Aggregate Metrics

| Metric | Phase 3 | Phase 4A.1 | Δ | Ratio |
|--------|---------|------------|---|-------|
| L_physics (9-eq) | 1.6807e-03 | 2.0289e-06 | −99.88% | **828.4×** |
| L_data (final ckpt) | 0.066331 | 0.066678 | +0.52% | 1.005× |

## Per-State Residuals (CORRECTED)

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

**All 9 per-state residuals improved by 177×–5,672×.** Aggregate: 828.4×.

## Physics Target (10⁻⁵)

- Phase 3: **NOT ACHIEVED** (168.07× target)
- Phase 4A.1: **ACHIEVED** (4.9× margin)

## Data Loss Reconciliation

- Phase 4A.1 `training_summary.json` recorded: 0.066702 (at plateau epoch 54)
- Phase 4A.2 fresh evaluation of `phase_4a_final.pt`: 0.066678 (post-L-BFGS)
- Difference: 3.6e-5 — L-BFGS slightly improved val_data_loss
- Both vs Phase 3 baseline (0.066331): +0.56% (plateau) / +0.52% (final)
- **Comparison uses final checkpoint: +0.52%**

## Boundary Behavior (REVISED)

| State | Phase 3 | Phase 4A.1 | Δ | Verdict |
|-------|---------|------------|---|---------|
| n | 5.420 | 5.487 | +0.067 | Slight |
| C1 | 2.094 | 2.090 | -0.004 | Unchanged |
| C2 | 2.155 | 2.150 | -0.005 | Unchanged |
| C3 | 2.444 | 2.438 | -0.007 | Unchanged |
| C4 | 3.169 | 3.178 | +0.008 | Unchanged |
| C5 | 4.165 | 4.506 | +0.340 | Worsened |
| C6 | 4.117 | 4.216 | +0.099 | Slight |
| Tf | 4.337 | 4.635 | +0.298 | Worsened |
| Tc | 5.692 | 6.145 | +0.453 | Worsened |

**Boundary artifact persists. 5 of 9 states slightly worsened** (T_c, C_5, T_f, C_6, n).
Adaptive fine-tuning did not address the FFT wraparound issue.

## Scientific Summary

**Phase 4A.1 achieves a genuine 828× reduction in 9-equation physics loss**
(1.68e-3 → 2.03e-6), satisfying the 10⁻⁵ target with 4.9× margin, at a data
accuracy cost of 0.52%. **Every one of the 9 PKE+TH residuals improved by**
**177×–5,672×.** The R_n bottleneck claimed in the original report was a
diagnostic bug, not a physical or architectural limitation.
