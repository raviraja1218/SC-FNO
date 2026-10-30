# Metric Aggregation Definitions (Step 2.4 evidence)

## Units
- **MAE, RMSE, max abs error:** computed in **physical units** (denormalized predictions)
- **Relative L2:** dimensionless `||y - ŷ||₂ / ||y||₂` in physical units
- **Training/validation MSE (Step 2.3):** normalized units — do NOT compare directly to Step 2.4 metrics

## Per-scenario, per-state metrics (1800 rows in `scenario_level_metrics.csv`)
For each scenario s (1..200) and state j (1..9), over 200 timesteps i:
- MAE_{s,j} = mean_i | y_{s,j,i} − ŷ_{s,j,i} |
- RMSE_{s,j} = sqrt( mean_i (y_{s,j,i} − ŷ_{s,j,i})² )
- RelL2_{s,j} = ||y_{s,j,·} − ŷ_{s,j,·}||₂ / ||y_{s,j,·}||₂ (undefined if ||y||₂ < 1e-8; count recorded = 0)
- MaxAbs_{s,j} = max_i | y_{s,j,i} − ŷ_{s,j,i} |

## Per-state aggregates (`metrics_per_state.csv`, 9 rows)
For each state j, across 200 scenarios:
- MAE mean / median = mean / median of MAE_{s,j} over s
- RMSE mean / median = mean / median of RMSE_{s,j} over s
- RelL2 mean / median = mean / median of RelL2_{s,j} over s (undefined excluded; count=0)
- MaxAbs mean / max = mean / max of MaxAbs_{s,j} over s

## Per-family aggregates (`metrics_per_family.csv`, 4 rows)
Per family F ∈ {step, ramp, sinusoidal, grf}:
- MAE mean = mean over (s,j) where s ∈ F
- RMSE mean = mean over (s,j) where s ∈ F
- RelL2 median = median over (s,j) where s ∈ F

## Family × state (`metrics_per_family_per_state.csv`, 36 rows)
Per (F, j): MAE mean, RMSE mean, RelL2 median across 50 scenarios in family F for state j.

## Overall aggregates (`aggregate_metrics.json`)
Over 200 × 9 = **1800 (scenario, state) pairs**:
- overall_mae_mean = mean of MAE_{s,j}
- overall_rmse_mean = mean of RMSE_{s,j}
- overall_rel_l2_median = median of RelL2_{s,j} (all defined here)

## Worst-case
- **Aggregate worst:** s* = argmax_s ( mean_j RelL2_{s,j} )
- **Per-state worst j:** s*_j = argmax_s RelL2_{s,j}

## Why physical units (not normalized)
Normalized units compress different physical states (Tf in K, C2 in concentration units) to mean-0/std-1, hiding engineer-meaningful error magnitudes. Training uses normalized MSE; **evaluation uses physical units**.

## Non-claims
Measured errors on the FNO baseline. **Not** claims of physical fidelity, safety qualification, or out-of-distribution generalization. Phase 3 SC-FNO is expected to improve specific states (especially neutron density, relative L2 7.55e-3).
