# Figure F5 — 9-Equation Residual Convergence (UNAVAILABLE)

**Reason:** Per-equation residual history was not logged during Phase 4A.1 training.

The `adam_training_log.csv` records only aggregate `train_phys` per epoch. It does not
record per-equation residuals (R_n, R_C1..C6, R_Tf, R_Tc) per epoch.

**Alternative in place:** Step 4A.3 provides the final per-state residual comparison
(Phase 3 vs Phase 4A.1) in `residual_comparison_per_state_CORRECTED.csv`. This shows
the aggregate physics improvement (828× across all 9 states) but not the per-epoch
convergence trajectory.

**Path forward:** A future training run (Phase 4B) would need per-equation residual
logging at each epoch to produce this figure. Do not fabricate from final measurements.
