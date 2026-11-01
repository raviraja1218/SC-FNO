# Step 4A.4 — Optimization Analysis Report

**Phase:** 4A.4 — Optimization Analysis
**Predecessor:** Step 4A.3 LOCKED at `f7474c0`
**Analysis type:** Read-only from Phase 4A.1 training logs
**Training performed in this step:** None

## Summary

The Phase 4A.1 adaptive optimizer ran for **55 Adam epochs** (wall time 269.8 s) followed by **17 L-BFGS closure calls** (wall time 0.887 s). Total training wall time: **270.7 s**.

**Training trajectory highlights:**
- Best validation loss: `0.066085` at epoch 28 (not preserved as checkpoint)
- Plateau detected at epoch 54, λ_phys = 0.3981
- Transition policy: `B_inherit_lambda` — L-BFGS inherited the plateau λ
- Peak GPU memory: Adam 69.3 MB, L-BFGS 78.2 MB
- 3 GB gate satisfied with ~38× headroom

## Figures Produced (6 of 8 planned)

| Figure | Content | Source |
|--------|---------|--------|
| F1 | Adam loss trajectory (train + val) | `adam_training_log.csv` |
| F2 | L-BFGS loss trajectory | `lbfgs_training_log.csv` |
| F3 | Combined optimization trace | Both CSVs |
| F4 | Automatic transition marker with annotations | `transition_event.json` |
| F6 | Data vs physics trade-off (color = λ_phys) | `adam_training_log.csv` |
| F8 | GPU memory evolution | `adam_training_log.csv` |

## Figures Unavailable (2 of 8 planned)

| Figure | Reason | Alternative |
|--------|--------|-------------|
| F5 | Per-equation residual history not logged per epoch | See Step 4A.3 `residual_comparison_per_state_CORRECTED.csv` for final per-state comparison |
| F7 | Gradient norms not logged during training | Not available; add to Phase 4B training script |

**Per reviewer guidance:** These figures are not fabricated. The unavailability
documents what the recorded evidence does and does not support.

## Supported Conclusions (from training evidence)

**Supported:**
- The optimizer completed a two-stage Adam→L-BFGS run with 55 Adam epochs and 17 L-BFGS closures.
- Plateau detection triggered at epoch 54 with λ_phys = 0.398, and the transition policy was honored (L-BFGS inherited λ).
- Best val_data_loss occurred at epoch 28 (0.066085), though this checkpoint was not preserved.
- GPU memory remained far below the 3 GB gate throughout.
- The Adam stage reduced both data loss and physics loss simultaneously (visible in F1 and F6).
- L-BFGS converged on the fixed batch within 17 closures (F2), reflecting limited remaining gradient signal.

**Not supported (requires separate investigation):**
- Individual causal contribution of training duration vs λ schedule vs optimizer sequencing
- Per-epoch per-equation physics convergence (data not logged)
- Gradient norm behavior over training (data not logged)

## Reviewer Constraints Honored

| Constraint | Status |
|-----------|--------|
| No fabricated epoch-wise per-equation curve | ✓ F5 unavailable |
| No fabricated gradient-norm curve | ✓ F7 unavailable |
| GPU memory honest (per-epoch, not just peak) | ✓ F8 from `peak_vram_MB` column |
| Hyperparameter table verbatim from pre-registration | ✓ Asserted; no adjustments |
| No training performed | ✓ Read-only script |
| No checkpoint modification | ✓ |
| No test-set use | ✓ |

## Milestone

`STEP_4A.4_LOCKED — OPTIMIZATION ANALYSIS COMPLETE`