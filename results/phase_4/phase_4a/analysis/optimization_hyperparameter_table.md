# Step 4A.4 — Optimization Hyperparameter Table

**All values verbatim from `pre_registration.yaml` (no retrospective adjustments).**

| Group | Parameter | YAML Path | Value |
|-------|-----------|-----------|-------|
| Config | Starting checkpoint | `starting_checkpoint.path` | `models/sc_fno/phase_3/pi_fno/best_val.pt` |
| Config | Checkpoint SHA | `starting_checkpoint.sha256` | `a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50` |
| Config | Checkpoint epoch | `starting_checkpoint.epoch` | `36` |
| Config | Starting λ_phys | `starting_checkpoint.lam_phys` | `0.002754` |
| Config | Seed | `seed` | `42` |
| Config | Device | `device` | `cuda` |
| Config | Batch size (training) | `batch_size_training` | `16` |
| Adam | Learning rate | `adam.lr` | `0.001` |
| Adam | Betas | `adam.betas` | `[0.9, 0.999]` |
| Adam | Eps | `adam.eps` | `1e-08` |
| Adam | Weight decay | `adam.weight_decay` | `0.0` |
| Adam | State | `adam.state` | `fresh` |
| Scheduler | Class | `scheduler.class` | `CosineAnnealingLR` |
| Scheduler | T_max | `scheduler.T_max` | `100` |
| Scheduler | eta_min | `scheduler.eta_min` | `0.0` |
| λ_phys | Init | `lambda_phys.init` | `0.002754` |
| λ_phys | Target | `lambda_phys.target` | `1.0` |
| λ_phys | Ramp epochs | `lambda_phys.ramp_epochs` | `64` |
| λ_phys | Transition policy | `lambda_phys.transition_policy` | `B_inherit_lambda` |
| Plateau | Metric | `plateau_detection.metric` | `validation_data_loss` |
| Plateau | Protocol | `plateau_detection.metric_protocol` | `A_phase3_exact_batch16` |
| Plateau | EMA alpha | `plateau_detection.ema_alpha` | `0.1` |
| Plateau | Reference window | `plateau_detection.reference_window` | `20` |
| Plateau | Delta | `plateau_detection.delta` | `0.0001` |
| Plateau | Patience | `plateau_detection.patience` | `3` |
| Plateau | Min Adam epochs | `plateau_detection.min_adam_epochs` | `20` |
| Plateau | Max Adam epochs | `plateau_detection.max_adam_epochs` | `100` |
| L-BFGS | Class | `lbfgs.class` | `torch.optim.LBFGS` |
| L-BFGS | History size | `lbfgs.history_size` | `50` |
| L-BFGS | Max iter | `lbfgs.max_iter` | `50000` |
| L-BFGS | tolerance_grad | `lbfgs.tolerance_grad` | `1e-07` |
| L-BFGS | tolerance_change | `lbfgs.tolerance_change` | `1e-09` |
| L-BFGS | line_search_fn | `lbfgs.line_search_fn` | `strong_wolfe` |
| L-BFGS | Fixed batch source | `lbfgs.fixed_batch.source` | `train_ids.csv first 16 scenarios` |

## Measured Outcomes (from training_summary.json)

| Metric | Value |
|--------|-------|
| Adam epochs completed | 55 |
| Adam wall time | 269.8 s |
| Transition epoch | 54 |
| Transition λ_phys | 0.398102 |
| Final val_data_loss at plateau | 0.066702 |
| L-BFGS wall time | 0.887 s |
| L-BFGS closure calls | 17 |
| L-BFGS final loss | 0.599033 |
| Peak VRAM (Adam) | 69.29 MB |
| Peak VRAM (L-BFGS) | 78.17 MB |