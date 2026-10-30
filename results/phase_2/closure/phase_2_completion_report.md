# Phase 2 Completion Report — Standard FNO Baseline

**Date:** 2026-10-30
**Phase:** 2 — Standard Fourier Neural Operator Baseline
**Status:** CLOSED — GATE 2 PASS across all 5 steps
**Lock tag:** `phase-2-lock`

## Section 1 — Executive Summary & Status

Phase 2 establishes and locks the Standard Fourier Neural Operator baseline against which Phase 3 SC-FNO will be compared.

| Step | Gate | Lock commit | Checks |
|---|---|---|---|
| 2.1 Architecture | PASS | 55c2482 | 10/10 |
| 2.2 Data pipeline | PASS | a5825b4 | 50/50 |
| 2.3 Training | PASS | fec9fd7 | 16/16 |
| 2.4 Evaluation | PASS | dee0c47 | 20/20 |
| 2.5 Closure | PASS | (final commit) | 16/16 |

**Reproducibility:** PASS — architecture reconstructs (50,409 params), `fno_best.pt` loads with SHA-256 match, re-inference on 5 test scenarios agrees with stored Step 2.4 predictions to per-state relative max diff < 3.39e-05.

## Section 2 — Model Architecture & Hardware Constraints

**Architecture:** 1D temporal FNO, 4 Fourier layers, latent width 32, 16 retained modes, 1 input channel (ρ_ext), 9 output channels, GELU, 200 timesteps.
**Total trainable parameters:** 50,409
**`fno_best.pt` SHA-256:** `24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889`
**Architecture checkpoint SHA-256:** `6e7bc42132dcb1e020ff1c7ef791e64a673580e7dcb88d623a8a11f435279d6d`

**Per-module parameter breakdown:**
| Module | Parameters |
|---|---|
| fno_blocks | 45,504 |
| lifting | 2,208 |
| projection | 2,697 |

Sum check: 50,409 = total: True

**Hardware:** NVIDIA RTX 4050 Laptop, 6,438 MB VRAM

| Phase | Peak allocated MB | Peak reserved MB | 3 GB target |
|---|---|---|---|
| Training (2.3) | 32.58 | 39.85 | MET |
| Inference B200 (2.4) | 51.05 | — | MET |

*`peak allocated = torch.cuda.max_memory_allocated()` — not total process GPU usage. See `vram_clarification.md`.*

## Section 3 — Data Processing & Normalization

**Dataset:** Phase 1 locked v4
**Aggregate SHA-256:** `9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6`
**Manifest SHA-256:** `993581ef514932c5f3c341ce176a401a5247ebe7217e83f70d3ebe26dd3cff12`

| Split | Count | SHA-256 |
|---|---|---|
| Train | 1600 | `16ad4670eee9555aec237c94d2d241e90b74b50ce8ea57eb07efe209fe852834` |
| Validation | 200 | `19b781283503b1a1cdb313dc68ce153086f18c25043f1ff5123e3cb3e3205bb6` |
| Test | 200 | `a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2` |

**Normalization:** training-only z-score, SHA-256 `4e832aa5cdaa940cb2e07bf64c5741cc5124ad40242dd9532da231ff257f6727`
**Family stratification:** 400/50/50 across 4 families per split — evidenced by Phase 1 `split_manifest.json` (read-only reference).

**Metric aggregation definitions:** see `metric_definitions.md`. MAE/RMSE/RelL2 in **physical units**. Overall metrics aggregate over 1800 (scenario, state) pairs. Per-state medians over 200 scenarios.

## Section 4 — Training Execution & Convergence

**Optimizer:** Adam lr=1e-3, betas=(0.9, 0.999)
**Scheduler:** CosineAnnealing, T_max=500, eta_min=0
**Batch size:** 16 | **Max epochs:** 500 | **Patience:** 30 (on val MSE)
**Gradient clipping:** max_norm=1.0 (fired 1/319 epochs — epoch 1 only)
**Precision:** FP32 | **Seed:** 42 (deterministic)

| Metric | Value |
|---|---|
| Epochs completed | 319 / 500 (early stopped) |
| Best epoch | 289 |
| Best val MSE | 6.666703e-02 |
| Total wall time | 8.39 min |
| Median per-step | 9.126 ms |
| Stability event | `early_stop_epoch_319` |

**Timing methodology:** 3 warm-up iterations, 10 measurement repeats, `torch.cuda.synchronize()` before/after each repeat, `time.perf_counter()`, median primary.
**No NaN, no exploding gradients, no OOM.**

## Section 5 — Required Manuscript Outputs (Evidence)

### Table 1 — Per-State Test Metrics (200 scenarios, physical units)

| State | MAE mean | RMSE mean | RelL2 median | Max abs |
|---|---|---|---|---|
| C1 | 1.170e-01 | 1.364e-01 | 2.008e-04 | 1.593e+00 |
| C2 | 7.031e-01 | 8.199e-01 | 4.538e-04 | 9.925e+00 |
| C3 | 4.184e-01 | 4.926e-01 | 1.136e-03 | 6.918e+00 |
| C4 | 3.754e-01 | 4.666e-01 | 1.705e-03 | 8.730e+00 |
| C5 | 1.537e-02 | 2.858e-02 | 2.173e-03 | 8.860e-01 |
| C6 | 4.036e-03 | 6.500e-03 | 3.816e-03 | 1.666e-01 |
| Tc | 5.092e-02 | 9.494e-02 | 8.375e-05 | 2.645e+00 |
| Tf | 8.145e-01 | 1.464e+00 | 8.203e-04 | 4.518e+01 |
| n | 6.017e-03 | 1.236e-02 | 7.546e-03 | 5.849e-01 |

### Table 2 — Per-Family Aggregate Test Metrics

| Family | MAE mean | RMSE mean | RelL2 median |
|---|---|---|---|
| grf | 2.7653e-01 | 3.7063e-01 | 2.0580e-03 |
| ramp | 2.0664e-02 | 2.6507e-02 | 2.1153e-04 |
| sinusoidal | 8.9915e-02 | 1.3485e-01 | 6.4651e-04 |
| step | 7.2612e-01 | 1.0331e+00 | 5.8960e-03 |

### Table 3 — Worst-Case Aggregate Scenario

`scenario_0458` (family=step) — mean RelL2 over 9 states = 1.8516e-02

### Table 4 — Per-State Worst-Case Scenarios

| State | Scenario | Family | RelL2 |
|---|---|---|---|
| n | scenario_0290 | step | 5.3008e-02 |
| C1 | scenario_0458 | step | 4.8134e-03 |
| C2 | scenario_0458 | step | 1.0897e-02 |
| C3 | scenario_0458 | step | 2.7761e-02 |
| C4 | scenario_0458 | step | 3.6169e-02 |
| C5 | scenario_0458 | step | 2.2425e-02 |
| C6 | scenario_1989 | grf | 4.3737e-02 |
| Tf | scenario_0458 | step | 7.9293e-03 |
| Tc | scenario_0458 | step | 8.1790e-04 |

### Table 5 — Speedup Decomposition (Reviewer Item 1)

| Quantity | Value | Label |
|---|---|---|
| Single-scenario latency | 11.54× (BDF 35.3 ms vs FNO B1 3.059 ms) | `single_scenario_latency` |
| Batch throughput | 64418 scenarios/sec at B200 | `batch_throughput` |
| Batch-amortized per-scenario | 2273.9× (B200 amortized vs BDF per-scenario) | `batch_amortized` |

**Primary for manuscript:** `single_scenario_latency_speedup` — always label the ~2274× figure as *batch-amortized*, never *single-scenario latency*.

### Figures (Step 2.4)

- `reports/figures/phase_2/f1_training_validation_loss.png`
- `reports/figures/phase_2/f2_neutron_density_prediction.png`
- `reports/figures/phase_2/f3_precursor_predictions.png`
- `reports/figures/phase_2/f4_fuel_temperature_prediction.png`
- `reports/figures/phase_2/f5_coolant_temperature_prediction.png`
- `reports/figures/phase_2/f6_statewise_error_comparison.png`
- `reports/figures/phase_2/worst_case_prediction.png`

## Section 6 — Phase 2 PASS/FAIL Gate Checklist


| Step | Criteria | Result |
|---|---|---|
| 2.1 Architecture | 10 | PASS |
| 2.2 Data pipeline | 50 | PASS |
| 2.3 Training | 16 | PASS |
| 2.4 Evaluation | 20 | PASS |
| 2.5 Closure | 16 | PASS |

**Total: 112/112 checks passed across all 5 gates.**

**Final declaration:** Phase 2 is officially CLOSED. Baseline FNO is locked and reproducible. Phase 3 (Physics-Informed SC-FNO) is AUTHORIZED.

## Section 5 (addendum) — Template Compliance Tables

### Table 1 — FNO Architecture and Hyperparameters
*Full CSV: `reports/tables/phase_2/table_1_architecture.csv`*

| Parameter | Value |
|-----------|-------|
| Architecture | 1D temporal FNO |
| Fourier layers | 4 |
| Latent width | 32 |
| Retained Fourier modes | 16 |
| Input channels | 1 (ρ_ext) |
| Output channels | 9 |
| Temporal points | 200 |
| Activation | GELU |
| Total trainable parameters | 50,409 |
|   fno_blocks | 45,504 |
|   lifting | 2,208 |
|   projection | 2,697 |
| Optimizer | Adam |
| Initial LR | 1e-3 |
| Scheduler | CosineAnnealingLR (T_max=500) |
| Batch size | 16 |
| Max epochs | 500 (early stopped at 319) |
| Best epoch | 289 |
| Best val MSE | 0.066667 |
| Gradient clip max_norm | 1.0 |
| Seed | 42 |
| Precision | FP32 |

### Table 2 — Focused Baseline Predictive Error: n, Tf, Tc
*Full CSV: `reports/tables/phase_2/table_2_focused_n_Tf_Tc.csv`*

| State | MAE mean | RMSE mean | RelL2 median | Max abs |
|-------|----------|-----------|--------------|---------|
| **n** (neutron density) | 6.017e-03 | 1.236e-02 | **7.546e-03** | 5.849e-01 |
| **Tf** (fuel temperature, K) | 8.145e-01 | 1.464e+00 | 8.200e-04 | 4.518e+01 |
| **Tc** (coolant temperature, K) | 5.092e-02 | 9.494e-02 | 8.404e-05 | 2.645e+00 |

**Notes:**
- **n** is the weakest state by relative L2 (7.55e-3), a candidate for Phase 3 improvement.
- **Tc** is the strongest state by relative L2 (8.40e-5).
- **Tf** shows the largest absolute errors (max 45.18 K) despite low relative L2, because fuel temperature magnitude is large (~900 K).

### Graph — Training and Validation Loss
*File: `reports/figures/phase_2/f1_training_validation_loss.png`* ✅ generated

### Graph — Worst-Case Prediction
*File: `reports/figures/phase_2/worst_case_prediction.png`* ✅ generated
Worst-case scenario: `scenario_0458` (step family, mean RelL2 = 1.85e-2).
