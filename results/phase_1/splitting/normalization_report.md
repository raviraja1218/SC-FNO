# Step 1.4 Block C — Normalization Statistics Report

## Source
- Training set only: **1600** scenarios (from `train_ids.csv`)
- Split seed: `split_v1_2026`
- Validation and test data: **NOT loaded** (guaranteed no leakage)

## Method
- Primary: **z-score** — `x_norm = (x − mean) / std`
- Optional per-time-point: `x_norm(t) = (x(t) − mean(t)) / std(t)`
- Fallback: min-max available from `min/max` fields
- Raw data is **not modified** — normalization is applied at training time by Phase 2 dataloaders

## Variables
rho_ext, n, C1, C2, C3, C4, C5, C6, Tf, Tc

## Global statistics (training set, all time points)

| Variable | mean | std | min | max | p5 | p95 | near-zero std? |
|----------|------|-----|-----|-----|-----|-----|----------------|
| rho_ext | 9.558930e-07 | 1.287255e-03 | -4.805469e-03 | 4.863916e-03 | -2.204005e-03 | 2.166307e-03 | no |
| n | 1.001622e+00 | 8.406447e-02 | 5.626406e-01 | 2.276468e+00 | 8.743834e-01 | 1.130677e+00 | no |
| C1 | 1.734083e+02 | 6.471224e-01 | 1.704070e+02 | 1.778023e+02 | 1.722903e+02 | 1.745051e+02 | no |
| C2 | 4.670169e+02 | 4.009489e+00 | 4.488481e+02 | 4.935501e+02 | 4.600664e+02 | 4.738267e+02 | no |
| C3 | 1.148644e+02 | 2.741634e+00 | 1.035785e+02 | 1.311900e+02 | 1.100802e+02 | 1.195654e+02 | no |
| C4 | 8.542417e+01 | 3.396224e+00 | 7.362846e+01 | 1.022589e+02 | 7.946121e+01 | 9.124342e+01 | no |
| C5 | 6.571628e+00 | 3.535424e-01 | 5.408572e+00 | 7.984104e+00 | 5.975841e+00 | 7.167395e+00 | no |
| C6 | 9.084327e-01 | 5.673977e-02 | 6.682944e-01 | 1.278489e+00 | 8.140383e-01 | 1.006004e+00 | no |
| Tf | 9.005170e+02 | 1.794549e+01 | 8.426325e+02 | 9.722812e+02 | 8.702022e+02 | 9.307596e+02 | no |
| Tc | 5.830309e+02 | 1.050461e+00 | 5.796492e+02 | 5.872864e+02 | 5.812482e+02 | 5.847956e+02 | no |

## Per-time-point statistics
- Computed at each of 200 time points across the training set
- Stored in `normalization_statistics.json` under `per_time_point`
- Useful for time-conditioned normalization if Phase 2 requires it

## Near-zero variance flags
None — all variables have std > 1e-6.

## Validation
- Loaded scenarios: 1600 (exactly the training split)
- Variables with stats: 10 (all required)
- Per-time-point arrays: 10 × 200 values each
- JSON file written: `normalization_statistics.json`
