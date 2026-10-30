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
| n | 4.650604e-01 | 7.333972e-02 | 2.405156e-01 | 1.286556e+00 | 3.098762e-01 | 5.350340e-01 | no |
| C1 | 1.677488e+02 | 3.086287e+00 | 1.614266e+02 | 1.734312e+02 | 1.630222e+02 | 1.727936e+02 | no |
| C2 | 4.316970e+02 | 1.865764e+01 | 3.943939e+02 | 4.671772e+02 | 4.037981e+02 | 4.629687e+02 | no |
| C3 | 9.009010e+01 | 1.142449e+01 | 6.956189e+01 | 1.150355e+02 | 7.477436e+01 | 1.113273e+02 | no |
| C4 | 5.384085e+01 | 1.108300e+01 | 3.790333e+01 | 8.583878e+01 | 4.225246e+01 | 7.862469e+01 | no |
| C5 | 3.354758e+00 | 6.467293e-01 | 2.660249e+00 | 6.710865e+00 | 2.915608e+00 | 4.902498e+00 | no |
| C6 | 4.376901e-01 | 7.733740e-02 | 2.764842e-01 | 9.592393e-01 | 3.368427e-01 | 4.980038e-01 | no |
| Tf | 7.042861e+02 | 3.370527e+01 | 6.736944e+02 | 9.000000e+02 | 6.841919e+02 | 7.739912e+02 | no |
| Tc | 6.025165e+02 | 5.178714e+00 | 5.830000e+02 | 6.228228e+02 | 5.976941e+02 | 6.162383e+02 | no |

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
