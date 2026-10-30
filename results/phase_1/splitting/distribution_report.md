# Step 1.4 Block D — Distribution + Outlier Analysis

## Summary
- Scenarios loaded: 2000
- Train: 1600 / Val: 200 / Test: 200
- Wall time: 2.4 s

## FFT frequency content (aggregate over 2,000 profiles)
- Frequency bins: 101
- Range: [0.1000, 10.0000] Hz
- Dominant frequency: 0.1000 Hz

## Cross-split consistency (KS test)
Kolmogorov–Smirnov: 2-sample test. p > 0.05 = no significant difference (good).

| Variable | KS train vs val | p | KS train vs test | p |
|----------|-----------------|---|------------------|---|
| n | 0.0233 | 0.0000 | 0.0145 | 0.0000 |
| C1 | 0.0461 | 0.0000 | 0.0192 | 0.0000 |
| C2 | 0.0469 | 0.0000 | 0.0188 | 0.0000 |
| C3 | 0.0447 | 0.0000 | 0.0168 | 0.0000 |
| C4 | 0.0411 | 0.0000 | 0.0168 | 0.0000 |
| C5 | 0.0310 | 0.0000 | 0.0148 | 0.0000 |
| C6 | 0.0259 | 0.0000 | 0.0135 | 0.0000 |
| Tf | 0.0313 | 0.0000 | 0.0142 | 0.0000 |
| Tc | 0.0342 | 0.0000 | 0.0137 | 0.0000 |
| rho_ext | 0.0278 | 0.0000 | 0.0139 | 0.0000 |

## Duplicate detection
- Threshold: cosine similarity > 0.99
- Near-duplicate pairs found: **17**
- See `duplicates.csv` for details

## Outliers flagged (±5σ from training mean)
- Scenarios with at least one outlier flag: **0**
- **Not removed** — informational only
- See `outliers_flagged.csv` for details

## Per-family ranges (a sample)

### step
- rho_ext: mean=-1.1701e-04, std=1.8724e-03, range=[-3.2250e-03, 3.2237e-03]
- n: mean=9.9583e-01, std=9.1800e-02, range=[6.7777e-01, 1.7319e+00]
- Tf: mean=8.9874e+02, std=2.7169e+01, range=[8.5112e+02, 9.5017e+02]
- Tc: mean=5.8293e+02, std=1.6003e+00, range=[5.8010e+02, 5.8598e+02]

### ramp
- rho_ext: mean=1.2528e-04, std=1.0423e-03, range=[-1.9350e-03, 1.9456e-03]
- n: mean=1.0063e+00, std=4.8813e-02, range=[8.4898e-01, 1.1818e+00]
- Tf: mean=9.0191e+02, std=1.5147e+01, range=[8.7072e+02, 9.3003e+02]
- Tc: mean=5.8311e+02, std=8.9205e-01, range=[5.8126e+02, 5.8478e+02]

### sinusoidal
- rho_ext: mean=-1.9572e-05, std=8.3428e-04, range=[-1.9462e-03, 1.9462e-03]
- n: mean=1.0008e+00, std=7.7926e-02, range=[7.5306e-01, 1.3995e+00]
- Tf: mean=9.0024e+02, std=1.0361e+01, range=[8.7076e+02, 9.2906e+02]
- Tc: mean=5.8301e+02, std=5.9504e-01, range=[5.8127e+02, 5.8472e+02]

### grf
- rho_ext: mean=3.4174e-05, std=1.1226e-03, range=[-4.8055e-03, 4.8639e-03]
- n: mean=1.0044e+00, std=1.0604e-01, range=[5.6264e-01, 2.2995e+00]
- Tf: mean=9.0146e+02, std=1.4020e+01, range=[8.4263e+02, 9.7228e+02]
- Tc: mean=5.8309e+02, std=8.0325e-01, range=[5.7965e+02, 5.8729e+02]
