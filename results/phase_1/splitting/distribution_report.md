# Step 1.4 Block D — Distribution + Outlier Analysis

## Summary
- Scenarios loaded: 2000
- Train: 1600 / Val: 200 / Test: 200
- Wall time: 1.7 s

## FFT frequency content (aggregate over 2,000 profiles)
- Frequency bins: 101
- Range: [0.1000, 10.0000] Hz
- Dominant frequency: 0.1000 Hz

## Cross-split consistency (KS test)
Kolmogorov–Smirnov: 2-sample test. p > 0.05 = no significant difference (good).

| Variable | KS train vs val | p | KS train vs test | p |
|----------|-----------------|---|------------------|---|
| n | 0.0145 | 0.0000 | 0.0069 | 0.0691 |
| C1 | 0.0019 | 0.9996 | 0.0007 | 1.0000 |
| C2 | 0.0020 | 0.9990 | 0.0009 | 1.0000 |
| C3 | 0.0029 | 0.9274 | 0.0012 | 1.0000 |
| C4 | 0.0061 | 0.1374 | 0.0036 | 0.7380 |
| C5 | 0.0188 | 0.0000 | 0.0085 | 0.0120 |
| C6 | 0.0144 | 0.0000 | 0.0076 | 0.0341 |
| Tf | 0.0190 | 0.0000 | 0.0085 | 0.0115 |
| Tc | 0.0168 | 0.0000 | 0.0070 | 0.0603 |
| rho_ext | 0.0278 | 0.0000 | 0.0139 | 0.0000 |

## Duplicate detection
- Threshold: cosine similarity > 0.99
- Near-duplicate pairs found: **106434**
- See `duplicates.csv` for details

## Outliers flagged (±5σ from training mean)
- Scenarios with at least one outlier flag: **2000**
- **Not removed** — informational only
- See `outliers_flagged.csv` for details

## Per-family ranges (a sample)

### step
- rho_ext: mean=-1.1701e-04, std=1.8724e-03, range=[-3.2250e-03, 3.2237e-03]
- n: mean=4.6303e-01, std=7.4697e-02, range=[2.5340e-01, 1.0366e+00]
- Tf: mean=7.0379e+02, std=3.4422e+01, range=[6.7524e+02, 9.0000e+02]
- Tc: mean=6.0238e+02, std=5.5065e+00, range=[5.8300e+02, 6.2160e+02]

### ramp
- rho_ext: mean=1.2528e-04, std=1.0423e-03, range=[-1.9350e-03, 1.9456e-03]
- n: mean=4.6686e-01, std=6.8846e-02, range=[2.6414e-01, 1.0000e+00]
- Tf: mean=7.0472e+02, std=3.3419e+01, range=[6.8141e+02, 9.0000e+02]
- Tc: mean=6.0263e+02, std=5.0818e+00, range=[5.8300e+02, 6.1964e+02]

### sinusoidal
- rho_ext: mean=-1.9572e-05, std=8.3428e-04, range=[-1.9462e-03, 1.9462e-03]
- n: mean=4.6469e-01, std=7.2075e-02, range=[2.6088e-01, 1.0000e+00]
- Tf: mean=7.0418e+02, std=3.3404e+01, range=[6.8138e+02, 9.0000e+02]
- Tc: mean=6.0249e+02, std=5.0234e+00, range=[5.8300e+02, 6.2058e+02]

### grf
- rho_ext: mean=3.4174e-05, std=1.1226e-03, range=[-4.8055e-03, 4.8639e-03]
- n: mean=4.6597e-01, std=7.7280e-02, range=[2.4052e-01, 1.2866e+00]
- Tf: mean=7.0454e+02, std=3.3528e+01, range=[6.7369e+02, 9.0000e+02]
- Tc: mean=6.0259e+02, std=5.0735e+00, range=[5.8300e+02, 6.2282e+02]
