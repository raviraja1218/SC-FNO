# Physics Revalidation Addendum

**Date:** 2026-11-02

## Defects Corrected

1. **Per-state derivative:** original `compute_du_dt` computed `∂(Σ_j u_j)/∂t` and assigned to all 9 channels; corrected to per-channel `∂u_j/∂t`.
2. **Forcing interpolation:** original used `rho_fn = zeros`; corrected to interpolate actual scenario forcing at query times.

## Comparison

| Model | Original (per-sum) | Corrected (per-channel) | Ratio |
|-------|-------------------|------------------------|-------|
| P3-PI-FNO | 0.0016723 | 125890099.35178006 | 1.3283808723726726e-11 |
| Phase4A.1 | 1.9519e-06 | 125904841.34646598 | 1.5502978115263616e-14 |
| P4-A | 174.07 | 122913714.49552794 | 1.4161967256008142e-06 |
| P4-B | 204.41 | 121658263.72859743 | 1.680198235082576e-06 |
| P4-C | 182.28 | 122811908.34227623 | 1.484220890794942e-06 |
| P4-D | 225.58 | 122817052.87604943 | 1.8367156247240517e-06 |

## Per-state physics loss (corrected protocol)

| Model | n | C1 | C2 | C3 | C4 | C5 | C6 | Tf | Tc |
|-------|---|---|---|---|---|---|---|---|---|
| P3-PI-FNO | 1.13e+09 | 2.62e+01 | 9.31e-01 | 3.77e-01 | 6.19e-02 | 3.06e-01 | 3.64e+00 | 4.39e-04 | 5.64e-02 |
| Phase4A.1 | 1.13e+09 | 1.90e+00 | 2.95e-02 | 1.25e-02 | 3.39e-03 | 1.64e-02 | 8.99e-02 | 1.13e-05 | 1.22e-02 |
| P4-A | 1.11e+09 | 3.88e+03 | 4.10e+01 | 6.85e+00 | 2.93e+00 | 5.65e+00 | 1.73e+01 | 1.85e+01 | 6.15e+01 |
| P4-B | 1.09e+09 | 1.80e+04 | 4.44e+01 | 1.49e+01 | 3.18e+00 | 9.07e+00 | 4.90e+01 | 1.27e+01 | 7.46e+01 |
| P4-C | 1.11e+09 | 9.68e+03 | 5.09e+01 | 1.20e+01 | 4.17e+00 | 8.29e+00 | 2.55e+01 | 1.67e+01 | 9.64e+01 |
| P4-D | 1.11e+09 | 6.42e+04 | 1.88e+02 | 4.44e+01 | 6.77e+00 | 1.97e+01 | 9.73e+01 | 7.11e+00 | 1.53e+02 |
