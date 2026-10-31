# Step 3.3 — Composite Loss Specification

## Frozen architecture

Data path and physics path are separated (Step 3.2 v2.1 addendum):

| Loss | Prediction | Parameters trained |
|------|-----------|-------------------|
| Data | `u_data = fno(rho_ext)` → `[B, 9, 200]` | `lifting` + `fno_blocks` + `projection` |
| Physics | `u_phys = decoder(z, t_q)`, `z` = pre-projection latent | `lifting` + `fno_blocks` + `decoder` |
| IC | `u_IC = decoder(z, t=0)` | `lifting` + `fno_blocks` + `decoder` |

Shared: `lifting` + `fno_blocks` (36 tensors).

## Composite loss

$$\mathcal{L}_{total} = \lambda_{data} \mathcal{L}_{data} + \lambda_{phys} \mathcal{L}_{physics} + \lambda_{IC} \mathcal{L}_{IC}$$

### Data loss
MSE on normalized 9-state predictions at 200 grid points vs ground truth.

### Physics loss
$$\mathcal{L}_{physics} = \frac{1}{9 N_q} \sum_{j=1}^{9} \sum_{k=1}^{N_q} \left(\frac{R_j(t_k)}{s_j}\right)^2$$
where `s_j = P95(|R_j|)` from Step 3.1 (per-state residual scale).
`N_q = 50` query times per batch element.
Residuals computed with **denormalized** u_hat and **denormalized** du_hat_dt (state-wise scaling).

### IC loss
$$\mathcal{L}_{IC} = \frac{1}{9 B} \sum_{b=1}^{B} \sum_{j=1}^{9} \left(\frac{\hat{u}_{IC,b,j}(0) - u_{gt,b,j}(0)}{s_{uj}}\right)^2$$
where `s_uj = normalization_std[j]` from Phase 1 training-only stats.

### Weight schedule (frozen for Step 3.3)

| Weight | Initial | Schedule |
|--------|---------|----------|
| λ_data | 1.0 | constant |
| λ_phys | 1e-4 | linear ramp to 1.0 over 100 epochs (Step 3.4) |
| λ_IC | 1.0 | constant |

Candidate — may be tuned in Step 3.4 based on gradient conflict monitoring.

## Per-state scaling (from Step 3.3.A)

| State | Residual p95 | Normalization std |
|-------|--------------|-------------------|
| n | 2.906e-01 (1/s) | 8.406e-02 |
| C1 | 1.739e-02 (1/s) | 6.471e-01 |
| C2 | 1.152e-01 (1/s) | 4.010e+00 |
| C3 | 1.034e-01 (1/s) | 2.742e+00 |
| C4 | 2.087e-01 (1/s) | 3.396e+00 |
| C5 | 6.138e-02 (1/s) | 3.535e-01 |
| C6 | 2.248e-02 (1/s) | 5.674e-02 |
| Tf | 2.941e+00 (K/s) | 1.794e+01 |
| Tc | 7.174e-02 (K/s) | 1.050e+00 |

## Item C verification (reviewer)

- **C1** (IC loss → decoder bias grad ≠ 0): PASSED
- **C2** (data loss → decoder bias grad = None; projection bias grad ≠ 0): PASSED
