# Residual Units and Scaling — Step 3.1

## Physical dimensions

| Residual | Physical dimension | Formula |
|----------|-------------------|---------|
| R_n | neutron-density / second | dn/dt − PKE rate |
| R_Ci (i=1..6) | precursor-concentration / second | dC_i/dt − PKE rate |
| R_Tf | K / second | dT_f/dt − fuel energy balance |
| R_Tc | K / second | dT_c/dt − coolant energy balance |

## Why units matter

Residuals are NOT directly comparable across states. R_n operates on
normalized neutron density (order 1), while R_Tf operates on temperature
(order 1000 K). Without per-state scaling, the loss would be dominated by
whichever state has the largest numerical magnitude — not the largest
physical violation.

Step 3.3 will divide each residual by its characteristic scale measured in
`residual_scales.json` (per-state p95 magnitude). The measured scales come
from finite differences on training trajectories, matching the units of the
autograd derivatives used during actual training.

## No leakage in measurement

The `residual_scales.json` measurement uses ground-truth trajectories to
establish *typical magnitudes*. It does NOT compute the training loss.
During training (Step 3.4), residuals are computed on **predicted** u_hat
from the model — same physical units, same scales, but predicted values.

---

## Clarifications (Reviewer follow-up)

### p95 scale definition

The characteristic scales in `residual_scales.json` are:

    s_j = P95(|R_j|)

i.e., the **95th percentile of the absolute residual magnitude** for state j,
measured on 20,000 finite-difference samples across 100 training scenarios
(200 samples per scenario × 100 scenarios). This is the positive
loss-normalization scale for state j in Step 3.3. It is **not** a signed
percentile and **not** a confidence bound.

### Ground-truth scales are NOT model-residual scales

These scales come from finite differences on **ground-truth** trajectories.
Once the model predicts its own trajectories during training, actual
residual magnitudes may be significantly larger — that is expected and is
the entire point of the physics loss. Step 3.3/3.4 must monitor actual
training residual magnitude rather than assume it stays within the p95
scale measured here.

### Nonzero source-term branch

`compute_residuals(..., q_t=...)` supports nonzero external source. This
path is now covered by test T10 (`test_T10_nonzero_source_term`), which
verifies that the residual shifts by exactly `-q_t` when `q_t` is supplied.

### ResidualOutput.stacked() interface

`ResidualOutput.stacked()` returns `[B, 9, N]` residuals in state ordering
`[n, C1..C6, Tf, Tc]`. Verified by test T9 (`test_T9_stacked_output_interface`).
