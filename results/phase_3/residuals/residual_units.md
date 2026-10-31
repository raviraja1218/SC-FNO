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
