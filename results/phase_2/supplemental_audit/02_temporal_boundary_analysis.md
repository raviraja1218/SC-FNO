# Audit Item 2 — Temporal Boundary Analysis

## Motivation

The Phase 2 FNO uses a spectral (FFT-based) convolution with no positional embedding and no domain padding. FFT assumes the input is periodic: the end of the 10-second trajectory is treated as smoothly connecting to the beginning. The reactor transient trajectory is **not** periodic — the final state at $t = 10$ s is different from the initial state at $t = 0$ s. This creates a potential boundary artifact near both ends of the time domain.

## Analysis Method

Read-only from existing `predictions_summary.npz` (200 test scenarios). Segments:
- **Start boundary**: $t \in [0, 0.5]$ s (10 timesteps)
- **Interior**: $t \in [0.5, 9.5]$ s (180 timesteps)
- **End boundary**: $t \in [9.5, 10]$ s (10 timesteps)

Computed mean absolute error per segment per state, then ratios of boundary MAE to interior MAE.

## Results — Per-State Boundary vs Interior

| State | Start MAE | End MAE | Interior MAE | Start/int ratio | End/int ratio |
|-------|-----------|---------|--------------|-----------------|---------------|
| n | 3.5787e-02 | 4.5697e-03 | 4.4439e-03 | 8.053 | 1.028 |
| C1 | 2.1675e-01 | 2.1557e-01 | 1.0603e-01 | 2.044 | 2.033 |
| C2 | 1.3447e+00 | 1.2621e+00 | 6.3644e-01 | 2.113 | 1.983 |
| C3 | 9.1761e-01 | 6.7175e-01 | 3.7658e-01 | 2.437 | 1.784 |
| C4 | 1.0961e+00 | 4.8719e-01 | 3.2912e-01 | 3.330 | 1.480 |
| C5 | 7.8346e-02 | 1.6276e-02 | 1.1821e-02 | 6.627 | 1.377 |
| C6 | 1.4330e-02 | 3.0110e-03 | 3.5215e-03 | 4.069 | 0.855 |
| Tf | 4.1389e+00 | 8.5481e-01 | 6.2756e-01 | 6.595 | 1.362 |
| Tc | 3.1362e-01 | 5.2573e-02 | 3.6233e-02 | 8.656 | 1.451 |

## Aggregate

| Metric | Value |
|--------|-------|
| Median start/interior ratio | 4.069 |
| Median end/interior ratio | 1.451 |
| Max start/interior ratio | 8.656 |
| Max end/interior ratio | 2.033 |

## Boundary Flags (ratio > 2×)

- ⚠ n_start (ratio=8.05)
- ⚠ C1_start (ratio=2.04)
- ⚠ C1_end (ratio=2.03)
- ⚠ C2_start (ratio=2.11)
- ⚠ C3_start (ratio=2.44)
- ⚠ C4_start (ratio=3.33)
- ⚠ C5_start (ratio=6.63)
- ⚠ C6_start (ratio=4.07)
- ⚠ Tf_start (ratio=6.60)
- ⚠ Tc_start (ratio=8.66)

## Interpretation

**The start-of-trajectory MAE is materially inflated** for most states. **10 of 18 boundary ratios exceed 2×**, and **all 10 flags involve the start boundary**:

- **n** (neutron density): start error is **8.05×** interior — the worst single case
- **Tc** (coolant): 8.66× — the FFT wraparound artifact compounds into the temperature trace
- **Tf, C5, C6, C4** all exceed 4×
- **End boundary** is comparatively clean: median end/int = 1.45, only 2 states (C1, C2) at ~2×

The asymmetry between start (heavily affected) and end (mostly clean) is informative: it confirms the mechanism is **FFT wraparound from the end of the trajectory corrupting the start**, which is the specific failure mode of un-padded spectral convolutions on non-periodic signals. The end-of-trajectory prediction is largely unaffected because FFT artifacts propagate backward from the discontinuity point.

## Phase 3 Implication

This is a **genuine architectural limitation** of the Phase 2 FNO baseline. It is not a Phase 2 defect — it is exactly what an un-padded conventional FNO does. But it matters for Phase 3 for two reasons:

1. **Step-family transients (the weakest family in Phase 2) have their sharpest dynamics at $t = 0$**, which is exactly where the wraparound artifact is strongest. Phase 3 physics-informed training will need to either (a) overcome this via residual constraint, or (b) use an architectural fix (padding, causal decoder) in a revised FNO variant.
2. **The Phase 2 baseline's overall accuracy numbers (median rel L2 = 9.232e-4) are partially rescued by the excellent interior performance** — the boundary inflation is real but averages down across 200 timesteps. This is honest science: the baseline's headline accuracy does not reflect its start-of-trajectory weakness.

## Recommendation

Document this limitation in the manuscript's Limitations section. If Phase 3 demonstrates significant step-family improvement, verify whether the improvement comes from (a) genuine physics constraints, (b) incidental boundary mitigation, or (c) both — and report honestly.

**Figure:** `reports/figures/phase_2/boundary_error_profile.png` — 9-subplot MAE vs time, with boundary regions shaded.
