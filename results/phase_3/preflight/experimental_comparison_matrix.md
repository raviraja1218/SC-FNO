# Experimental Comparison Matrix — Phase 3 SC-FNO

**Date:** 2026-10-31
**Decision ID:** ADR-3.0-003
**Status:** FROZEN (all variants trained in Step 3.4)

## Purpose

Separate the effect of **physics constraints** from the effect of
**temporal padding** on the observed improvement. If both are introduced
simultaneously, improvement cannot be attributed exclusively to physics.

## Four controlled variants

| Variant | Physics loss | Padding | Initialization | Purpose |
|---------|-------------|---------|----------------|---------|
| **P2-FNO** | No | No | Phase 2 `fno_best.pt` | **Locked reference** (Phase 2) |
| **P-FNO** | No | Yes (240-pt) | Phase 2 `fno_best.pt` | **Isolate padding effect** |
| **PI-FNO** | Yes | No | Phase 2 `fno_best.pt` | **Isolate physics effect** |
| **P-PI-FNO** | Yes | Yes (240-pt) | Phase 2 `fno_best.pt` | **Combined intervention** (primary candidate) |

### Padding mode sub-variants for P-FNO and P-PI-FNO

Each of the 3 padding modes produces a separate run:

- P-FNO-zero, P-FNO-reflect, P-FNO-replicate
- P-PI-FNO-zero, P-PI-FNO-reflect, P-PI-FNO-replicate

Total runs: 1 (P2-FNO baseline, no training) + 3 (P-FNO) + 1 (PI-FNO) + 3 (P-PI-FNO) = **8 models evaluated**

### Selection of primary SC-FNO candidate

**P-PI-FNO variant selected only if:**
1. Padding-only variant (P-FNO) shows measurable boundary improvement
2. Physics loss improves step-family accuracy without >5% degradation on ramp/sinusoidal
3. Combined variant achieves strict superset of both effects

If padding-only does NOT improve boundary ratio, then primary candidate = PI-FNO (physics only).

## Success criteria per variant

### P-FNO (padding-only)

- **Boundary ratio** (n, Tc): measured vs P2-FNO 8.05× / 8.66×
- **Step-family Rel L2:** measured vs P2-FNO 5.896e-3
- **Ramp/sinusoidal/GRF:** measured vs P2-FNO (regression check)
- **No physics loss** — this variant tests whether padding alone is sufficient

### PI-FNO (physics-only)

- **Physics loss convergence:** ≥ 10× reduction from epoch-1 reference
- **Step-family Rel L2:** measured, target < 5.896e-3
- **Ramp/sinusoidal:** ≤ 5% degradation
- **Boundary ratio:** measured (physics may indirectly improve start-of-trajectory via residual constraint)

### P-PI-FNO (combined)

- All PI-FNO criteria
- Boundary ratio < 2.0 (per padding mode)
- **Attribution:** improvement must be decomposable into (padding effect) + (physics effect) + (synergy or interaction)

## Attribution analysis protocol

After all variants trained, compute:

    improvement_total = P2_FNO_metric - P_PI_FNO_metric
    improvement_padding = P2_FNO_metric - P_FNO_metric
    improvement_physics = P2_FNO_metric - PI_FNO_metric
    synergy = improvement_total - improvement_padding - improvement_physics

Report:
- Positive synergy → padding + physics together better than sum of parts
- Zero synergy → independent effects
- Negative synergy → interference

## Baseline preservation

- **P2-FNO checkpoint:** `models/baseline/fno_best.pt` (SHA `24fce0ba...`) **unchanged**
- **P2-FNO test metrics:** `results/phase_2/evaluation/*` **unchanged**
- **Phase 1 dataset:** aggregate `9049c7b4...` **unchanged**

## Fairness rules

1. Same Phase 1 dataset + split + normalization
2. Same 200 untouched test scenarios
3. Same metric definitions (median over 1800 pair-level Rel L2)
4. Report both improvements AND regressions (no cherry-picking)
5. Report confidence intervals for paired improvements
6. Compare against `fno_best.pt`, not retrained baseline
