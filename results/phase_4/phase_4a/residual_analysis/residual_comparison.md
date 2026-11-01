# Step 4A.2 — Residual Convergence Assessment

**Phase 4A.1 v2 vs Phase 3 PI-FNO — 200 validation scenarios**

- Phase 3 PI-FNO SHA: `a0522e57f0b43c8a...`
- Phase 4A.1 final SHA: `681fe6789dd2911a...`

## Aggregate Metrics

| Metric | Phase 3 | Phase 4A.1 | Δ | Δ% |
|--------|---------|------------|---|----|
| L_data | 0.066331 | 0.066678 | +0.000347 | +0.52% |
| L_physics | 1.6807e-03 | 2.0289e-06 | -1.6787e-03 | -99.88% |
| L_IC | 5.0996e-07 | 1.2592e-08 | -4.9737e-07 | -97.53% |

## Physics Target (10⁻⁵)

- Phase 3: **NOT ACHIEVED** (168.07× target)
- Phase 4A.1: **ACHIEVED** (0.203× target → 4.9× margin)

## Per-State Residuals

| State | Phase 3 | Phase 4A.1 | Δ% |
|-------|---------|------------|-----|
| n | 1.897e+03 | 1.897e+03 | +0.01% |
| C1 | 2.789e-03 | 1.758e-06 | -99.94% |
| C2 | 3.039e-03 | 4.796e-06 | -99.84% |
| C3 | 1.707e-03 | 1.973e-06 | -99.88% |
| C4 | 1.159e-03 | 1.127e-06 | -99.90% |
| C5 | 5.273e-04 | 7.698e-07 | -99.85% |
| C6 | 1.911e-03 | 4.421e-06 | -99.77% |
| Tf | 4.240e-04 | 1.172e-06 | -99.72% |
| Tc | 1.479e-03 | 8.176e-06 | -99.45% |

**Key finding:** 8 of 9 states improved by 180–1,030×. R_n (neutron density residual) is unchanged at ~1,897 and dominates the aggregate L_physics.

## Boundary Behavior (start/interior ratio)

| State | Phase 3 | Phase 4A.1 | Δ |
|-------|---------|------------|---|
| n | 5.420 | 5.487 | +0.067 |
| C1 | 2.094 | 2.090 | -0.004 |
| C2 | 2.155 | 2.150 | -0.005 |
| C3 | 2.444 | 2.438 | -0.007 |
| C4 | 3.169 | 3.178 | +0.008 |
| C5 | 4.165 | 4.506 | +0.340 |
| C6 | 4.117 | 4.216 | +0.099 |
| Tf | 4.337 | 4.635 | +0.298 |
| Tc | 5.692 | 6.145 | +0.453 |

**Boundary artifact unchanged.** No fix, no worsening — as expected for measurement-only Phase 4A.2.

## Stability

- phase3_all_finite: ✓
- phase4a1_all_finite: ✓
- phase3_L_physics_finite: ✓
- phase4a1_L_physics_finite: ✓
- phase3_under_3GB: ✓
- phase4a1_under_3GB: ✓