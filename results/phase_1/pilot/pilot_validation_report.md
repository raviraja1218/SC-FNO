# Phase 1.2 Block C — Pilot Validation Report (v2, post SC-FNO-P0-DEFECT-003)

## Corrections applied
- **SC-FNO-P1-DEFECT-002**: thermal absolute threshold raised from 1e-6 K (v1, unachievable)
  to 1e-2 K (v2, calibrated). See `results/phase_1/solver/bdf_radau_methodology.v2.md`.
- **SC-FNO-P0-DEFECT-003**: `coupled_rhs` fixed to capture Tf, Tc from full 9-state y
  before calling pke_rhs (previously mis-interpreted C5, C6 as temperatures,
  producing huge spurious positive feedback). See `results/phase_1/pilot/pre_defect_003/`
  for preserved pre-fix evidence. Regression test `test_T5_negative_step_decreases_power`
  added to `tests/test_solver.py`.

## BDF/Radau comparison (v2, post-fix)
- Scenarios compared: 100
- Comparison passed: 100/100
- Physical validation passed: 100/100
- Failures: 0
- Overall: True

## Thresholds (v2)
- Kinetics (n, C_i): max_abs_diff <= 1e-6 × max(|value|)
- Thermal (T_f, T_c): max_abs_diff <= 1e-2 K
- Relative L2 per state: <= 1e-5

## Per-state statistics (post-fix)
| State | Scenarios | Passed | Max abs diff | Max rel L2 |
|-------|-----------|--------|--------------|------------|
| n | 100 | 100 | 7.847e-07 | 9.673e-07 |
| C1 | 100 | 100 | 8.174e-08 | 3.641e-09 |
| C2 | 100 | 100 | 5.390e-07 | 9.058e-09 |
| C3 | 100 | 100 | 1.447e-06 | 1.150e-07 |
| C4 | 100 | 100 | 6.605e-06 | 4.969e-07 |
| C5 | 100 | 100 | 2.053e-07 | 1.260e-07 |
| C6 | 100 | 100 | 1.005e-07 | 4.031e-07 |
| Tf | 100 | 100 | 1.150e-05 | 5.036e-08 |
| Tc | 100 | 100 | 9.178e-06 | 4.306e-08 |

## Physical validation
- Trajectories passing all physical checks: 100/100

## Total comparison wall time
43.68 s

## Failure register
- Total failures: 0

No failures — all 100 scenarios pass BDF/Radau comparison and physical validation.
