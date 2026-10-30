# Phase 1.2 Block C — Data Quality Report (v2, post SC-FNO-P0-DEFECT-003)

## State variable ranges across 100 pilot scenarios (corrected physics)

| State | Min | Max |
|-------|-----|-----|
| n | 2.561452e-01 | 1.041059e+00 |
| C_i | 2.864542e-01 | 4.670296e+02 |
| T_f [K] | 674.418 | 900.000 |
| T_c [K] | 583.000 | 621.623 |

## Non-negativity
- n >= 0 in all 100 scenarios: 100/100
- C_i >= 0 in all 100 scenarios: 100/100

## Finiteness
- T_f finite: 100/100
- T_c finite: 100/100

## Plausibility
- T_f < 5000 K: 100/100
- T_c < 2000 K: 100/100

**Note:** The pre-fix trajectory produced T_f up to 132,859 K — a direct consequence
of the coupled_rhs feedback bug (SC-FNO-P0-DEFECT-003). Post-fix, T_f stays within
the initial value (no spurious heating) because pilot step insertions of amplitude
≤0.5 β_eff produce small reactivity transients that do not thermally challenge the
core in 10 s. Longer-horizon scenarios in Step 1.3 may produce larger ΔT and will be
re-validated at that stage.

## Silent discards
None. Failure register at `pilot_solver_failures.csv`.
