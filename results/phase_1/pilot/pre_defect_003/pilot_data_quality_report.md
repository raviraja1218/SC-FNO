# Phase 1.2 Block C — Data Quality Report (v2)

## State variable ranges across 100 pilot scenarios (BDF)

| State | Min | Max |
|-------|-----|-----|
| n | 1.000000e+00 | 1.115053e+03 |
| C_i | 9.069767e-01 | 6.366990e+04 |
| T_f [K] | 900.000 | 134354.564 |
| T_c [K] | 583.000 | 38896.219 |

## Non-negativity
- n >= 0 in all 100 scenarios: 100/100
- C_i >= 0 in all 100 scenarios: 100/100

## Finiteness
- T_f finite: 100/100
- T_c finite: 100/100

## Plausibility
- T_f < 5000 K: 0/100
- T_c < 2000 K: 0/100

## Silent discards
None. Failure register at `pilot_solver_failures.csv`.
