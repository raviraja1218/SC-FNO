# Step 1.4 Block A — Integrity Verification Report

## Status
- Scenarios checked: 2000
- Passed: 2000
- Failed: 0
- Wall time: 3.3 s

## Checks performed per scenario
1. All 5 required files present
2. Arrays load cleanly
3. forcing shape == (200,)
4. trajectory shape == (200, 9)
5. time shape == (200,)
6. time strictly increasing, [0, 10]
7. No NaN or Inf in any array
8. Initial state: n0=1, Tf0=900 K, Tc0=583 K, C_i0=equilibrium
9. metadata.json present and consistent (scenario_id matches dir name)
10. solver_diagnostics.json shows success=true
11. Family consistent with Step 1.3 manifest
12. Scenario ID present in Step 1.3 manifest

## Family distribution (validated scenarios)
- step: 500
- ramp: 500
- sinusoidal: 500
- grf: 500

## Failures
None — all 2,000 scenarios pass all integrity checks.
