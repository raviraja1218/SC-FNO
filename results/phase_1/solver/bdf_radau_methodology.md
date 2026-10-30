# BDF / Radau Cross-Check Methodology

## Purpose
Establish that numerical integration is method-independent by comparing BDF
(variable-order, variable-step implicit multistep) with Radau (implicit
Runge–Kutta) at identical tolerances (rtol=1e-8, atol=1e-10).

## Reference configuration
BDF is the locked primary method (Phase 0). Radau serves as an independent
stiff solver for cross-validation during the Step 1.2 pilot.

## Acceptance thresholds (defined BEFORE inspecting pilot results)
- n, C_i:   max|Δ| ≤ 1e-6 relative to max|value|
- T_f, T_c: max|Δ| ≤ 1e-6 K
- Relative L2: ≤ 1e-5 per state

## Failure handling
Radau/BDF disagreement above threshold triggers investigation (tighten
tolerances, check stiffness, inspect state ordering) — NOT silent removal
of the scenario.

## Phase 1.1 verification
The four analytical acceptance tests (T1–T4) validate the BDF implementation
against exact analytical limits BEFORE pilot-scale BDF/Radau comparison
begins in Step 1.2.
