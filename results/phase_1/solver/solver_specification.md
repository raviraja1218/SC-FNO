# Step 1.1 — Solver Specification

## State vector (9 components)
u(t) = [n(t), C1(t), C2(t), C3(t), C4(t), C5(t), C6(t), Tf(t), Tc(t)]^T

## Governing equations
dn/dt     = (rho_total - beta_eff)/Lambda * n + sum_i lambda_i C_i + q(t)
dC_i/dt   = beta_i/Lambda * n - lambda_i C_i            (i = 1..6)
M_f c_p,f dT_f/dt = f_f P0 n - UA (T_f - T_c)
M_c c_p,c dT_c/dt = (1-f_f) P0 n + UA (T_f - T_c) - 2 m_dot c_p,c (T_c - T_in)
rho_total(t) = rho_ext(t) + alpha_f (T_f - T_f_ref) + alpha_c (T_c - T_c_ref)

## Locked parameters (v3 supersession)
- beta_eff = 0.006502  (= sum(beta_i); Phase 0 correction SC-FNO-P0-DEFECT-001)
- Lambda   = 1.0e-4 s
- beta_i   = [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273]
- lambda_i = [0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100] 1/s
- q(t) INCLUDED per Phase 0 lock; default q = 0

## Numerical integration
Primary:    SciPy solve_ivp BDF, rtol=1e-8, atol=1e-10 (adaptive internal)
Cross-check: Radau, same tolerances
Output grid: 200 points, t_end default = 10 s

## Analytical acceptance tests (all PASS)

| Test | Configuration | Tolerance | Result |
|------|---------------|-----------|--------|
| T1 Kinetic equilibrium | rho=0, q=0, temps frozen | \\|dn/dt\\|, max\\|dC_i/dt\\| < 1e-10 | PASS |
| T2 Prompt jump (exact) | rho=0.5 beta_eff, dC_i/dt=0, exact ODE solution | rel L2 < 1e-6 | PASS |
| T3 Inhour (frozen feedback) | rho=1e-3, precursors free, asymptotic period | rel err < 1% | PASS |
| T4 Adiabatic heating | U=0, constant n, analytic dT_f/dt | rel err < 0.1% | PASS |

## T2 mathematical formulation (corrected — SC-FNO-REV-001)

With precursors frozen (dC_i/dt = 0), the neutron equation is

    dn/dt = a*n + S

where
    a = (rho - beta_eff)/Lambda          (constant)
    S = sum_i lambda_i*C_i(0)            (constant, generally NONZERO)

The exact solution is

    n(t) = (n0 + S/a)*exp(a*t) - S/a

This is NOT n0*exp(a*t) unless S = 0. With equilibrium initial precursors,
S = beta_eff*n0/Lambda ≠ 0, so the S/a correction is essential.

The naive formula n(t) = n0*exp(a*t) is a singular-perturbation limit that
holds only after the delayed-neutron source is negligible — which it is not
at t=0 for the standard equilibrium initial condition. The corrected test
compares the solver output to the exact analytical n(t) above.

## Test-physics notes
- T2 verifies the solver against the exact frozen-precursor ODE solution.
  The naive exponential n0*exp((rho-beta_eff)/Lambda*t) was the original
  formula; reviewer SC-FNO-REV-001 correctly identified that it omits the
  delayed-neutron source term S/a and is not the exact solution.
- The algebraic prompt-jump ratio beta_eff/(beta_eff - rho) is verified
  independently as a parameter/formula check (T2 bonus assertion).
- T3 uses frozen temperature feedback (kinetics-only) so the inhour relation
  is exact. Full coupled thermal-feedback integration would be governed by a
  different (coupled) eigenvalue.
