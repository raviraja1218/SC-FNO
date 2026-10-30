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
| T1 Kinetic equilibrium | rho=0, q=0, temps frozen | |dn/dt|, max|dC_i/dt| < 1e-10 | PASS |
| T2 Prompt jump (frozen precursors) | rho=0.5 beta_eff, dC_i/dt=0, exact exp comparison | rel err < 1e-6 | PASS |
| T3 Inhour (frozen feedback) | rho=1e-3, precursors free, asymptotic period | rel err < 1% | PASS |
| T4 Adiabatic heating | U=0, constant n, analytic dT_f/dt | rel err < 0.1% | PASS |

## Test-physics notes
- T2 verifies the solver reproduces the exact analytical prompt response
  n(t) = n0 * exp((rho - beta_eff)/Lambda * t) under frozen-precursor
  conditions. This is the correct formulation of the prompt-jump limit:
  the ratio beta_eff/(beta_eff - rho) is the initial-condition transformation,
  not the trajectory of the full PKE with evolving precursors.
- T3 uses frozen temperature feedback (kinetics-only) so the inhour relation
  is exact. Full coupled thermal-feedback integration would be governed by a
  different (coupled) eigenvalue.
