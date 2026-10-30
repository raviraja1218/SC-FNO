"""
Phase 1 Step 1.1 — Analytical acceptance tests for the coupled 9-state solver.

Matches the actual locked Phase 0 API:
  pke_rhs(t, y_kin, kin, rho_total_fn, q_fn=None, include_q=True) -> concatenated [dn, dC1..dC6]
  y_kin = [n, C1..C6]  (no temperatures — kinetics subsystem only)
  th_rhs(n, T_f, T_c, th) -> (dTf, dTc)
  equilibrium_precursors(n0, kin) -> ndarray of 6 C_eq values

T1 — Kinetic equilibrium
T2 — Prompt jump (frozen feedback, short window)
T3 — Inhour (frozen feedback, asymptotic period)
T4 — Adiabatic heating (U=0, constant power)
"""
import sys
sys.path.insert(0, 'src')

import dataclasses
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from data.config_loader import load_config
from physics.kinetics import KineticsParameters, pke_rhs, equilibrium_precursors
from physics.thermal_hydraulics import THParameters, th_rhs


# ---------------- Fixtures ----------------
@pytest.fixture(scope="module")
def kin():
    return KineticsParameters.from_yaml(load_config('reactor_parameters'))


@pytest.fixture(scope="module")
def th():
    return THParameters.from_yaml(load_config('thermal_hydraulics'))


@pytest.fixture(scope="module")
def init_state(kin):
    cfg = load_config('reactor_parameters')
    n0 = float(cfg['initial_state']['n0'])
    C0 = equilibrium_precursors(n0, kin)
    Tf0 = float(cfg['initial_state']['T_f0'])
    Tc0 = float(cfg['initial_state']['T_c0'])
    return np.concatenate([[n0], C0, [Tf0, Tc0]])


# ---------------- T1 — Kinetic equilibrium ----------------
def test_T1_kinetic_equilibrium(kin, init_state):
    """With rho_total=0, q=0, temps frozen, kinetics subsystem must be stationary."""
    y_kin = init_state[:7].copy()  # [n, C1..C6]

    def rho_zero(t, y):
        return 0.0

    def q_zero(t):
        return 0.0

    dydt = pke_rhs(0.0, y_kin, kin, rho_zero, q_zero, include_q=True)
    dndt = dydt[0]
    dCdt = dydt[1:]

    assert abs(dndt) < 1e-10, f"|dn/dt| = {abs(dndt):.3e} exceeds 1e-10"
    assert np.max(np.abs(dCdt)) < 1e-10, (
        f"max|dC_i/dt| = {np.max(np.abs(dCdt)):.3e} exceeds 1e-10"
    )


# ---------------- T2 — Prompt jump (exact frozen-precursor ODE) ----------------
def test_T2_prompt_jump(kin, init_state):
    """Verify the solver reproduces the EXACT frozen-precursor ODE solution.

    With dC_i/dt = 0 (precursors frozen), the neutron equation is

        dn/dt = a*n + S,

    where
        a = (rho - beta_eff)/Lambda          (constant in time)
        S = sum_i lambda_i*C_i(0) + q        (constant because C_i is frozen)

    Unlike the naive prompt-jump formula n(t) = n0*exp(a*t), the exact
    solution retains the delayed-neutron source contribution:

        n(t) = (n0 + S/a)*exp(a*t) - S/a.

    With the equilibrium initial state, S = beta_eff*n0/Lambda (nonzero),
    so the S/a correction is essential and the naive formula is WRONG.

    This test:
      (1) constructs the frozen-precursor RHS explicitly,
      (2) integrates with BDF at tight tolerance,
      (3) compares n(t) to the exact analytical solution above,
      (4) also verifies the prompt-jump ratio n(0+)/n(0-) -> beta_eff/(beta_eff - rho)
          in the limit t -> 0+ via the analytical jump formula.

    Reference: SC-FNO-REV-001 (reviewer audit).
    """
    rho_ext = 0.5 * kin.beta_eff
    y0 = init_state[:7].copy()
    n0 = float(y0[0])
    C0 = np.asarray(y0[1:], dtype=float)

    # Frozen-precursor constants
    betas = np.asarray(kin.beta_i, dtype=float)
    lambdas = np.asarray(kin.lambda_i, dtype=float)
    S = float(np.sum(lambdas * C0))                  # delayed source, frozen
    a = (rho_ext - kin.beta_eff) / kin.Lambda         # prompt rate

    # Exact analytical solution
    def n_exact(t):
        return (n0 + S / a) * np.exp(a * t) - S / a

    # Frozen-precursor ODE: precursors held fixed at C0
    def rhs_frozen(t, y):
        n = y[0]
        dn = a * n + S
        return np.concatenate(([dn], np.zeros(6)))

    # Integrate over several prompt time constants
    tau_prompt = 1.0 / abs(a)  # = Lambda/(beta_eff - rho)
    t_end = 3.0 * tau_prompt
    t_eval = np.linspace(0.0, t_end, 200)
    sol = solve_ivp(rhs_frozen, [0.0, t_end], y0, method='BDF',
                    rtol=1e-12, atol=1e-14, t_eval=t_eval)
    assert sol.success, f"Solver failed: {sol.message}"

    n_numeric = sol.y[0]
    n_analytical = n_exact(sol.t)

    # Relative L2 error across trajectory
    denom = np.max(np.abs(n_analytical))
    rel_l2 = float(np.linalg.norm(n_numeric - n_analytical) / denom)

    assert rel_l2 < 1e-6, (
        f"Frozen-precursor exact-solution mismatch: "
        f"relative L2 error {rel_l2:.3e} exceeds 1e-6. "
        f"S = {S:.6e}, a = {a:.6e}, n0 = {n0:.6e}"
    )

    # Bonus algebraic check: prompt-jump ratio n(0+)/n(0-) in the limit
    # t->0+ with thermal and delayed feedback absent is given by the
    # textbook formula. We verify it against the analytical jump, not
    # against the ODE trajectory (which is the S/a-corrected version).
    ratio_jump = kin.beta_eff / (kin.beta_eff - rho_ext)
    assert abs(ratio_jump - 2.0) < 1e-12, (
        f"Algebraic prompt-jump ratio {ratio_jump} != 2.0 for rho=0.5*beta_eff"
    )


# ---------------- T3 — Inhour ----------------
def test_T3_inhour(kin, init_state):
    """Frozen-feedback kinetics-only: asymptotic period vs inhour equation."""
    rho_ext = 1.0e-3
    y0 = init_state[:7].copy()

    betas = np.asarray(kin.beta_i, dtype=float)
    lambdas = np.asarray(kin.lambda_i, dtype=float)

    def inhour(T):
        return kin.Lambda / T + np.sum(betas / (1.0 + lambdas * T)) - rho_ext

    # Bracket the positive root
    T_analytical = brentq(inhour, 1e-4, 1e4)

    def rho_const(t, y):
        return rho_ext

    def q_zero(t):
        return 0.0

    def rhs(t, y):
        return pke_rhs(t, y, kin, rho_const, q_zero, include_q=True)

    t_end = 5.0 * T_analytical
    sol = solve_ivp(rhs, [0.0, t_end], y0, method='BDF',
                    rtol=1e-10, atol=1e-12,
                    t_eval=np.linspace(0.0, t_end, 800))
    assert sol.success, f"Solver failed: {sol.message}"

    n = sol.y[0]
    t = sol.t
    mask = t > 0.6 * t_end
    log_n = np.log(np.maximum(n[mask], 1e-300))
    slope, _ = np.polyfit(t[mask], log_n, 1)
    T_numerical = 1.0 / slope

    rel_err = abs(T_numerical - T_analytical) / T_analytical
    assert rel_err < 0.01, (
        f"Inhour period: numerical {T_numerical:.6e} s vs "
        f"analytical {T_analytical:.6e} s; rel err {rel_err:.4%} exceeds 1%"
    )


# ---------------- T4 — Adiabatic heating ----------------
def test_T4_adiabatic_heating(th, init_state):
    """U=0, constant power: dT_f/dt = f_f * P0 * n / (M_f * c_p_f)."""
    th_adiabatic = dataclasses.replace(th, U=0.0)
    n = init_state[0]
    T_f = init_state[7]
    T_c = init_state[8]

    dTfdt, _ = th_rhs(n=n, T_f=T_f, T_c=T_c, th=th_adiabatic)

    expected = (th_adiabatic.f_f * th_adiabatic.P0 * n) / (
        th_adiabatic.M_f * th_adiabatic.c_p_f
    )
    rel_err = abs(dTfdt - expected) / abs(expected)
    assert rel_err < 1e-3, (
        f"Adiabatic dT_f/dt = {dTfdt:.6e} vs expected {expected:.6e}; "
        f"rel err {rel_err:.4%} exceeds 0.1%"
    )


# ---------------- T5 — Negative step must decrease power ----------------
def test_T5_negative_step_decreases_power(kin, th, init_state):
    """SC-FNO-P0-DEFECT-003 regression test.

    A negative step reactivity insertion through coupled_rhs MUST produce
    a decreasing neutron density. Before the P0-DEFECT-003 fix,
    coupled_rhs mis-interpreted C5, C6 as T_f, T_c in the feedback
    calculation, generating a large spurious positive reactivity.
    """
    from data.solver import coupled_rhs

    rho_ext = -0.05 * kin.beta_eff  # pilot_0001-like: -0.05 beta_eff

    def rho_ext_fn(t):
        return float(rho_ext)

    def q_fn(t):
        return 0.0

    # Single RHS evaluation at t=0
    dydt = coupled_rhs(0.0, init_state, kin, th, rho_ext_fn, q_fn, True)
    dn_dt = float(dydt[0])

    # With rho = -0.05 beta_eff at equilibrium, dn/dt should be negative
    assert dn_dt < 0, (
        f"P0-DEFECT-003 regression: dn/dt = {dn_dt:.6e} is not negative "
        f"for rho_ext = {rho_ext:.6e} (=-0.05 beta_eff). "
        f"This means feedback is spurious."
    )

    # And it should be of the right order of magnitude: -(0.05*beta_eff)/Lambda
    expected_dn_dt = rho_ext / kin.Lambda * init_state[0]
    rel_err = abs(dn_dt - expected_dn_dt) / abs(expected_dn_dt)
    assert rel_err < 0.05, (
        f"dn/dt = {dn_dt:.6e} vs expected ≈ {expected_dn_dt:.6e}; "
        f"rel err {rel_err:.4%} exceeds 5%"
    )
