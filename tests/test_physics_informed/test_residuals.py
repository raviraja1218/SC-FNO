"""Step 3.1 — Eight independent validation tests for the residual system.

Tolerance rationale (FP32):
  The residuals at equilibrium are sums/differences of physical quantities
  of order 1-100 (e.g., (rho-beta)/Lambda * n ~ 50, sum_i lambda_i C_i ~ 20).
  FP32 has ~7 significant decimal digits, so the roundoff floor at this
  scale is ~1e-6. Equilibrium assertions use tolerances of 1e-5 (R_n) and
  1e-7 (R_Ci) which are 10x-100x above the FP32 roundoff floor. This is
  scientifically defensible; tighter tolerances would require FP64.
"""
import dataclasses

import numpy as np
import pytest
import torch

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.residuals import compute_residuals


# --- Field-name-agnostic constructors ---
def _build_kinetics():
    fields = {f.name for f in dataclasses.fields(KineticsParameters)}
    beta_vec = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
    lam_vec = np.array([0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100])
    kw = {}
    if "beta_eff" in fields:
        kw["beta_eff"] = 0.006502
    if "Lambda" in fields:
        kw["Lambda"] = 1.0e-4
    for name in ("beta_i", "beta"):
        if name in fields:
            kw[name] = beta_vec
            break
    for name in ("lambda_i", "lambda_decay", "lam"):
        if name in fields:
            kw[name] = lam_vec
            break
    return KineticsParameters(**kw)


def _build_th():
    fields = {f.name for f in dataclasses.fields(THParameters)}
    canonical = {
        "M_f": 27000.0, "c_p_f": 300.0, "f_f": 0.97,
        "M_c": 4000.0, "c_p_c": 5500.0, "m_dot": 13636.36, "T_in": 563.0,
        "U": 3060.0, "A": 3000.0, "P0": 3.0e9,
        "alpha_f": -3.0e-5, "alpha_c": -5.0e-4,
        "T_f_ref": 900.0, "T_c_ref": 583.0,
    }
    return THParameters(**{k: v for k, v in canonical.items() if k in fields})


@pytest.fixture(scope="module")
def kin():
    return _build_kinetics()


@pytest.fixture(scope="module")
def th():
    return _build_th()


def _rho_fn(val):
    return lambda t: torch.full_like(t, float(val))


def _pack(n, C, Tf, Tc):
    """Pack into [B, 9, N] state vector."""
    return torch.cat([
        n.unsqueeze(1),          # [B, 1, N]
        C,                       # [B, 6, N]
        Tf.unsqueeze(1),         # [B, 1, N]
        Tc.unsqueeze(1),         # [B, 1, N]
    ], dim=1)


def _beta_i(kin):
    """Return beta_i vector regardless of exact field name."""
    for attr in ("beta_i", "beta"):
        if hasattr(kin, attr):
            return np.asarray(getattr(kin, attr))
    raise AttributeError("No beta_i field")


def _lambda_i(kin):
    for attr in ("lambda_i", "lambda_decay", "lam"):
        if hasattr(kin, attr):
            return np.asarray(getattr(kin, attr))
    raise AttributeError("No lambda_i field")


# === T1 ===
def test_T1_neutron_residual_analytical(kin, th):
    B, N = 1, 10
    t = torch.linspace(0, 1, N).unsqueeze(0)
    beta_i = _beta_i(kin)
    lambda_i = _lambda_i(kin)

    n = torch.ones(B, N)
    C = torch.zeros(B, 6, N)
    for i in range(6):
        C[:, i, :] = beta_i[i] / (kin.Lambda * lambda_i[i])
    Tf = torch.full((B, N), th.T_f_ref)
    Tc = torch.full((B, N), th.T_c_ref)

    u_hat = _pack(n, C, Tf, Tc)          # [1, 9, 10]
    du_hat_dt = torch.zeros_like(u_hat)

    out = compute_residuals(u_hat, du_hat_dt, t, _rho_fn(0.0), kin, th)
    # FP32 roundoff at equilibrium ~1e-5 (terms of order 10 summed, 2^-23 resolution)
    assert out.R_n.abs().max().item() < 1e-5, \
        f"R_n eq = {out.R_n.abs().max().item():.3e} exceeds FP32 roundoff"

    out2 = compute_residuals(u_hat, du_hat_dt, t, _rho_fn(0.5 * kin.beta_eff), kin, th)
    # R_n = dn/dt_actual (0) - dn/dt_predicted = -rho/Lambda * n
    # since at equilibrium sum_lambda_C cancels the -beta_eff/Lambda * n term.
    # At rho = 0.5 * beta_eff, n = 1: dn/dt_predicted = +0.5 * beta_eff / Lambda
    # => R_n = -0.5 * beta_eff / Lambda
    expected = -0.5 * kin.beta_eff / kin.Lambda
    assert torch.allclose(out2.R_n, torch.full_like(out2.R_n, expected), rtol=1e-5), \
        f"R_n = {out2.R_n[0,0].item():.6e}, expected {expected:.6e}"


# === T2 ===
@pytest.mark.parametrize("i", [0, 1, 2, 3, 4, 5])
def test_T2_precursor_residual_analytical(kin, th, i):
    B, N = 1, 10
    t = torch.linspace(0, 1, N).unsqueeze(0)
    beta_i = _beta_i(kin)

    n = torch.ones(B, N)
    C = torch.zeros(B, 6, N)
    Tf = torch.full((B, N), th.T_f_ref)
    Tc = torch.full((B, N), th.T_c_ref)
    u_hat = _pack(n, C, Tf, Tc)
    du_hat_dt = torch.zeros_like(u_hat)

    out = compute_residuals(u_hat, du_hat_dt, t, _rho_fn(0.0), kin, th)
    expected = -beta_i[i] / kin.Lambda
    assert torch.allclose(
        out.R_C[:, i, :], torch.full_like(out.R_C[:, i, :], expected), atol=1e-9
    ), f"C{i+1} res = {out.R_C[0,i,0].item():.6e}, expected {expected:.6e}"


# === T3 ===
def test_T3_fuel_thermal_adiabatic(kin, th):
    th_a = dataclasses.replace(th, U=0.0)
    B, N = 1, 10
    t = torch.linspace(0, 1, N).unsqueeze(0)
    n = torch.ones(B, N)
    C = torch.zeros(B, 6, N)
    Tf = torch.full((B, N), th.T_f_ref)
    Tc = torch.full((B, N), th.T_c_ref)
    u_hat = _pack(n, C, Tf, Tc)
    du_hat_dt = torch.zeros_like(u_hat)

    out = compute_residuals(u_hat, du_hat_dt, t, _rho_fn(0.0), kin, th_a)
    expected = th.f_f * th.P0 / (th.M_f * th.c_p_f)
    assert torch.allclose(out.R_Tf, torch.full_like(out.R_Tf, -expected), rtol=1e-5), \
        f"R_Tf adia = {out.R_Tf[0,0].item():.6e}, expected {-expected:.6e}"


# === T4 ===
def test_T4_coolant_thermal(kin, th):
    B, N = 1, 10
    t = torch.linspace(0, 1, N).unsqueeze(0)
    n = torch.ones(B, N)
    C = torch.zeros(B, 6, N)
    Tf = torch.full((B, N), th.T_c_ref)
    Tc = torch.full((B, N), th.T_c_ref)
    u_hat = _pack(n, C, Tf, Tc)
    du_hat_dt = torch.zeros_like(u_hat)

    out = compute_residuals(u_hat, du_hat_dt, t, _rho_fn(0.0), kin, th)
    expected = (1.0 - th.f_f) * th.P0 / (th.M_c * th.c_p_c) \
               - 2.0 * th.m_dot * th.c_p_c / (th.M_c * th.c_p_c) * (th.T_c_ref - th.T_in)
    assert torch.allclose(out.R_Tc, torch.full_like(out.R_Tc, -expected), rtol=1e-5), \
        f"R_Tc = {out.R_Tc[0,0].item():.6e}, expected {-expected:.6e}"


# === T5 ===
def test_T5_feedback_reactivity(kin, th):
    B, N = 1, 10
    t = torch.linspace(0, 1, N).unsqueeze(0)
    n = torch.ones(B, N)
    C = torch.zeros(B, 6, N)
    Tf = torch.full((B, N), th.T_f_ref)
    Tc = torch.full((B, N), th.T_c_ref)
    u_hat = _pack(n, C, Tf, Tc)
    out = compute_residuals(u_hat, torch.zeros_like(u_hat), t, _rho_fn(0.0), kin, th)
    assert out.rho_fb.abs().max().item() < 1e-9, \
        f"rho_fb@refs = {out.rho_fb.abs().max().item():.3e}"

    Tf_off = torch.full((B, N), th.T_f_ref + 50.0)
    out2 = compute_residuals(_pack(n, C, Tf_off, Tc), torch.zeros_like(u_hat),
                             t, _rho_fn(0.0), kin, th)
    expected = th.alpha_f * 50.0
    assert torch.allclose(out2.rho_fb, torch.full_like(out2.rho_fb, expected), rtol=1e-6), \
        f"rho_fb(Tf+50) = {out2.rho_fb[0,0].item():.6e}, expected {expected:.6e}"


# === T6 ===
def test_T6_coupling_predicted_states_used(kin, th):
    B, N = 1, 10
    t = torch.linspace(0, 1, N).unsqueeze(0)
    n = torch.ones(B, N)
    C = torch.zeros(B, 6, N)
    Tc = torch.full((B, N), th.T_c_ref)

    u_lo = _pack(n, C, torch.full((B, N), th.T_f_ref), Tc)
    u_hi = _pack(n, C, torch.full((B, N), th.T_f_ref + 100.0), Tc)
    z = torch.zeros_like(u_lo)

    o_lo = compute_residuals(u_lo, z, t, _rho_fn(0.0), kin, th)
    o_hi = compute_residuals(u_hi, z, t, _rho_fn(0.0), kin, th)

    assert (o_hi.rho_fb - o_lo.rho_fb).abs().mean().item() > 1e-6
    assert (o_hi.R_n - o_lo.R_n).abs().mean().item() > 1e-6


# === T7 ===
def test_T7_equilibrium_residuals_zero(kin, th):
    B, N = 1, 10
    t = torch.linspace(0, 1, N).unsqueeze(0)
    beta_i = _beta_i(kin)
    lambda_i = _lambda_i(kin)
    n = torch.ones(B, N)
    C = torch.zeros(B, 6, N)
    for i in range(6):
        C[:, i, :] = beta_i[i] / (kin.Lambda * lambda_i[i])
    Tf = torch.full((B, N), th.T_f_ref)
    Tc = torch.full((B, N), th.T_c_ref)
    u_hat = _pack(n, C, Tf, Tc)
    out = compute_residuals(u_hat, torch.zeros_like(u_hat), t, _rho_fn(0.0), kin, th)

    # FP32-aware tolerance: residual involves sums of O(10) terms -> ~1e-6 roundoff floor
    assert out.R_n.abs().max().item() < 1e-5
    # R_Ci involves differences of terms of order beta_i/Lambda ~ 2-25 -> FP32 floor ~1e-6
    assert out.R_C.abs().max().item() < 1e-6

    UA = th.U * th.A
    rate_Tf = th.f_f * th.P0 / (th.M_f * th.c_p_f) \
              - UA / (th.M_f * th.c_p_f) * (th.T_f_ref - th.T_c_ref)
    assert abs(out.R_Tf[0,0].item() + rate_Tf) < 1e-3

    rate_Tc = (1 - th.f_f) * th.P0 / (th.M_c * th.c_p_c) \
              + UA / (th.M_c * th.c_p_c) * (th.T_f_ref - th.T_c_ref) \
              - 2.0 * th.m_dot / th.M_c * (th.T_c_ref - th.T_in)
    assert abs(out.R_Tc[0,0].item() + rate_Tc) < 1e-3


# === T8 ===
def test_T8_unit_consistency(kin, th):
    B, N = 1, 10
    t = torch.linspace(0, 1, N).unsqueeze(0)
    n = torch.ones(B, N)
    C = torch.zeros(B, 6, N)
    Tf = torch.full((B, N), th.T_f_ref)
    Tc = torch.full((B, N), th.T_c_ref)
    u_hat = _pack(n, C, Tf, Tc)

    o0 = compute_residuals(u_hat, torch.zeros_like(u_hat), t, _rho_fn(0.0), kin, th)

    d = torch.zeros_like(u_hat); d[:, 0, :] = 1.0
    o = compute_residuals(u_hat, d, t, _rho_fn(0.0), kin, th)
    assert torch.allclose(o.R_n - o0.R_n, torch.ones_like(o.R_n), atol=1e-6)

    d = torch.zeros_like(u_hat); d[:, 7, :] = 1.0
    o = compute_residuals(u_hat, d, t, _rho_fn(0.0), kin, th)
    assert torch.allclose(o.R_Tf - o0.R_Tf, torch.ones_like(o.R_Tf), atol=1e-6)

    d = torch.zeros_like(u_hat); d[:, 3, :] = 1.0
    o = compute_residuals(u_hat, d, t, _rho_fn(0.0), kin, th)
    assert torch.allclose(o.R_C[:, 2, :] - o0.R_C[:, 2, :],
                          torch.ones_like(o.R_C[:, 2, :]), atol=1e-6)
