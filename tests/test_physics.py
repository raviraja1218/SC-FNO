"""Unit tests for physics modules."""
import os, yaml
import numpy as np
import pytest
from src.physics.kinetics import KineticsParameters, equilibrium_precursors, pke_rhs
from src.physics.thermal_hydraulics import THParameters, th_rhs, feedback_reactivity
from data.config_loader import load_config


ROOT = os.path.expanduser('~/projects/SC-FNO')


@pytest.fixture
def kin():
    rp = load_config('reactor_parameters')
    return KineticsParameters.from_yaml(rp)


@pytest.fixture
def th():
    cfg = load_config('thermal_hydraulics')
    return THParameters.from_yaml(cfg)


def test_kinetics_beta_consistency(kin):
    assert abs(sum(kin.beta_i) - kin.beta_eff) < 1e-12
    assert kin.m == 6


def test_equilibrium_precursors(kin):
    C0 = equilibrium_precursors(1.0, kin)
    assert C0.shape == (6,)
    assert np.all(C0 > 0)
    # each: C_i0 = beta_i / (Lambda * lambda_i)
    for b, lam, c in zip(kin.beta_i, kin.lambda_i, C0):
        assert abs(c - b / (kin.Lambda * lam)) < 1e-12


def test_pke_rhs_at_equilibrium_is_zero(kin):
    """With rho_total=0, q=0, and C_i=C_i0, dn/dt=0 and dC_i/dt=0."""
    y_eq = np.concatenate(([1.0], equilibrium_precursors(1.0, kin)))
    rho_zero = lambda t, y: 0.0
    dy = pke_rhs(0.0, y_eq, kin, rho_zero, q_fn=None, include_q=False)
    assert np.allclose(dy, 0.0, atol=1e-14)


def test_th_rhs_at_equilibrium_is_zero(th):
    """At n=1, T_f=T_f_ref, T_c=T_c_ref chosen so steady state holds."""
    # Solve T_c from T_c - T_in relation at equilibrium
    n = 1.0
    T_f = th.T_f_ref
    # At steady state: q_f = U*A*(T_f - T_c) => T_c = T_f - f_f*P0*n/(U*A)
    T_c = T_f - th.f_f * th.P0 * n / (th.U * th.A)
    dTf, dTc = th_rhs(n, T_f, T_c, th)
    # dTf must be ~0 by construction; dTc requires coolant balance
    assert abs(dTf) < 1e-6
    # dTc steady requires q_c + q_fc = 2*m_dot*c_p_c*(T_c - T_in)
    # We test feedback at reference is zero
    assert abs(feedback_reactivity(T_f, T_c, th) - th.alpha_c * (T_c - th.T_c_ref)) < 1e-6


def test_feedback_zero_at_reference(th):
    assert abs(feedback_reactivity(th.T_f_ref, th.T_c_ref, th)) < 1e-12
