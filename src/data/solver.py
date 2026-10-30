"""Coupled PKE + thermal-hydraulic transient solver via SciPy solve_ivp.

State vector: [n, C_1, ..., C_6, T_f, T_c]  (length 9)
Primary method: BDF (stiff). Cross-check: Radau (stiff, independent).

Phase 0.5 locked: q(t) is INCLUDED in the n-equation (default q=0).
Phase 1.1: equilibrium initial conditions; no hardcoded C_i0.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp

from src.physics.kinetics import (
    KineticsParameters, equilibrium_precursors, pke_rhs
)
from src.physics.thermal_hydraulics import THParameters, th_rhs
from src.physics.reactivity import make_rho_total


@dataclass
class SolverConfig:
    method: str = "BDF"        # or "Radau"
    rtol: float = 1e-8
    atol: float = 1e-10
    t_start: float = 0.0
    t_end: float = 10.0
    n_time_points: int = 200


def initial_state(kin: KineticsParameters, th: THParameters) -> np.ndarray:
    """Equilibrium initial state: n=1, C_i=C_i0, T_f=T_f_ref, T_c=T_c_ref."""
    n0 = 1.0
    C0 = equilibrium_precursors(n0, kin)
    return np.concatenate(([n0], C0, [th.T_f_ref, th.T_c_ref]))


def coupled_rhs(t, y, kin, th, rho_ext_fn, q_fn, include_q):
    """Full coupled RHS: [dn/dt, dC/dt, dT_f/dt, dT_c/dt]."""
    n = float(y[0])
    rho_total_fn = make_rho_total(rho_ext_fn, th)
    d_kin = pke_rhs(t, y[:kin.m + 1], kin, rho_total_fn, q_fn, include_q)
    T_f, T_c = float(y[-2]), float(y[-1])
    dT_f, dT_c = th_rhs(n, T_f, T_c, th)
    return np.concatenate((d_kin, [dT_f, dT_c]))


def solve_transient(rho_ext_fn: Callable[[float], float],
                    kin: KineticsParameters,
                    th: THParameters,
                    cfg: SolverConfig,
                    q_fn: Callable[[float], float] | None = None,
                    include_q: bool = True) -> dict:
    """Run coupled transient. Returns dict with t, y, success, solver metadata."""
    y0 = initial_state(kin, th)
    t_eval = np.linspace(cfg.t_start, cfg.t_end, cfg.n_time_points)
    if q_fn is None:
        q_fn = lambda t: 0.0
    sol = solve_ivp(
        fun=lambda t, y: coupled_rhs(t, y, kin, th, rho_ext_fn, q_fn, include_q),
        t_span=(cfg.t_start, cfg.t_end),
        y0=y0,
        method=cfg.method,
        t_eval=t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    return {
        "t": sol.t,
        "y": sol.y,               # shape (9, Nt)
        "success": bool(sol.success),
        "message": sol.message,
        "nfev": int(sol.nfev),
        "method": cfg.method,
        "rtol": cfg.rtol,
        "atol": cfg.atol,
    }
