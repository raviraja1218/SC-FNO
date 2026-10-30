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
    """Full coupled RHS: [dn/dt, dC/dt, dT_f/dt, dT_c/dt].

    SC-FNO-P0-DEFECT-003 fix: the previous implementation called
    make_rho_total(rho_ext_fn, th) which returns a rho_total(t, y)
    closure that reads y[-2], y[-1] as T_f, T_c. But pke_rhs passes
    y[:kin.m+1] (=[n, C1..C6]) to that closure, so y[-2]=C5 and y[-1]=C6
    were being misinterpreted as temperatures, producing huge spurious
    positive feedback reactivity and physically impossible power/temperature
    growth.

    Fix: capture T_f, T_c from the full y here, then define rho_total_fn
    that ignores its second argument and uses the captured temperatures.
    """
    n = float(y[0])
    T_f, T_c = float(y[-2]), float(y[-1])

    def rho_total_fn(t_inner, _y_kin):
        return float(rho_ext_fn(t_inner)) + feedback_reactivity(T_f, T_c, th)

    d_kin = pke_rhs(t, y[:kin.m + 1], kin, rho_total_fn, q_fn, include_q)
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


# ============================================================
# Phase 1.2 Block B — batch-scenario wrapper
# ============================================================
from dataclasses import dataclass, field
from typing import Optional
import time as _time
from physics.reactivity import feedback_reactivity


@dataclass
class PilotScenarioResult:
    """Container for one scenario's solver output and diagnostics."""
    scenario_id: str
    family: str
    success: bool
    message: str
    t: np.ndarray                    # (n_t,)
    rho_ext: np.ndarray              # (n_t,)
    trajectory: np.ndarray           # (n_t, 9) -> [n, C1..C6, Tf, Tc]
    nfev: int
    njev: int
    nlu: int
    wall_time_s: float
    rtol: float
    atol: float
    method: str
    q_t_default: float
    include_q: bool
    config_versions: dict = field(default_factory=dict)


def solve_one_scenario(spec,
                       t_grid: np.ndarray,
                       kin: KineticsParameters,
                       th: THParameters,
                       rho_ext_fn: Callable[[float], float],
                       solver_method: str = "BDF",
                       rtol: float = 1e-8,
                       atol: float = 1e-10,
                       q_t_default: float = 0.0,
                       include_q: bool = True,
                       config_versions: Optional[dict] = None) -> PilotScenarioResult:
    """Solve one pilot scenario end-to-end. Purely additive wrapper around
    solve_transient / coupled_rhs; does not change their behaviour."""
    from scipy.integrate import solve_ivp

    y0 = initial_state(kin, th)

    def q_fn(t):
        return q_t_default

    def rhs(t, y):
        return coupled_rhs(t, y, kin, th, rho_ext_fn, q_fn, include_q)

    t0 = _time.perf_counter()
    wall = 0.0  # initialize so the exception path can always reference it
    try:
        sol = solve_ivp(rhs, [float(t_grid[0]), float(t_grid[-1])], y0,
                        method=solver_method,
                        t_eval=t_grid,
                        rtol=rtol, atol=atol)
        wall = _time.perf_counter() - t0
    except Exception as exc:  # pragma: no cover (diagnostic path)
        wall = _time.perf_counter() - t0
        return PilotScenarioResult(
            scenario_id=getattr(spec, "scenario_id", "unknown"),
            family=getattr(spec, "family", "unknown"),
            success=False,
            message=f"Integration raised: {exc!r}",
            t=t_grid.copy(),
            rho_ext=np.asarray([rho_ext_fn(float(tt)) for tt in t_grid]),
            trajectory=np.full((len(t_grid), 9), np.nan),
            nfev=-1, njev=-1, nlu=-1,
            wall_time_s=wall,
            rtol=rtol, atol=atol,
            method=solver_method,
            q_t_default=q_t_default,
            include_q=include_q,
            config_versions=config_versions or {},
        )

    if not sol.success or sol.y.shape[1] != len(t_grid):
        return PilotScenarioResult(
            scenario_id=spec.scenario_id,
            family=spec.family,
            success=False,
            message=f"solver failed: {sol.message}; got {sol.y.shape[1]}/{len(t_grid)} points",
            t=t_grid.copy(),
            rho_ext=np.asarray([rho_ext_fn(float(tt)) for tt in t_grid]),
            trajectory=np.full((len(t_grid), 9), np.nan),
            nfev=getattr(sol, "nfev", -1),
            njev=getattr(sol, "njev", -1),
            nlu=getattr(sol, "nlu", -1),
            wall_time_s=wall,
            rtol=rtol, atol=atol,
            method=solver_method,
            q_t_default=q_t_default,
            include_q=include_q,
            config_versions=config_versions or {},
        )

    traj = sol.y.T  # (n_t, 9)
    rho_arr = np.asarray([rho_ext_fn(float(tt)) for tt in t_grid])

    return PilotScenarioResult(
        scenario_id=spec.scenario_id,
        family=spec.family,
        success=True,
        message=sol.message,
        t=t_grid.copy(),
        rho_ext=rho_arr,
        trajectory=traj,
        nfev=int(sol.nfev),
        njev=int(getattr(sol, "njev", 0)),
        nlu=int(getattr(sol, "nlu", 0)),
        wall_time_s=wall,
        rtol=rtol, atol=atol,
        method=solver_method,
        q_t_default=q_t_default,
        include_q=include_q,
        config_versions=config_versions or {},
    )
