"""Lumped-parameter thermal-hydraulic model: fuel and coolant temperatures.

Two-node model: fuel node T_f, coolant node T_c.
State convention: [T_f, T_c]
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class THParameters:
    M_f: float; c_p_f: float; f_f: float
    M_c: float; c_p_c: float; m_dot: float; T_in: float
    U: float; A: float
    P0: float
    alpha_f: float; alpha_c: float
    T_f_ref: float; T_c_ref: float

    @classmethod
    def from_yaml(cls, th: dict) -> "THParameters":
        f, c, ht, p, fb = th["fuel"], th["coolant"], th["heat_transfer"], th["power"], th["reactivity_feedback"]
        return cls(
            M_f=float(f["M_f"]), c_p_f=float(f["c_p_f"]), f_f=float(f["f_f"]),
            M_c=float(c["M_c"]), c_p_c=float(c["c_p_c"]),
            m_dot=float(c["m_dot"]), T_in=float(c["T_in"]),
            U=float(ht["U"]), A=float(ht["A"]),
            P0=float(p["P0"]),
            alpha_f=float(fb["alpha_f"]), alpha_c=float(fb["alpha_c"]),
            T_f_ref=float(fb["T_f_ref"]), T_c_ref=float(fb["T_c_ref"]),
        )


def th_rhs(n: float, T_f: float, T_c: float, th: THParameters) -> tuple[float, float]:
    """Return (dT_f/dt, dT_c/dt) given current n, T_f, T_c."""
    q_f = th.f_f * th.P0 * n
    q_c = (1.0 - th.f_f) * th.P0 * n
    q_fc = th.U * th.A * (T_f - T_c)
    dT_f = (q_f - q_fc) / (th.M_f * th.c_p_f)
    dT_c = (q_c + q_fc - 2.0 * th.m_dot * th.c_p_c * (T_c - th.T_in)) / (th.M_c * th.c_p_c)
    return dT_f, dT_c


def feedback_reactivity(T_f: float, T_c: float, th: THParameters) -> float:
    return th.alpha_f * (T_f - th.T_f_ref) + th.alpha_c * (T_c - th.T_c_ref)
