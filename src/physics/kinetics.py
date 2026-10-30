"""Point Kinetics Equations (PKE) with m delayed neutron precursor groups.

State vector convention: [n, C_1, ..., C_m]
External inputs: rho_ext(t), q(t), and temperature feedback from TH module.

Reference: Keepin, Physics of Nuclear Kinetics (1965); Duderstadt & Hamilton,
Nuclear Reactor Analysis (1976), Ch. 7.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Callable
import numpy as np


@dataclass(frozen=True)
class KineticsParameters:
    beta_eff: float
    Lambda: float
    beta_i: Sequence[float]     # length m
    lambda_i: Sequence[float]   # length m

    @property
    def m(self) -> int:
        return len(self.beta_i)

    def __post_init__(self):
        if len(self.beta_i) != len(self.lambda_i):
            raise ValueError("beta_i and lambda_i must have equal length")
        if abs(sum(self.beta_i) - self.beta_eff) > 1e-10:
            raise ValueError(
                f"sum(beta_i)={sum(self.beta_i)} != beta_eff={self.beta_eff}"
            )

    @classmethod
    def from_yaml(cls, rp: dict) -> "KineticsParameters":
        n = rp["neutronics"]
        groups = n["precursor_groups"]
        return cls(
            beta_eff=float(n["beta_eff"]),
            Lambda=float(n["Lambda"]),
            beta_i=[float(g["beta_i"]) for g in groups],
            lambda_i=[float(g["lambda_i"]) for g in groups],
        )


def equilibrium_precursors(n0: float, kin: KineticsParameters) -> np.ndarray:
    """C_i0 = beta_i * n0 / (Lambda * lambda_i) — steady state under rho=0, q=0."""
    return np.array([b * n0 / (kin.Lambda * lam)
                     for b, lam in zip(kin.beta_i, kin.lambda_i)])


def pke_rhs(t: float,
            y: np.ndarray,
            kin: KineticsParameters,
            rho_total_fn: Callable[[float, np.ndarray], float],
            q_fn: Callable[[float], float] | None = None,
            include_q: bool = True) -> np.ndarray:
    """Right-hand side of PKE.

    Parameters
    ----------
    t : float
        Current time [s].
    y : ndarray, shape (m+1,)
        State vector [n, C_1, ..., C_m] where n is normalized neutron density
        and C_i are precursor concentrations [arbitrary units consistent with n].
    kin : KineticsParameters
    rho_total_fn : callable (t, y) -> float
        Total reactivity rho(t) = rho_ext(t) + alpha_f*dT_f + alpha_c*dT_c.
    q_fn : callable or None
        External neutron source q(t). Must be provided if include_q is True.
    include_q : bool
        If True (Phase 0.4 locked decision), q(t) term enters n-equation.
    """
    n = y[0]
    C = y[1:]
    rho = rho_total_fn(t, y)

    q = 0.0
    if include_q:
        if q_fn is None:
            raise ValueError("q_fn required when include_q=True")
        q = q_fn(t)

    dn_dt = (rho - kin.beta_eff) / kin.Lambda * n + float(np.dot(kin.lambda_i, C)) + q
    dC_dt = np.array([b / kin.Lambda * n - lam * c
                      for b, lam, c in zip(kin.beta_i, kin.lambda_i, C)])
    return np.concatenate(([dn_dt], dC_dt))
