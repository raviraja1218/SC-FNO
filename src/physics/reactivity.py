"""Total reactivity composition: external + thermal-hydraulic feedback."""
from __future__ import annotations
from typing import Callable
from .thermal_hydraulics import THParameters, feedback_reactivity


def make_rho_total(rho_ext_fn: Callable[[float], float],
                   th: THParameters) -> Callable[[float, "np.ndarray"], float]:
    """Return rho_total(t, y) combining external reactivity and TH feedback.

    State vector y = [n, C_1..C_m, T_f, T_c]. T_f, T_c at positions -2, -1.
    """
    def rho_total(t, y):
        T_f = float(y[-2]); T_c = float(y[-1])
        return float(rho_ext_fn(t)) + feedback_reactivity(T_f, T_c, th)
    return rho_total
