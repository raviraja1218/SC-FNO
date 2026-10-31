"""Step 3.1 — Nine-state coupled reactor physics residuals.

Authoritative equations from locked Phase 1 modules:
  src/physics/kinetics.py            (KineticsParameters with fields: beta_eff, Lambda, beta_i, lambda_i)
  src/physics/thermal_hydraulics.py  (THParameters, feedback_reactivity)

State ordering (locked):  [n, C1, C2, C3, C4, C5, C6, Tf, Tc]

CRITICAL: All residual computation uses PREDICTED states from `u_hat`.
Ground-truth T_f, T_c are NEVER passed into residual computation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters


@dataclass
class ResidualOutput:
    R_n:      torch.Tensor   # [B, N]
    R_C:      torch.Tensor   # [B, 6, N]
    R_Tf:     torch.Tensor   # [B, N]
    R_Tc:     torch.Tensor   # [B, N]
    rho_total: torch.Tensor  # [B, N]
    rho_ext:   torch.Tensor  # [B, N]
    rho_fb:    torch.Tensor  # [B, N]

    def stacked(self) -> torch.Tensor:
        """Return [B, 9, N] residuals in state ordering."""
        return torch.cat([
            self.R_n.unsqueeze(1),
            self.R_C,
            self.R_Tf.unsqueeze(1),
            self.R_Tc.unsqueeze(1),
        ], dim=1)


def _bcast_scalar(x, ref: torch.Tensor) -> torch.Tensor:
    """Return a 0-dim scalar tensor on ref's device/dtype.

    Caller is expected to rely on broadcasting, NOT on full-shape expansion.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=ref.device, dtype=ref.dtype).reshape(())
    return torch.tensor(float(x), device=ref.device, dtype=ref.dtype)

def compute_residuals(
    u_hat: torch.Tensor,             # [B, 9, N]
    du_hat_dt: torch.Tensor,         # [B, 9, N]
    t_query: torch.Tensor,           # [B, N]
    rho_ext_fn: Callable[[torch.Tensor], torch.Tensor],
    kin: KineticsParameters,
    th: THParameters,
    q_t: Optional[torch.Tensor] = None,
) -> ResidualOutput:
    B, n_states, N = u_hat.shape
    assert n_states == 9, f"Expected 9 states, got {n_states}"
    assert du_hat_dt.shape == u_hat.shape
    assert t_query.shape == (B, N)

    ref = u_hat

    # --- Unpack predicted states ---
    n_hat  = u_hat[:, 0, :]              # [B, N]
    C_hat  = u_hat[:, 1:7, :]            # [B, 6, N]
    Tf_hat = u_hat[:, 7, :]              # [B, N]
    Tc_hat = u_hat[:, 8, :]              # [B, N]

    dn_dt  = du_hat_dt[:, 0, :]          # [B, N]
    dC_dt  = du_hat_dt[:, 1:7, :]        # [B, 6, N]
    dTf_dt = du_hat_dt[:, 7, :]          # [B, N]
    dTc_dt = du_hat_dt[:, 8, :]          # [B, N]

    # --- Parameters ---
    beta_eff = _bcast_scalar(kin.beta_eff, ref)   # scalar tensor
    Lambda   = _bcast_scalar(kin.Lambda,   ref)   # scalar tensor

    # beta_i and lambda_i -> [1, 6, 1] for broadcasting over [B, 6, N]
    beta_vec = torch.as_tensor(kin.beta_i,    device=ref.device, dtype=ref.dtype).view(1, 6, 1)
    lam_vec  = torch.as_tensor(kin.lambda_i,  device=ref.device, dtype=ref.dtype).view(1, 6, 1)

    # --- Total reactivity ---
    rho_ext = rho_ext_fn(t_query)                                 # [B, N]
    alpha_f = _bcast_scalar(th.alpha_f, ref)
    alpha_c = _bcast_scalar(th.alpha_c, ref)
    Tf_ref  = _bcast_scalar(th.T_f_ref, ref)
    Tc_ref  = _bcast_scalar(th.T_c_ref, ref)
    rho_fb = alpha_f * (Tf_hat - Tf_ref) + alpha_c * (Tc_hat - Tc_ref)
    rho_total = rho_ext + rho_fb

    # --- q(t) source ---
    if q_t is None:
        q = torch.zeros_like(n_hat)
    else:
        q = q_t.to(device=ref.device, dtype=ref.dtype)
        assert q.shape == n_hat.shape

    # --- R_n ---
    sum_lambda_C = (lam_vec * C_hat).sum(dim=1)                   # [B, N]
    pke_rate_n = (rho_total - beta_eff) / Lambda * n_hat + sum_lambda_C + q
    R_n = dn_dt - pke_rate_n

    # --- R_Ci (6 separate channels) ---
    # broadcast: [1,6,1]/scalar * [B,1,N] - [1,6,1] * [B,6,N] = [B,6,N]
    pke_rate_C = beta_vec / Lambda * n_hat.unsqueeze(1) - lam_vec * C_hat
    R_C = dC_dt - pke_rate_C

    # --- R_Tf ---
    M_f   = _bcast_scalar(th.M_f,   ref)
    c_p_f = _bcast_scalar(th.c_p_f, ref)
    f_f   = _bcast_scalar(th.f_f,   ref)
    P0    = _bcast_scalar(th.P0,    ref)
    UA    = _bcast_scalar(th.U, ref) * _bcast_scalar(th.A, ref)

    rate_Tf = (f_f * P0 / (M_f * c_p_f)) * n_hat \
              - (UA / (M_f * c_p_f)) * (Tf_hat - Tc_hat)
    R_Tf = dTf_dt - rate_Tf

    # --- R_Tc ---
    M_c   = _bcast_scalar(th.M_c,   ref)
    c_p_c = _bcast_scalar(th.c_p_c, ref)
    m_dot = _bcast_scalar(th.m_dot, ref)
    T_in  = _bcast_scalar(th.T_in,  ref)

    rate_Tc = ((1.0 - f_f) * P0 / (M_c * c_p_c)) * n_hat \
              + (UA / (M_c * c_p_c)) * (Tf_hat - Tc_hat) \
              - (2.0 * m_dot * c_p_c / (M_c * c_p_c)) * (Tc_hat - T_in)
    R_Tc = dTc_dt - rate_Tc

    return ResidualOutput(
        R_n=R_n, R_C=R_C, R_Tf=R_Tf, R_Tc=R_Tc,
        rho_total=rho_total, rho_ext=rho_ext, rho_fb=rho_fb,
    )
