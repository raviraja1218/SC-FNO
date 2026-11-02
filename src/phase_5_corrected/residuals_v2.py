"""Phase 5 Corrected — 9-equation residuals with TRUE forcing.

Bug B fix: rho_ext_fn is a callable that returns the actual interpolated
external reactivity at query times, NOT zeros_like(t_q).
"""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F


def make_rho_ext_fn(forcing_norm_batch, t_grid=None):
    """Build rho_ext_fn from scenario forcing.

    Args:
        forcing_norm_batch: [B, 200] tensor — normalized rho_ext values
        t_grid: [200] times (default linspace(0, 10, 200))

    Returns:
        rho_fn(t_q) -> [B, Nq] interpolated normalized rho_ext
    """
    B, T = forcing_norm_batch.shape
    if t_grid is None:
        t_grid = torch.linspace(0.0, 10.0, T, device=forcing_norm_batch.device,
                                dtype=forcing_norm_batch.dtype)
    t_grid = t_grid.to(forcing_norm_batch.device)

    def rho_fn(t_q):
        if t_q.dim() == 1:
            t_q = t_q.unsqueeze(0).expand(B, -1)
        t_clamped = t_q.clamp(min=float(t_grid[0]), max=float(t_grid[-1]))
        idx_right = torch.searchsorted(t_grid, t_clamped.contiguous(),
                                       right=False).clamp(1, T - 1)
        idx_left = idx_right - 1
        t_left = t_grid[idx_left]
        t_right = t_grid[idx_right]
        frac = (t_clamped - t_left) / (t_right - t_left + 1e-12)
        b_idx = torch.arange(B, device=forcing_norm_batch.device).unsqueeze(1).expand_as(idx_left)
        f_left = forcing_norm_batch[b_idx, idx_left]
        f_right = forcing_norm_batch[b_idx, idx_right]
        return f_left * (1 - frac) + f_right * frac

    return rho_fn


def compute_residuals_true_forcing(
    u_hat, du_dt, t_q, rho_ext_fn, kin, th, q_t=None,
):
    """Compute 9 residuals with TRUE forcing.

    Wraps physics_informed.residuals.compute_residuals (Phase 3 code)
    but is documented as Phase 5 corrected entry point. Uses actual
    rho_ext_fn, not zero-forcing.

    u_hat: [B, 9, Nq]     — PHYSICAL states (denormalized)
    du_dt: [B, 9, Nq]     — per-channel derivatives (normalized units)
    t_q:   [B, Nq]        — query times
    """
    from physics_informed.residuals import compute_residuals
    return compute_residuals(
        u_hat=u_hat, du_hat_dt=du_dt, t_query=t_q,
        rho_ext_fn=rho_ext_fn, kin=kin, th=th, q_t=q_t,
    )
