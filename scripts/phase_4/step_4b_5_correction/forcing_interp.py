"""Interpolate forcing at query times."""
import torch


def make_rho_ext_fn(forcing_norm_batch, t_grid=None):
    """Return rho_fn(t_q) -> interpolated forcing [B, Nq].

    Args:
        forcing_norm_batch: [B, 200] normalized rho_ext
        t_grid: [200] times (default linspace 0..10)
    """
    B, T = forcing_norm_batch.shape
    if t_grid is None:
        t_grid = torch.linspace(0.0, 10.0, T,
                                device=forcing_norm_batch.device,
                                dtype=forcing_norm_batch.dtype)
    t_grid = t_grid.to(forcing_norm_batch.device)

    def rho_fn(t_q):
        if t_q.dim() == 1:
            t_q = t_q.unsqueeze(0).expand(B, -1)
        t_clamped = t_q.clamp(min=float(t_grid[0]), max=float(t_grid[-1]))
        idx_right = torch.searchsorted(t_grid, t_clamped.contiguous(), right=False).clamp(1, T - 1)
        idx_left = idx_right - 1
        t_left = t_grid[idx_left]
        t_right = t_grid[idx_right]
        frac = (t_clamped - t_left) / (t_right - t_left + 1e-12)
        b_idx = torch.arange(B, device=forcing_norm_batch.device).unsqueeze(1).expand_as(idx_left)
        f_left = forcing_norm_batch[b_idx, idx_left]
        f_right = forcing_norm_batch[b_idx, idx_right]
        return f_left * (1 - frac) + f_right * frac

    return rho_fn
