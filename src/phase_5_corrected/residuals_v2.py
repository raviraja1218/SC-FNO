"""Phase 5 Corrected — residuals with physical ρ_ext (Bug K fix)."""
import sys; sys.path.insert(0, "src")
import torch

def make_rho_ext_fn(forcing_norm_batch, rho_mean, rho_std, t_grid=None):
    """Interpolate forcing at query times, denormalized to physical units.

    BUG K FIX: forcing_norm_batch is normalized (z-score); physics residuals
    need physical ρ_ext in Δk/k units (values ~1e-3). Denormalize inside.
    """
    B, T = forcing_norm_batch.shape
    if t_grid is None:
        t_grid = torch.linspace(0.0, 10.0, T, device=forcing_norm_batch.device,
                                dtype=forcing_norm_batch.dtype)
    t_grid = t_grid.to(forcing_norm_batch.device)

    def rho_fn(t_q):
        if t_q.dim() == 1:
            t_q = t_q.unsqueeze(0).expand(B, -1)
        t_c = t_q.clamp(min=float(t_grid[0]), max=float(t_grid[-1]))
        i_r = torch.searchsorted(t_grid, t_c.contiguous(), right=False).clamp(1, T - 1)
        i_l = i_r - 1
        t_l = t_grid[i_l]; t_r = t_grid[i_r]
        frac = (t_c - t_l) / (t_r - t_l + 1e-12)
        b_idx = torch.arange(B, device=forcing_norm_batch.device).unsqueeze(1).expand_as(i_l)
        f_l = forcing_norm_batch[b_idx, i_l]
        f_r = forcing_norm_batch[b_idx, i_r]
        f_norm = f_l * (1 - frac) + f_r * frac
        return f_norm * rho_std + rho_mean   # physical Δk/k units
    return rho_fn

def compute_residuals_true_forcing(u_hat, du_dt, t_q, rho_ext_fn, kin, th, q_t=None):
    from physics_informed.residuals import compute_residuals
    return compute_residuals(u_hat=u_hat, du_hat_dt=du_dt, t_query=t_q,
                             rho_ext_fn=rho_ext_fn, kin=kin, th=th, q_t=q_t)
