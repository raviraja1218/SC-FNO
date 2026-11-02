"""Phase 5 Corrected — composite loss including supervised decoder term.

L_total = λ_data·L_data + λ_phys·L_phys + λ_ic·L_ic + λ_dec·L_data_decoder

L_data_decoder anchors decoder output to interpolated ground truth at t_q.
This is the fix for Bug J (decoder temporal collapse).
"""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F

STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]


def interp_gt_to_tq(yb, t_q, t_end=10.0):
    """Linearly interpolate ground truth [B, 9, 200] to query times [B, Nq].

    Args:
        yb: [B, 9, 200] ground truth at 200 uniform points in [0, t_end]
        t_q: [B, Nq] query times in seconds
    Returns:
        y_at_tq: [B, 9, Nq]
    """
    B, S, N = yb.shape
    Nq = t_q.shape[1]
    t_grid = torch.linspace(0.0, t_end, N, device=yb.device, dtype=yb.dtype)
    t_qn = (t_q / t_end) * (N - 1)  # [B, Nq] in index space
    i0 = t_qn.floor().long().clamp(0, N - 2)         # [B, Nq]
    i1 = (i0 + 1).clamp(0, N - 1)
    frac = (t_qn - i0.float()).unsqueeze(1)          # [B, 1, Nq]

    # Gather per-channel
    b_idx = torch.arange(B, device=yb.device).unsqueeze(1).unsqueeze(2).expand(B, S, Nq)
    s_idx = torch.arange(S, device=yb.device).unsqueeze(0).unsqueeze(2).expand(B, S, Nq)
    y0 = yb[b_idx, s_idx, i0.unsqueeze(1).expand(B, S, Nq)]
    y1 = yb[b_idx, s_idx, i1.unsqueeze(1).expand(B, S, Nq)]
    return y0 * (1 - frac) + y1 * frac


def compute_l_data_decoder(model, rho_ext, yb, t_q):
    """Supervised decoder loss: MSE(decoder(z, t_q), interp(yb, t_q)).

    Normalized yb; normalized decoder output. Both on same scale.
    """
    u_hat = model.forward_decoder(rho_ext, t_q)      # [B, 9, Nq] normalized
    y_at_tq = interp_gt_to_tq(yb, t_q)               # [B, 9, Nq] normalized
    return ((u_hat - y_at_tq) ** 2).mean()


def compute_l_ic(model, rho_ext, yb, loss_fn):
    """IC loss: MSE(decoder(z, t=0), yb[:, :, 0]) in physical units."""
    B = rho_ext.shape[0]
    t0 = torch.zeros(B, 1, device=rho_ext.device)
    u0_norm = model.forward_decoder(rho_ext, t0)     # [B, 9, 1] normalized
    u0_phys = loss_fn._denorm_batch(u0_norm)         # [B, 9, 1] physical
    y0_phys = loss_fn._denorm_batch(yb[:, :, 0].unsqueeze(-1)).squeeze(-1)  # [B, 9]
    return ((u0_phys[:, :, 0] - y0_phys) ** 2).mean()


def compute_l_phys(u_hat_phys, du_dt, t_q, rho_ext_fn, kin, th, loss_fn):
    """9-equation physics loss with per-state p95 normalization."""
    from phase_5_corrected.residuals_v2 import compute_residuals_true_forcing
    R = compute_residuals_true_forcing(
        u_hat=u_hat_phys, du_dt=du_dt, t_q=t_q,
        rho_ext_fn=rho_ext_fn, kin=kin, th=th,
    )
    stacked = R.stacked()                             # [B, 9, Nq]
    terms = []
    for sj, k in enumerate(STATES):
        s_j = loss_fn.residual_scales[k]["residual_p95"]
        terms.append(((stacked[:, sj, :] / s_j) ** 2).mean())
    return torch.stack(terms).mean()


def compute_composite_loss_v2(
    model, rho_ext, yb_norm, t_q, rho_ext_fn, kin, th, loss_fn,
    lambda_phys, lambda_dec=1.0, lambda_ic=1.0,
):
    """Full composite loss (Phase 5 Corrected)."""
    from phase_5_corrected.derivatives_v2 import compute_du_dt_per_channel

    # Data path
    pred_data = model.forward_data(rho_ext)          # [B, 9, 200] normalized
    L_data = ((pred_data - yb_norm) ** 2).mean()

    # Decoder path (needs grad for derivative)
    u_hat_norm, du_dt = compute_du_dt_per_channel(model, rho_ext, t_q)  # [B,9,Nq]
    u_hat_phys = loss_fn._denorm_batch(u_hat_norm)   # [B, 9, Nq] physical

    L_data_decoder = compute_l_data_decoder(model, rho_ext, yb_norm, t_q)
    L_phys = compute_l_phys(u_hat_phys, du_dt, t_q, rho_ext_fn, kin, th, loss_fn)
    L_ic = compute_l_ic(model, rho_ext, yb_norm, loss_fn)

    L_total = 1.0 * L_data + lambda_phys * L_phys + lambda_ic * L_ic + lambda_dec * L_data_decoder

    return {
        "L_data": L_data,
        "L_phys": L_phys,
        "L_ic": L_ic,
        "L_data_decoder": L_data_decoder,
        "L_total": L_total,
    }
