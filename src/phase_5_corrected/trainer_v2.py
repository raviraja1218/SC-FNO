"""Phase 5 Corrected — training loop with supervised decoder data loss.

Fixes Bug J: L_data_decoder anchors decoder output to interpolated ground
truth at query times, preventing temporal collapse.
"""
import sys
sys.path.insert(0, "src")

import time
import torch
import numpy as np

from phase_5_corrected.losses_v2 import compute_composite_loss_v2
from phase_5_corrected.residuals_v2 import make_rho_ext_fn


def phase5_default_ramp(epoch, init=1e-4, target=1.0, ramp_epochs=100):
    """λ_phys ramp for Phase 5 corrected."""
    if epoch >= ramp_epochs:
        return target
    return init * (10.0 ** (epoch / ramp_epochs * np.log10(target / init)))


def train_epoch_phase5_corrected(
    model, optimizer, scheduler, X_tr, Y_tr, kin, th, loss_fn, epoch,
    batch_size=16, t_q_length=50, lambda_phys=None, device="cuda",
    log_dict=None,
):
    """Train one epoch with supervised decoder data loss."""
    model.train()
    N = X_tr.shape[0]

    if lambda_phys is None:
        lambda_phys = phase5_default_ramp(epoch)

    T_QUERY = torch.linspace(0.5, 9.5, t_q_length, device=device)

    perm = torch.randperm(N)
    X_tr_p, Y_tr_p = X_tr[perm], Y_tr[perm]

    sums = {"L_data": 0.0, "L_phys": 0.0, "L_ic": 0.0,
            "L_data_decoder": 0.0, "L_total": 0.0}
    n_batches = 0

    for bi in range(0, N, batch_size):
        b_end = min(bi + batch_size, N)
        xb = X_tr_p[bi:b_end].unsqueeze(1).to(device)     # [B, 1, 200]
        yb = Y_tr_p[bi:b_end].to(device)                  # [B, 9, 200]
        B = xb.shape[0]
        tq = T_QUERY.unsqueeze(0).expand(B, -1).to(device)  # [B, Nq]

        # Build per-batch rho_ext_fn
        rho_fn = make_rho_ext_fn(xb[:, 0, :])              # normalize is already in xb

        optimizer.zero_grad()
        losses = compute_composite_loss_v2(
            model, xb, yb, tq, rho_fn, kin, th, loss_fn,
            lambda_phys=lambda_phys,
        )
        losses["L_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            sums[k] += float(v.detach())
        n_batches += 1

    scheduler.step()

    for k in sums:
        sums[k] /= max(n_batches, 1)

    if log_dict is not None:
        log_dict.append({
            "epoch": epoch, "lambda_phys": lambda_phys,
            **{k: sums[k] for k in sums},
        })

    return sums


@torch.no_grad()
def validate_phase5_corrected(model, X_va, Y_va, batch_size=16, device="cuda"):
    """Validation data loss (projection MLP path)."""
    model.eval()
    N = X_va.shape[0]
    total = 0.0
    n = 0
    for bi in range(0, N, batch_size):
        b_end = min(bi + batch_size, N)
        xb = X_va[bi:b_end].unsqueeze(1).to(device)
        yb = Y_va[bi:b_end].to(device)
        pred = model.forward_data(xb)
        total += float(((pred - yb) ** 2).mean()) * (b_end - bi)
        n += b_end - bi
    return total / max(n, 1)
