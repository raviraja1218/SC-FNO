"""Phase 5 Corrected — decoder health (Bug J prevention).

Bug J prevention metric: du_dt std > 0.05 (was 1.4e-3 in Bug J).

Quality checks (secondary):
  - u_hat variance > 50% of ground truth variance
  - reconstruction MSE < 0.5

Uses structured z (temporal signal like real FNO latent) to match real training.
"""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import pytest

from phase_5_corrected.decoder_v2 import build_decoder_v2


def make_toy_dataset(n_scenarios=16, n_points=200, n_channels=9, seed=42):
    """Toy: y_j(t) = A_j * sin(2π * f_j * t + φ_j) + offset_j."""
    torch.manual_seed(seed)
    t = torch.linspace(0.0, 10.0, n_points)
    A = torch.rand(n_channels) * 1.0 + 0.5
    f = torch.rand(n_channels) * 0.4 + 0.1
    phi = torch.rand(n_channels) * 2 * 3.14159
    offset = torch.randn(n_channels) * 0.1

    y_base = (A[None, :, None] * torch.sin(2 * 3.14159 * f[None, :, None] * t[None, None, :]
                                            + phi[None, :, None])
              + offset[None, :, None])

    y = torch.zeros(n_scenarios, n_channels, n_points)
    for s in range(n_scenarios):
        scale = 1.0 + 0.1 * torch.randn(n_channels)
        shift = 0.05 * torch.randn(n_channels)
        y[s] = y_base[0] * scale[:, None] + shift[:, None]
    return y, t


def test_decoder_health_toy_fit():
    """Verify decoder learns temporal dynamics (Bug J prevention)."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    y_toy, t_grid = make_toy_dataset(n_scenarios=16, n_points=200)
    y_toy = y_toy.to(device)
    B = y_toy.shape[0]

    # Structured z (matches real FNO latent — carries temporal information)
    torch.manual_seed(0)
    z_noise = torch.randn(B, 32, 200, device=device) * 0.1
    z_signal = torch.sin(torch.linspace(0, 10, 200, device=device)).unsqueeze(0).unsqueeze(0)
    z = z_noise + z_signal.expand(B, 32, 200) * 0.5

    t_q = torch.linspace(0.5, 9.5, 50, device=device)

    decoder = build_decoder_v2(latent_dim=32, hidden_dim=64, n_freqs=16, T=10.0, seed=42).to(device)

    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    def interp(yb, t_q):
        t_qn = (t_q / 10.0) * 199
        i0 = t_qn.floor().long().clamp(0, 198)
        i1 = (i0 + 1).clamp(0, 199)
        frac = (t_qn - i0.float()).unsqueeze(0).unsqueeze(0)
        b_idx = torch.arange(B, device=yb.device).unsqueeze(1).unsqueeze(2).expand(B, 9, len(t_q))
        s_idx = torch.arange(9, device=yb.device).unsqueeze(0).unsqueeze(2).expand(B, 9, len(t_q))
        y0 = yb[b_idx, s_idx, i0.unsqueeze(0).unsqueeze(0).expand(B, 9, -1)]
        y1 = yb[b_idx, s_idx, i1.unsqueeze(0).unsqueeze(0).expand(B, 9, -1)]
        return y0 * (1 - frac) + y1 * frac

    y_at_tq = interp(y_toy, t_q)

    for epoch in range(100):
        opt.zero_grad()
        u_hat = decoder(z, t_q.unsqueeze(0).expand(B, -1))
        loss = ((u_hat - y_at_tq) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

    with torch.no_grad():
        u_hat = decoder(z, t_q.unsqueeze(0).expand(B, -1))

    u_hat_variance = u_hat.var(dim=-1).mean().item()
    gt_variance = y_at_tq.var(dim=-1).mean().item()
    recon_mse = ((u_hat - y_at_tq) ** 2).mean().item()

    t_q_g = t_q.detach().clone().requires_grad_(True)
    u_hat_g = decoder(z, t_q_g.unsqueeze(0).expand(B, -1))
    du_dt = torch.autograd.grad(
        outputs=u_hat_g[:, 0, :], inputs=t_q_g,
        grad_outputs=torch.ones_like(u_hat_g[:, 0, :]),
        create_graph=False, retain_graph=True,
    )[0]
    du_dt_std = du_dt.std().item()

    print(f"\n  [DECODER HEALTH] gt variance (theoretical max): {gt_variance:.4f}")
    print(f"  [DECODER HEALTH] u_hat variance: {u_hat_variance:.4f} "
          f"({u_hat_variance/gt_variance:.1%} of max) — threshold > 50%")
    print(f"  [DECODER HEALTH] reconstruction MSE: {recon_mse:.6f} — threshold < 0.5")
    print(f"  [DECODER HEALTH] du_dt std: {du_dt_std:.4f} — threshold > 0.05 (Bug J prevention)")

    # CRITICAL: Bug J prevention
    assert du_dt_std > 0.05, f"du_dt std {du_dt_std:.4f} < 0.05 (Bug J recurrent)"

    # Quality thresholds (achievable)
    assert u_hat_variance > 0.5 * gt_variance, \
        f"u_hat variance {u_hat_variance:.4f} < 50% of max {gt_variance:.4f}"
    assert recon_mse < 0.5, f"recon_mse {recon_mse:.6f} > 0.5"
