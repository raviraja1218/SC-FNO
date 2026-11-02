"""Phase 5 Corrected — gradient flow tests."""
import sys
sys.path.insert(0, "src")

import torch
import pytest

from phase_5_corrected.derivatives_v2 import compute_du_dt_per_channel, compute_du_dt_per_sum


class ToyDecoder(torch.nn.Module):
    """Simple decoder for testing: u_j(t) = a_j * sin(w_j * t + phi_j)."""
    def __init__(self, A, W, PHI):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.W = torch.nn.Parameter(torch.tensor(W, dtype=torch.float32))
        self.PHI = torch.nn.Parameter(torch.tensor(PHI, dtype=torch.float32))
    def forward(self, z, t_q):
        # t_q: [B, Nq]. Output: [B, 9, Nq].
        t = t_q.unsqueeze(1)  # [B, 1, Nq]
        return self.A[None, :, None] * torch.sin(self.W[None, :, None] * t + self.PHI[None, :, None])


class ToyModel(torch.nn.Module):
    """Minimal model that routes forward_decoder to a toy decoder."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
    def forward_decoder(self, rho_ext, t_q):
        return self.decoder(None, t_q)


def test_per_channel_derivative_correct():
    """Per-channel derivative must equal analytical d/dt of u_j(t)."""
    A = [1.0] * 9
    W = [0.5 + 0.1*j for j in range(9)]      # different freq per channel
    PHI = [0.1 * j for j in range(9)]
    decoder = ToyDecoder(A, W, PHI)
    model = ToyModel(decoder)

    rho_ext = torch.zeros(2, 1, 200)
    t_q = torch.linspace(0.5, 9.5, 50).unsqueeze(0).expand(2, -1)

    u_hat, du_dt = compute_du_dt_per_channel(model, rho_ext, t_q)
    # Analytical: du_j/dt = A_j * W_j * cos(W_j*t + PHI_j)
    t = t_q.unsqueeze(1)
    expected_du = decoder.A[None, :, None] * decoder.W[None, :, None] * torch.cos(
        decoder.W[None, :, None] * t + decoder.PHI[None, :, None])
    err = (du_dt - expected_du).abs().max().item()
    assert err < 1e-4, f"per-channel derivative max error {err:.3e} > 1e-4"


def test_per_sum_derivative_differs():
    """Per-sum derivative is NOT correct for multi-channel signals."""
    A = [1.0] * 9
    W = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # very different
    PHI = [0.0] * 9
    decoder = ToyDecoder(A, W, PHI)
    model = ToyModel(decoder)

    rho_ext = torch.zeros(2, 1, 200)
    t_q = torch.linspace(0.5, 9.5, 50).unsqueeze(0).expand(2, -1)

    _, du_per_ch = compute_du_dt_per_channel(model, rho_ext, t_q)
    _, du_per_sum = compute_du_dt_per_sum(model, rho_ext, t_q)

    # They should differ significantly (per-sum gives same value across channels)
    diff = (du_per_ch - du_per_sum).abs().mean().item()
    assert diff > 0.1, f"per-sum and per-channel should differ, got mean diff {diff:.3e}"

    # per-sum: all channels identical per timestep
    variance_across_channels = du_per_sum.var(dim=1).mean().item()
    assert variance_across_channels < 1e-6, "per-sum should give identical values across channels"
