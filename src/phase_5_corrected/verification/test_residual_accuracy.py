"""Phase 5 Corrected — residual accuracy tests (minimal for 5.A; full in 5.B)."""
import sys
sys.path.insert(0, "src")

import torch


def test_true_forcing_used():
    """Verify rho_ext_fn is actually used (not zero-forcing)."""
    from phase_5_corrected.residuals_v2 import make_rho_ext_fn

    # Two scenarios with different forcing
    forcing_a = torch.zeros(1, 200)
    forcing_a[0, 100:] = 1.0              # step at t=5
    forcing_b = torch.zeros(1, 200)
    forcing_b[0, :] = -1.0                # constant negative

    t_q = torch.linspace(0.5, 9.5, 50).unsqueeze(0)

    rho_fn_a = make_rho_ext_fn(forcing_a)
    rho_fn_b = make_rho_ext_fn(forcing_b)

    rho_a = rho_fn_a(t_q)
    rho_b = rho_fn_b(t_q)

    diff = (rho_a - rho_b).abs().mean().item()
    assert diff > 0.1, f"rho_a and rho_b should differ, got mean diff {diff:.4f}"
    # rho_a should be 0 for early times, 1 for late; rho_b always -1
    assert rho_a[0, 0].abs() < 1e-5, f"rho_a at t=0.5 should be 0"
    assert rho_a[0, -1] > 0.9, f"rho_a at t=9.5 should be ~1"
    assert abs(rho_b[0, 0].item() - (-1.0)) < 1e-5, f"rho_b should be -1"
