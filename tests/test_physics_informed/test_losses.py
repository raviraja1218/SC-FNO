"""Step 3.3 — Composite loss tests (Item C1, C2 from reviewer)."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.decoder import build_decoder
from physics_informed.derivatives import load_phase2_fno, SCFNOModel
from physics_informed.losses import CompositeLoss, load_composite_loss

RESULTS = Path("results/phase_3/loss_integration")
RESULTS.mkdir(parents=True, exist_ok=True)


def _kin():
    return KineticsParameters(
        beta_eff=0.006502, Lambda=1e-4,
        beta_i=[0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273],
        lambda_i=[0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100],
    )

def _th():
    return THParameters(
        M_f=27000.0, c_p_f=300.0, f_f=0.97, M_c=4000.0, c_p_c=5500.0,
        m_dot=13636.36, T_in=563.0, U=3060.0, A=3000.0, P0=3.0e9,
        alpha_f=-3e-5, alpha_c=-5e-4, T_f_ref=900.0, T_c_ref=583.0,
    )

def _make_batch(B=2, Nq=20, device="cpu"):
    torch.manual_seed(0)
    rho_ext = torch.randn(B, 1, 200, device=device) * 1e-3
    u_gt = torch.randn(B, 9, 200, device=device) * 0.1
    t_q = torch.linspace(0.5, 9.5, Nq, device=device).unsqueeze(0).expand(B, -1).contiguous()
    def rho_fn(tq):
        return torch.zeros_like(tq)
    return rho_ext, u_gt, t_q, rho_fn


def _build_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fno = load_phase2_fno(device=device)
    decoder = build_decoder(seed=42).to(device)
    return SCFNOModel(fno, decoder).to(device).train()


def _load_loss():
    return load_composite_loss(lam_data=1.0, lam_phys=1e-4, lam_IC=1.0)


# ============================================================
# Item C1: IC loss reaches final decoder bias
# ============================================================
def test_C1_IC_loss_reaches_final_decoder_bias():
    """IC loss MUST produce non-zero gradient on decoder.head[-1].bias."""
    model = _build_model()
    loss_fn = _load_loss()
    device = next(model.parameters()).device

    rho_ext, u_gt, _, _ = _make_batch(device=device)
    u_gt_t0 = u_gt[:, :, 0]

    L_IC = loss_fn.compute_ic_loss(model, rho_ext, u_gt_t0)
    model.zero_grad()
    L_IC.backward()

    final_bias = model.decoder.head[-1].bias
    assert final_bias.grad is not None, \
        "IC loss produced None gradient on decoder.head[-1].bias"
    assert final_bias.grad.abs().sum().item() > 0, \
        "IC loss produced ZERO gradient on decoder.head[-1].bias"
    assert torch.isfinite(final_bias.grad).all()

    out = {
        "test": "C1 — IC loss gradient reaches decoder.head[-1].bias",
        "loss_value": float(L_IC.item()),
        "bias_grad_norm": float(final_bias.grad.abs().sum().item()),
        "bias_grad_finite": bool(torch.isfinite(final_bias.grad).all()),
        "pass": True,
    }
    (RESULTS / "item_c1_ic_grad_reaches_bias.json").write_text(json.dumps(out, indent=2))


# ============================================================
# Item C2: Data loss does NOT reach decoder
# ============================================================
def test_C2_data_loss_does_NOT_route_through_decoder():
    """Data loss routes only through FNO. Decoder bias must have grad=None."""
    model = _build_model()
    loss_fn = _load_loss()
    device = next(model.parameters()).device

    rho_ext, u_gt, _, _ = _make_batch(device=device)

    L_data = loss_fn.compute_data_loss(model, rho_ext, u_gt)
    model.zero_grad()
    L_data.backward()

    # Decoder bias must have NO gradient (path not involved)
    final_bias = model.decoder.head[-1].bias
    assert final_bias.grad is None, \
        f"Decoder bias received gradient from data loss (unexpected: {final_bias.grad})"

    # But projection bias (final FNO layer) MUST have gradient
    proj_final_bias = model.fno.projection.fcs[-1].bias
    assert proj_final_bias.grad is not None, \
        "Projection bias has None gradient from data loss — path is broken"
    assert proj_final_bias.grad.abs().sum().item() > 0, \
        "Projection bias has zero gradient from data loss"

    out = {
        "test": "C2 — Data loss routes through FNO only, not decoder",
        "loss_value": float(L_data.item()),
        "decoder_bias_grad_is_none": True,
        "projection_bias_grad_norm": float(proj_final_bias.grad.abs().sum().item()),
        "pass": True,
    }
    (RESULTS / "item_c2_data_does_not_reach_decoder.json").write_text(json.dumps(out, indent=2))


# ============================================================
# Sanity: composite loss runs, all three components finite
# ============================================================
def test_composite_loss_runs_finite():
    """Composite loss computes all three, all finite, gradient flows."""
    model = _build_model()
    loss_fn = _load_loss()
    device = next(model.parameters()).device

    rho_ext, u_gt, t_q, rho_fn = _make_batch(device=device)
    kin, th = _kin(), _th()

    out = loss_fn(model, rho_ext, u_gt, t_q, rho_fn, kin, th)

    assert torch.isfinite(out.total_loss), f"total_loss not finite: {out.total_loss}"
    assert torch.isfinite(out.data_loss), "data_loss not finite"
    assert torch.isfinite(out.physics_loss), "physics_loss not finite"
    assert torch.isfinite(out.ic_loss), "ic_loss not finite"

    model.zero_grad()
    out.total_loss.backward()

    # Shared params receive gradient
    shared_grad = 0.0
    for p in model.fno.fno_blocks.parameters():
        if p.grad is not None:
            shared_grad += float(p.grad.abs().sum().item())
    assert shared_grad > 0, "No gradient on shared FNO blocks"

    (RESULTS / "composite_loss_verification.json").write_text(json.dumps({
        "data_loss": out.data_loss.item(),
        "physics_loss": out.physics_loss.item(),
        "ic_loss": out.ic_loss.item(),
        "total_loss": out.total_loss.item(),
        "shared_grad_norm": shared_grad,
        "all_finite": True,
        "pass": True,
    }, indent=2))
