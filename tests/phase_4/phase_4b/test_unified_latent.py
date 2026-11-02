"""Step 4B.3 — Unit tests for UnifiedLatentFNO (12 tests)."""
import sys
import math
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "src")
from models.boundary_aware_fno import UnifiedLatentFNO
from training.unified_latent import unified_latent_forward, unified_latent_step
from physics_informed.decoder import build_decoder


def _make_fake_fno(latent_dim=32, out_dim=9):
    """Build a minimal FNO-like module with .projection hook point.

    Mimics the real neuralop.FNO convention:
      - Pre-projection latent: [B, latent_dim, N] (channel-first)
      - Projection input: [B, latent_dim, N]
      - Projection output: [B, out_dim, N]

    Our fake projection is a 1x1 Conv1d so it accepts [B, C, N] and outputs
    [B, out_dim, N] directly. This matches how SCFNOModel expects to capture
    `inputs[0]` = [B, 32, N] from the hook on projection.
    """
    class FakeFNO(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_proj = nn.Conv1d(1, latent_dim, kernel_size=1)
            self.projection = nn.Conv1d(latent_dim, out_dim, kernel_size=1)

        def forward(self, x):
            # x: [B, 1, N] -> [B, latent_dim, N] -> [B, out_dim, N]
            z = self.latent_proj(x)            # [B, latent_dim, N]
            out = self.projection(z)           # [B, out_dim, N]
            return out
    return FakeFNO()


def _make_wrapped(pad_size=20):
    torch.manual_seed(0)
    fno = _make_fake_fno()
    return UnifiedLatentFNO(fno, pad_size=pad_size, pad_mode="reflect")


# ============================================================
# T1 — Padding shape
# ============================================================
def test_t1_padding_shape():
    model = _make_wrapped(pad_size=20)
    x = torch.randn(2, 1, 200)
    # Forward: pad 200 -> 240 -> FNO -> crop 240 -> 200
    out = model(x)
    assert out.shape == (2, 9, 200), f"Expected (2,9,200), got {tuple(out.shape)}"


# ============================================================
# T2 — Latent identity (bit-exact, most important)
# ============================================================
def test_t2_latent_identity_bit_exact():
    model = _make_wrapped()
    x = torch.randn(2, 1, 200)
    model.eval()
    with torch.no_grad():
        latent_a = model.get_latent(x)
        latent_b = model.get_latent(x)
    assert latent_a.shape == latent_b.shape == (2, 32, 200)
    assert torch.equal(latent_a, latent_b), (
        f"Latents must be bit-exact; max diff {(latent_a-latent_b).abs().max():.2e}"
    )


# ============================================================
# T3 — Cropping preserves physical points
# ============================================================
def test_t3_cropping_matches_manual():
    model = _make_wrapped(pad_size=20)
    x = torch.randn(1, 1, 200)
    model.eval()
    with torch.no_grad():
        # Manual forward: pad, FNO, capture pre-projection, crop
        import torch.nn.functional as F
        x_padded = F.pad(x, (20, 20), mode="reflect")
        captured = {}
        def _hook(module, inputs, output):
            captured["latent"] = inputs[0]
        hook = model.fno.projection.register_forward_hook(_hook)
        try:
            _ = model.fno(x_padded)
        finally:
            hook.remove()
        expected = captured["latent"][:, :, 20:-20]  # [1, 32, 200]

        actual = model.get_latent(x)
    assert torch.equal(actual, expected), "Cropped latent must match manual crop"


# ============================================================
# T4 — Query bounds
# ============================================================
def test_t4_query_bounds_in_physical_domain():
    """Verify the query sampling convention stays inside [0, 10]."""
    # Reconstruct the frozen sampling
    near_start = torch.linspace(0.5, 1.5, 4)
    middle = torch.linspace(1.5, 8.5, 42)
    near_end = torch.linspace(8.5, 9.5, 4)
    t_q = torch.cat([near_start, middle, near_end])
    assert t_q.shape == (50,)
    assert t_q.min() >= 0.5
    assert t_q.max() <= 9.5
    # All queries strictly inside [0, 10]
    assert (t_q >= 0.0).all() and (t_q <= 10.0).all()


# ============================================================
# T5 — Causal decoder unchanged
# ============================================================
def test_t5_decoder_class_unchanged():
    from physics_informed.decoder import CoordinateDecoder
    decoder = build_decoder(seed=42)
    assert isinstance(decoder, CoordinateDecoder)
    # No attention mask attribute should exist
    assert not hasattr(decoder, "attn_mask"), "Decoder must not have an attention mask"


# ============================================================
# T6 — Physical domain not extended (padding is not a physical observation)
# ============================================================
def test_t6_padding_not_physical():
    """Padding length is 240 but returned latent is 200. The 40 padded
    positions are never exposed to the decoder."""
    model = _make_wrapped(pad_size=20)
    x = torch.randn(1, 1, 200)
    model.eval()
    with torch.no_grad():
        latent = model.get_latent(x)
    # Latent must be exactly 200 points, not 240
    assert latent.shape[-1] == 200, f"Latent must be 200 points, got {latent.shape[-1]}"


# ============================================================
# T7 — Data + physics shapes
# ============================================================
def test_t7_data_and_physics_shapes():
    model = _make_wrapped()
    decoder = build_decoder(seed=42)
    x = torch.randn(2, 1, 200)
    t_data = torch.linspace(0.0, 10.0, 200).unsqueeze(0).expand(2, -1)
    t_q = torch.linspace(0.5, 9.5, 50).unsqueeze(0).expand(2, -1)
    model.eval()
    with torch.no_grad():
        latent, u_phys = unified_latent_forward(model, decoder, x, t_q)
        u_data = decoder(latent, t_data)
    assert latent.shape == (2, 32, 200)
    assert u_data.shape == (2, 9, 200)
    assert u_phys.shape == (2, 9, 50)


# ============================================================
# T8 — Boundary ratio diagnostic (measure, do not pass/fail)
# ============================================================
def test_t8_boundary_ratio_measurement():
    """Diagnostic: measure start/interior ratio on a synthetic case.
    Not a pass/fail test — records the value for later analysis."""
    model = _make_wrapped()
    x = torch.randn(4, 1, 200)
    model.eval()
    with torch.no_grad():
        out = model(x)  # [4, 9, 200]
    # Compute MAE over first 20 vs middle 160 (interior)
    err = out.abs()  # proxy since we don't have GT here
    start = err[:, :, :20].mean().item()
    interior = err[:, :, 20:180].mean().item()
    ratio = start / max(interior, 1e-12)
    # Just record; no assertion
    import json, pathlib
    pathlib.Path("results/phase_4/phase_4b/boundary").mkdir(parents=True, exist_ok=True)
    with open("results/phase_4/phase_4b/boundary/boundary_ratio_measurement.json", "w") as f:
        json.dump({"start_mae": start, "interior_mae": interior, "start_interior_ratio": ratio,
                   "note": "synthetic test, diagnostic only"}, f, indent=2)


# ============================================================
# T9 — Numerical stability with extreme inputs
# ============================================================
def test_t9_numerical_stability():
    model = _make_wrapped()
    x = torch.randn(2, 1, 200) * 100.0  # large values
    model.eval()
    with torch.no_grad():
        latent = model.get_latent(x)
        out = model(x)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(out).all()


# ============================================================
# T10 — Single forward pass count
# ============================================================
def test_t10_single_forward_count():
    """Verify get_latent triggers exactly ONE forward call."""
    model = _make_wrapped()
    call_count = {"n": 0}
    original_forward = model.fno.forward
    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)
    model.fno.forward = counting_forward
    x = torch.randn(1, 1, 200)
    model.eval()
    with torch.no_grad():
        _ = model.get_latent(x)
    assert call_count["n"] == 1, f"Expected 1 forward call, got {call_count['n']}"


# ============================================================
# T11 — Training-loop contract (one latent, two decoders)
# ============================================================
def test_t11_training_loop_contract():
    model = _make_wrapped()
    decoder = build_decoder(seed=42)
    x = torch.randn(2, 1, 200)
    t_data = torch.linspace(0.0, 10.0, 200).unsqueeze(0).expand(2, -1)
    t_q = torch.linspace(0.5, 9.5, 50).unsqueeze(0).expand(2, -1)
    model.eval()
    with torch.no_grad():
        latent, u_phys = unified_latent_forward(model, decoder, x, t_q)
        u_data = decoder(latent, t_data)
    assert torch.isfinite(u_data).all()
    assert torch.isfinite(u_phys).all()


# ============================================================
# T12 — Reproducibility (bit-exact across calls)
# ============================================================
def test_t12_reproducibility():
    torch.manual_seed(123)
    model = _make_wrapped()
    model.eval()
    x = torch.randn(2, 1, 200)
    with torch.no_grad():
        a = model.get_latent(x)
        b = model.get_latent(x)
    assert torch.equal(a, b), "Latent must be bit-exact reproducible"
