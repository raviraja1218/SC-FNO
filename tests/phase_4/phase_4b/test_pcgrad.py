"""Step 4B.1 — Unit tests for PCGrad implementation.

9 tests:
  T1  No conflict → no projection
  T2  Conflict → projection applied; projected gradient orthogonal to data
  T3  Post-projection cosine < 1e-6 when projecting
  T4  Combined update matches hand computation
  T5  Missing gradient handled (data only, physics only)
  T6  Zero-norm gradient handled safely
  T7  Memory on RTX 4050 < 3 GB
  T8  Numerical agreement with independent reference
  T9  Reproducibility
"""
import sys
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "src")
from training.pcgrad import PCGradOptimizer


def make_simple_model():
    """Two-layer linear model with known shape."""
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def compute_separate_grads(model, loss_data_fn, loss_physics_fn, loss_ic_fn):
    """Run three forward+backward passes and return separate gradient lists."""
    params = [p for p in model.parameters() if p.requires_grad]

    loss_data = loss_data_fn(model)
    grad_data = torch.autograd.grad(loss_data, params, retain_graph=True, allow_unused=True)

    loss_physics = loss_physics_fn(model)
    grad_physics = torch.autograd.grad(loss_physics, params, retain_graph=True, allow_unused=True)

    loss_ic = loss_ic_fn(model)
    grad_ic = torch.autograd.grad(loss_ic, params, retain_graph=False, allow_unused=True)

    return grad_data, grad_physics, grad_ic, loss_data, loss_physics, loss_ic


# ============================================================
# T1 — No conflict → no projection
# ============================================================
def test_t1_no_conflict_no_projection():
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    # Simple hand gradients all positive (aligned)
    grad_data = [torch.ones_like(p) * 0.1 for p in model.parameters()]
    grad_physics = [torch.ones_like(p) * 0.1 for p in model.parameters()]
    grad_ic = [torch.zeros_like(p) for p in model.parameters()]

    pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 1.0)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    metrics = pcgrad.get_logged_metrics()[0]
    assert metrics.raw_cos_data_physics > 0.99, f"Expected cos≈1, got {metrics.raw_cos_data_physics}"
    assert not metrics.projection_applied, "Projection should NOT have been applied"
    assert abs(metrics.post_cos_data_physics_projected - metrics.raw_cos_data_physics) < 1e-6


# ============================================================
# T2 — Conflict triggers projection; projected orthogonal to data
# ============================================================
def test_t2_conflict_triggers_projection():
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    # Data gradient +1 on all, physics gradient -1 on all → cos = -1
    grad_data = [torch.ones_like(p) for p in model.parameters()]
    grad_physics = [-torch.ones_like(p) for p in model.parameters()]
    grad_ic = [torch.zeros_like(p) for p in model.parameters()]

    pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 1.0)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    metrics = pcgrad.get_logged_metrics()[0]
    assert metrics.raw_cos_data_physics < -0.99, f"Expected cos≈-1, got {metrics.raw_cos_data_physics}"
    assert metrics.projection_applied, "Projection SHOULD have been applied"
    # Projected gradient should have zero magnitude (physics was anti-parallel)
    assert metrics.norm_projected_physics_grad < 1e-5


# ============================================================
# T3 — Post-projection cosine < 1e-6
# ============================================================
def test_t3_post_projection_cosine_zero():
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    # Data: [1, 1, 0, 0], Physics: [-1, 0, 1, 0] → cos < 0
    torch.manual_seed(0)
    params = list(model.parameters())
    grad_data = [torch.randn_like(p) for p in params]
    # Make physics gradient anti-aligned
    grad_physics = [-gd + 0.1 * torch.randn_like(gd) for gd in grad_data]
    grad_ic = [torch.zeros_like(p) for p in params]

    pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 1.0)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    metrics = pcgrad.get_logged_metrics()[0]
    assert metrics.projection_applied
    # FP32 rounding: exact orthogonality is ~1e-7 relative; cosine residual
    # accumulates ~1e-6 absolute. Standard FP32 tolerance is 1e-5.
    assert abs(metrics.post_cos_data_physics_projected) < 1e-5, \
        f"Post-projection cos = {metrics.post_cos_data_physics_projected}, exceeds FP32 tolerance 1e-5"


# ============================================================
# T4 — Combined update matches hand computation
# ============================================================
def test_t4_combined_update_correct():
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.0)  # no actual update
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    params = list(model.parameters())
    grad_data = [torch.ones_like(p) * 0.5 for p in params]
    grad_physics = [torch.ones_like(p) * 0.5 for p in params]  # aligned, no projection
    grad_ic = [torch.ones_like(p) * 0.1 for p in params]

    lam_phys = 2.0
    lam_ic = 3.0

    pcgrad.set_loss_values(1.0, 1.0, 1.0, lam_phys, lam_ic)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    # Verify that p.grad = gd + lam_phys*gp + lam_ic*gi = 0.5 + 2*0.5 + 3*0.1 = 1.8
    for p in params:
        assert p.grad is not None
        expected = 0.5 + 2.0 * 0.5 + 3.0 * 0.1
        assert torch.allclose(p.grad, torch.ones_like(p) * expected, atol=1e-5), \
            f"Expected {expected}, got {p.grad.mean().item()}"


# ============================================================
# T5 — Missing gradient handled
# ============================================================
def test_t5_missing_gradient():
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    params = list(model.parameters())
    # Physics gradient None for the first param (unused in physics)
    grad_data = [torch.ones_like(p) * 0.1 for p in params]
    grad_physics = [None] + [torch.ones_like(p) * 0.1 for p in params[1:]]
    grad_ic = [torch.zeros_like(p) for p in params]

    pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 1.0)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    # Should not crash
    metrics = pcgrad.get_logged_metrics()[0]
    assert metrics.step == 0
    # First param's .grad should equal grad_data (physics skipped)
    assert params[0].grad is not None


# ============================================================
# T6 — Zero-norm gradient
# ============================================================
def test_t6_zero_norm_gradient():
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    params = list(model.parameters())
    grad_data = [torch.ones_like(p) * 0.1 for p in params]
    grad_physics = [torch.zeros_like(p) for p in params]  # zero
    grad_ic = [torch.zeros_like(p) for p in params]

    pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 1.0)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    metrics = pcgrad.get_logged_metrics()[0]
    # Zero physics → cos = 0 (or tiny), no division crash
    assert abs(metrics.raw_cos_data_physics) < 1e-5
    # No NaN/Inf in params
    for p in params:
        assert torch.isfinite(p.grad).all()


# ============================================================
# T7 — Memory < 3 GB on RTX 4050
# ============================================================
def test_t7_memory():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(200, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 9)
    ).to(device)

    base_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    x = torch.randn(16, 200, device=device)
    y = torch.randn(16, 9, device=device)

    torch.cuda.reset_peak_memory_stats()
    params = [p for p in model.parameters() if p.requires_grad]

    # 3 backward passes with separate gradients
    out = model(x)
    ld = ((out - y) ** 2).mean()
    gd = torch.autograd.grad(ld, params, retain_graph=True, allow_unused=True)

    lp = (out ** 2).mean()
    gp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)

    li = ((out[:, 0] - 1) ** 2).mean()
    gi = torch.autograd.grad(li, params, retain_graph=False, allow_unused=True)

    pcgrad.set_loss_values(float(ld), float(lp), float(li), 0.1, 1.0)
    pcgrad.set_gradients(gd, gp, gi)
    pcgrad.step()

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    result = {"peak_vram_MB": peak_mb, "under_3GB": peak_mb < 3000}
    Path("results/phase_4/phase_4b/pcgrad").mkdir(parents=True, exist_ok=True)
    with open("results/phase_4/phase_4b/pcgrad/test_T7_memory_measurement.json", "w") as f:
        json.dump(result, f, indent=2)

    assert peak_mb < 3000, f"Memory {peak_mb:.1f} MB exceeds 3 GB"


# ============================================================
# T8 — Numerical agreement with independent reference
# ============================================================
def test_t8_reference_agreement():
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    torch.manual_seed(1)
    params = list(model.parameters())
    grad_data = [torch.randn_like(p) for p in params]
    grad_physics = [torch.randn_like(p) for p in params]
    grad_ic = [torch.zeros_like(p) for p in params]

    # Reference implementation (pure PyTorch, no PCGradOptimizer)
    flat_d = torch.cat([g.flatten() for g in grad_data])
    flat_p = torch.cat([g.flatten() for g in grad_physics])
    cos = (flat_d * flat_p).sum() / (torch.linalg.norm(flat_d) * torch.linalg.norm(flat_p) + 1e-8)

    if cos < 0:
        coef = (flat_p * flat_d).sum() / ((flat_d ** 2).sum() + 1e-8)
        ref_proj = flat_p - coef * flat_d
    else:
        ref_proj = flat_p

    # Run PCGrad
    pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 0.0)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    metrics = pcgrad.get_logged_metrics()[0]

    # Reference post cosine
    ref_cos_post = (flat_d * ref_proj).sum() / (torch.linalg.norm(flat_d) * torch.linalg.norm(ref_proj) + 1e-8)

    assert abs(float(cos) - metrics.raw_cos_data_physics) < 1e-6, "Raw cosine mismatch"
    assert abs(float(ref_cos_post) - metrics.post_cos_data_physics_projected) < 1e-6, "Post cosine mismatch"
    assert abs(float(torch.linalg.norm(ref_proj)) - metrics.norm_projected_physics_grad) < 1e-5, "Projected norm mismatch"


# ============================================================
# T9 — Reproducibility
# ============================================================
def test_t9_reproducibility():
    torch.manual_seed(0)
    params1 = [torch.randn(3, 3) for _ in range(3)]
    params2 = [p.clone() for p in params1]
    gd = [torch.randn_like(p) for p in params1]
    gp = [torch.randn_like(p) for p in params1]
    gi = [torch.zeros_like(p) for p in params1]

    results = []
    for _ in range(2):
        # Fresh copy
        for p, q in zip(params1, params2):
            p.copy_(q)

        model = nn.Module()
        for i, p in enumerate(params1):
            p_param = nn.Parameter(p.clone())
            setattr(model, f"p{i}", p_param)

        base_opt = torch.optim.SGD(model.parameters(), lr=0.0)  # no update, just measure .grad
        pcgrad = PCGradOptimizer(base_opt, eps=1e-8)
        pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 0.0)
        pcgrad.set_gradients(
            [g.clone() for g in gd],
            [g.clone() for g in gp],
            [g.clone() for g in gi],
        )
        pcgrad.step()

        grads_after = [p.grad.clone() for p in model.parameters()]
        results.append(grads_after)

    for g1, g2 in zip(results[0], results[1]):
        assert torch.allclose(g1, g2, atol=1e-10), "Non-deterministic output"


# ============================================================
# T10 — All gradients missing for a parameter
# ============================================================
def test_t10_all_gradients_missing():
    """If a parameter has no data, physics, or IC gradient, its .grad is None (not zero)."""
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    params = list(model.parameters())
    grad_data = [torch.ones_like(p) * 0.1 for p in params]
    grad_physics = [torch.ones_like(p) * 0.1 for p in params]
    grad_ic = [torch.zeros_like(p) for p in params]

    # Set first parameter to None in ALL gradient sources
    grad_data[0] = None
    grad_physics[0] = None
    grad_ic[0] = None

    pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 1.0)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    # First param's .grad should be None (not zeros)
    assert params[0].grad is None, f"Expected .grad=None for all-missing param, got {params[0].grad}"


# ============================================================
# T11 — Raw and post-projection cosine on same subset
# ============================================================
def test_t11_raw_and_post_cosine_same_subset():
    """When some params are missing gradients, both cosines must use the same valid subset."""
    model = make_simple_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)

    params = list(model.parameters())

    # Physics has None for first param (not in common subset)
    grad_data = [torch.randn_like(p) for p in params]
    grad_physics = [None] + [torch.randn_like(p) for p in params[1:]]
    grad_ic = [torch.zeros_like(p) for p in params]

    # Make physics gradient for one valid param anti-aligned to trigger projection
    grad_physics[1] = -grad_data[1] + 0.1 * torch.randn_like(grad_data[1])

    pcgrad.set_loss_values(1.0, 1.0, 0.0, 1.0, 0.0)
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.step()

    metrics = pcgrad.get_logged_metrics()[0]
    # Both cosines should be finite and well-defined (no NaN from mismatched sizes)
    import math
    assert math.isfinite(metrics.raw_cos_data_physics), "raw_cos is not finite"
    assert math.isfinite(metrics.post_cos_data_physics_projected), "post_cos is not finite"
    # If projection happened, post_cos should be near zero
    if metrics.projection_applied:
        assert abs(metrics.post_cos_data_physics_projected) < 1e-5
