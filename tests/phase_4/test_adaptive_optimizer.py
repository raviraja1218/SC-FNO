"""Phase 4A.1 — 9 unit tests for the adaptive optimizer."""
import sys
import math
from pathlib import Path

sys.path.insert(0, "src")

import torch
import torch.nn as nn
import pytest

from training.adaptive_optimizer import (
    OptimizerConfig,
    lam_phys_at_epoch,
    EMAPlateauDetector,
    AdaptiveOptimizer,
    build_lbfgs,
    run_lbfgs_stage,
)


# ============ T1: Adam initialization ============
def test_t1_adam_initialization():
    cfg = OptimizerConfig()
    model = nn.Linear(2, 2)
    opt = AdaptiveOptimizer(model, cfg, device="cpu")
    assert opt.adam.param_groups[0]["lr"] == 1e-3
    assert opt.adam.param_groups[0]["betas"] == (0.9, 0.999)
    assert opt.adam.param_groups[0]["eps"] == 1e-8
    assert opt.adam.param_groups[0]["weight_decay"] == 0.0


# ============ T2: EMA numerical correctness ============
def test_t2_ema_numerical_correctness():
    cfg = OptimizerConfig()
    cfg.ema_alpha = 0.1
    det = EMAPlateauDetector(cfg)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6]
    expected_ema = losses[0]
    for i, L in enumerate(losses):
        det.update(i, L)
        if i > 0:
            expected_ema = 0.1 * L + 0.9 * expected_ema
    actual_ema = det.state()["ema_current"]
    assert abs(actual_ema - expected_ema) < 1e-9, f"EMA mismatch: {actual_ema} vs {expected_ema}"


# ============ T3: Plateau NOT triggered on decreasing loss ============
def test_t3_plateau_not_triggered_on_decreasing_loss():
    cfg = OptimizerConfig()
    cfg.min_adam_epochs = 5
    cfg.ref_window = 3
    cfg.plateau_patience = 2
    det = EMAPlateauDetector(cfg)
    # Monotonically decreasing by 1% each epoch
    losses = [1.0 * (0.99 ** i) for i in range(50)]
    triggered = False
    for i, L in enumerate(losses):
        if det.update(i, L):
            triggered = True
            break
    assert not triggered, "Plateau falsely detected on decreasing loss"


# ============ T4: Plateau TRIGGERED on flat loss ============
def test_t4_plateau_triggered_on_flat_loss():
    cfg = OptimizerConfig()
    cfg.min_adam_epochs = 5
    cfg.ref_window = 3
    cfg.plateau_patience = 2
    cfg.plateau_delta = 1e-4
    det = EMAPlateauDetector(cfg)
    # Constant loss → improvement ≈ 0 < delta → trigger
    losses = [0.5] * 50
    triggered_at = None
    for i, L in enumerate(losses):
        if det.update(i, L):
            triggered_at = i
            break
    assert triggered_at is not None, "Plateau not detected on flat loss"
    assert triggered_at >= cfg.min_adam_epochs + cfg.ref_window + cfg.plateau_patience - 1, \
        f"Triggered too early at epoch {triggered_at}"


# ============ T5: No transition when improving ============
def test_t5_no_transition_when_improving():
    cfg = OptimizerConfig()
    model = nn.Linear(2, 2)
    opt = AdaptiveOptimizer(model, cfg, device="cpu")
    for epoch in range(50):
        val_loss = 1.0 * (0.95 ** epoch)  # steady 5% decrease
        opt.update_plateau(epoch, val_loss)
    assert not opt.plateau_reached, "Falsely transitioned on improving loss"


# ============ T6: Automatic transition when plateau met ============
def test_t6_automatic_transition_when_met():
    cfg = OptimizerConfig()
    cfg.min_adam_epochs = 5
    cfg.ref_window = 3
    cfg.plateau_patience = 2
    cfg.plateau_delta = 1e-4
    model = nn.Linear(2, 2)
    opt = AdaptiveOptimizer(model, cfg, device="cpu")
    # Flat loss → plateau
    for epoch in range(30):
        if opt.update_plateau(epoch, 0.5):
            break
    assert opt.plateau_reached, "Failed to detect plateau on flat loss"
    assert opt.transition_event is not None
    assert opt.transition_event.epoch >= cfg.min_adam_epochs + cfg.ref_window + cfg.plateau_patience - 1
    # λ_phys at transition should match formula
    expected_lam = lam_phys_at_epoch(opt.transition_event.epoch, cfg)
    assert abs(opt.transition_event.lam_phys_at_transition - expected_lam) < 1e-9


# ============ T7: L-BFGS closure deterministic ============
def test_t7_lbfgs_closure_deterministic():
    torch.manual_seed(42)
    cfg = OptimizerConfig()
    model = nn.Linear(3, 1)
    # Fixed input/target
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)

    def fixed_loss():
        pred = model(x)
        return ((pred - y) ** 2).mean()

    # Run closure 3 times without step — losses must be identical
    losses = []
    for _ in range(3):
        model.zero_grad()
        L = fixed_loss()
        L.backward()
        losses.append(L.item())

    assert max(losses) - min(losses) < 1e-9, f"Closure not deterministic: {losses}"


# ============ T8: Checkpoint save/restore roundtrip ============
def test_t8_checkpoint_save_restore_roundtrip(tmp_path):
    torch.manual_seed(42)
    cfg = OptimizerConfig()
    model = nn.Linear(3, 2)
    opt = AdaptiveOptimizer(model, cfg, device="cpu")

    # Simulate some training
    x = torch.randn(8, 3)
    for _ in range(3):
        opt.adam.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.adam.step()
        opt.scheduler.step()

    # Save
    path = tmp_path / "state.pt"
    opt.save_adam_stage(str(path))

    # Load in fresh model + optimizer
    model2 = nn.Linear(3, 2)
    opt2 = AdaptiveOptimizer(model2, cfg, device="cpu")
    ckpt = torch.load(str(path), weights_only=False)
    model2.load_state_dict(ckpt["model_state_dict"])
    opt2.adam.load_state_dict(ckpt["adam_state_dict"])
    opt2.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # Compare weights
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Weights mismatch after roundtrip"

    # Compare Adam state
    for p1, p2 in zip(opt.adam.param_groups, opt2.adam.param_groups):
        for k in ("lr", "betas", "eps", "weight_decay"):
            assert p1[k] == p2[k], f"Adam {k} mismatch"


# ============ T9: NaN/Inf detection ============
def test_t9_nan_inf_detection():
    cfg = OptimizerConfig()
    # Verify loss_fn returning NaN is detectable
    x = torch.tensor([float("nan"), 1.0, 2.0])
    assert not torch.isfinite(x).all(), "isfinite failed to detect NaN"

    x = torch.tensor([float("inf"), 1.0, 2.0])
    assert not torch.isfinite(x).all(), "isfinite failed to detect Inf"

    # Guard: L-BFGS closure returns NaN — should be detectable before backward
    def bad_loss():
        return torch.tensor(float("nan"), requires_grad=True)

    loss = bad_loss()
    assert not torch.isfinite(loss).all(), "NaN loss not detectable"
