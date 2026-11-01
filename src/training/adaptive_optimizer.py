"""Phase 4A.1 — Adaptive Adam → L-BFGS optimizer with plateau detection.

Design (approved in Phase 4A.1 plan):
    - Adam stage: lr=1e-3, betas=(0.9, 0.999), fresh state from best_val.pt epoch 36
    - λ_phys ramp: 0.002754 → 1.0 over 64 epochs (restart counter at 0)
    - Plateau detection: EMA α=0.1, ref_window=20, δ=1e-4, patience P=3, min_epochs=20
    - Transition policy (B): inherit λ_phys at plateau; L-BFGS uses same λ
    - L-BFGS: history=50, max_iter=50000, deterministic closure on fixed batch
    - No optimizer state from Phase 3 exists (verified Phase 4A.0.R6)
"""
import math
import time
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch


@dataclass
class OptimizerConfig:
    """All hyperparameters for adaptive optimization (frozen pre-registration)."""
    # Adam stage
    adam_lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    scheduler_t_max: int = 100
    scheduler_eta_min: float = 0.0

    # λ_phys schedule
    lam_phys_init: float = 0.002754
    lam_phys_target: float = 1.0
    lam_phys_ramp_epochs: int = 64  # 0.002754 → 1.0 over 64 epochs

    # Plateau detection
    ema_alpha: float = 0.1
    ref_window: int = 20
    plateau_delta: float = 1e-4
    plateau_patience: int = 3
    min_adam_epochs: int = 20
    max_adam_epochs: int = 100

    # Transition policy
    transition_policy: str = "B_inherit_lambda"  # A_delay_to_1, B_inherit_lambda, C_force_at_cap

    # L-BFGS stage
    lbfgs_history_size: int = 50
    lbfgs_max_iter: int = 50_000
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 1e-9
    lbfgs_fixed_batch_size: int = 16

    # Misc
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def lam_phys_at_epoch(epoch: int, cfg: OptimizerConfig) -> float:
    """Log-linear ramp from lam_phys_init to lam_phys_target over lam_phys_ramp_epochs."""
    if epoch >= cfg.lam_phys_ramp_epochs:
        return cfg.lam_phys_target
    frac = epoch / cfg.lam_phys_ramp_epochs
    ratio = cfg.lam_phys_target / cfg.lam_phys_init
    return cfg.lam_phys_init * (10.0 ** (frac * math.log10(ratio)))


class EMAPlateauDetector:
    """Exponential moving average with relative-improvement plateau detection.

    Rule: at epoch e >= min_adam_epochs, compute
        I_e = (M[e-ref_window] - M[e]) / max(|M[e-ref_window]|, eps)
    If I_e < delta for `patience` consecutive checks → plateau detected.
    """

    def __init__(self, cfg: OptimizerConfig):
        self.cfg = cfg
        self.alpha = cfg.ema_alpha
        self.ref_window = cfg.ref_window
        self.delta = cfg.plateau_delta
        self.patience = cfg.plateau_patience
        self.min_epochs = cfg.min_adam_epochs
        self.eps = 1e-12

        self._history: list = []  # EMA values per epoch
        self._ema: Optional[float] = None
        self._consecutive_below = 0
        self.plateau_epoch: Optional[int] = None

    def update(self, epoch: int, loss: float) -> bool:
        """Update detector with new epoch loss. Returns True if plateau detected."""
        # Initialize EMA on first call
        if self._ema is None:
            self._ema = loss
        else:
            self._ema = self.alpha * loss + (1.0 - self.alpha) * self._ema
        self._history.append(self._ema)

        # Detection requires:
        #   (a) epoch >= min_adam_epochs (allow model to settle)
        #   (b) at least min_epochs + ref_window + 1 EMA observations
        #       so the reference window starts at or after min_epochs
        if epoch < self.min_epochs:
            return False
        if len(self._history) < self.min_epochs + self.ref_window + 1:
            return False

        # Compute relative improvement vs. ref_window epochs ago
        ref_idx = len(self._history) - 1 - self.ref_window
        m_ref = self._history[ref_idx]
        m_now = self._history[-1]
        improvement = (m_ref - m_now) / max(abs(m_ref), self.eps)

        if improvement < self.delta:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        if self._consecutive_below >= self.patience:
            self.plateau_epoch = epoch
            return True
        return False

    def state(self) -> dict:
        return {
            "ema_history": self._history,
            "ema_current": self._ema,
            "consecutive_below": self._consecutive_below,
            "plateau_epoch": self.plateau_epoch,
            "config": {
                "alpha": self.alpha,
                "ref_window": self.ref_window,
                "delta": self.delta,
                "patience": self.patience,
                "min_epochs": self.min_epochs,
            },
        }


@dataclass
class TransitionEvent:
    epoch: int
    lam_phys_at_transition: float
    ema_value: float
    improvement_last_check: float
    policy: str
    raw_val_loss: float


class AdaptiveOptimizer:
    """Orchestrates Adam → L-BFGS with plateau detection.

    Usage:
        opt = AdaptiveOptimizer(model, cfg, device="cuda")
        # Adam stage
        for epoch in range(max_epochs):
            train_one_epoch(model, opt.adam, ...)
            val_loss = evaluate(model, ...)
            opt.scheduler.step()
            if opt.update_plateau(epoch, val_loss):
                break
        # Transition
        if opt.plateau_reached:
            opt.save_adam_stage(adam_ckpt_path)
            lbfgs_loss = run_lbfgs_stage(model, opt, fixed_batch, ...)
    """

    def __init__(self, model: torch.nn.Module, cfg: OptimizerConfig, device: str = "cuda"):
        self.model = model
        self.cfg = cfg
        self.device = device

        # Fresh Adam (no Phase 3 state exists)
        self.adam = torch.optim.Adam(
            model.parameters(),
            lr=cfg.adam_lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
            weight_decay=0.0,
        )
        # Fresh scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.adam,
            T_max=cfg.scheduler_t_max,
            eta_min=cfg.scheduler_eta_min,
        )
        # Plateau detector
        self.plateau_detector = EMAPlateauDetector(cfg)
        self.transition_event: Optional[TransitionEvent] = None

    def current_lam_phys(self, epoch: int) -> float:
        return lam_phys_at_epoch(epoch, self.cfg)

    def update_plateau(self, epoch: int, val_loss: float) -> bool:
        """Feed epoch's val loss to detector. Return True if plateau reached."""
        detected = self.plateau_detector.update(epoch, val_loss)
        if detected and self.transition_event is None:
            lam_at_epoch = self.current_lam_phys(epoch)
            improvement = 0.0
            hist = self.plateau_detector.state()["ema_history"]
            if len(hist) > self.cfg.ref_window:
                ref_idx = len(hist) - 1 - self.cfg.ref_window
                improvement = (hist[ref_idx] - hist[-1]) / max(abs(hist[ref_idx]), 1e-12)
            self.transition_event = TransitionEvent(
                epoch=epoch,
                lam_phys_at_transition=lam_at_epoch,
                ema_value=self.plateau_detector.state()["ema_current"],
                improvement_last_check=improvement,
                policy=self.cfg.transition_policy,
                raw_val_loss=val_loss,
            )
        return detected

    @property
    def plateau_reached(self) -> bool:
        return self.transition_event is not None

    def save_adam_stage(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "adam_state_dict": self.adam.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "plateau_detector_state": self.plateau_detector.state(),
            "transition_event": asdict(self.transition_event) if self.transition_event else None,
            "cfg": self.cfg.to_dict(),
        }, path)

    def save_final(self, path: str, lbfgs_stats: dict):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "cfg": self.cfg.to_dict(),
            "plateau_detector_state": self.plateau_detector.state(),
            "transition_event": asdict(self.transition_event) if self.transition_event else None,
            "lbfgs_stats": lbfgs_stats,
        }, path)


def build_lbfgs(model: torch.nn.Module, cfg: OptimizerConfig) -> torch.optim.LBFGS:
    """Fresh L-BFGS with the frozen config."""
    return torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,  # L-BFGS line search controls step size
        history_size=cfg.lbfgs_history_size,
        max_iter=cfg.lbfgs_max_iter,
        tolerance_grad=cfg.lbfgs_tolerance_grad,
        tolerance_change=cfg.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )


def run_lbfgs_stage(
    model: torch.nn.Module,
    optimizer_lbfgs: torch.optim.LBFGS,
    loss_fn: Callable[[], torch.Tensor],
    lambda_phys: float,
    max_iter: int,
    log_fn: Optional[Callable] = None,
) -> dict:
    """Run deterministic L-BFGS. Returns statistics dict.

    Closure re-evaluates `loss_fn` (fixed batch, no shuffle). One `optimizer.step()`
    call performs up to `max_iter` internal iterations.
    """
    start_time = time.time()
    closure_calls = {"count": 0}
    last_loss = {"value": None}

    def closure():
        closure_calls["count"] += 1
        optimizer_lbfgs.zero_grad()
        loss = loss_fn()
        loss.backward()
        last_loss["value"] = float(loss.item())
        if log_fn is not None:
            log_fn(closure_calls["count"], last_loss["value"])
        return loss

    optimizer_lbfgs.step(closure)

    return {
        "wall_time_s": time.time() - start_time,
        "closure_calls": closure_calls["count"],
        "final_loss": last_loss["value"],
        "lambda_phys": lambda_phys,
        "max_iter": max_iter,
    }


# ============================================================================
# Phase 4A.1 CORRECTIVE ADDENDUM — correct model loading for Phase 3 checkpoint
# ============================================================================
# Phase 3's training script used `SCFNOModel(PaddedFNO(fno, pad_size), decoder)`.
# Even for PI-FNO (padding=0), the PaddedFNO wrapper was present, so the saved
# state_dict has keys with `fno.fno.*` prefix. Loading into a single-wrapped
# `SCFNOModel(fno_raw, decoder)` (which has `fno.*` prefix) silently drops all
# FNO backbone weights, leaving the model with Phase 2's FNO + Phase 3's decoder.
#
# This loader reconstructs the exact Phase 3 model structure, guaranteeing
# bit-exact weight match with the checkpoint.

def load_phase3_pi_fno_model(checkpoint_path: str, device: str = "cuda"):
    """Load a Phase 3 PI-FNO checkpoint into a fully-loaded SCFNOModel.

    Reconstructs the PaddedFNO wrapper used during Phase 3 training so the
    state_dict keys match exactly (`fno.fno.*` prefix).

    Returns:
        (model, checkpoint_dict) — model on `device`, checkpoint for metadata.

    Raises:
        RuntimeError if any state_dict key is missing or unexpected.
    """
    import torch
    from physics_informed.derivatives import load_phase2_fno, SCFNOModel
    from physics_informed.decoder import build_decoder
    from training.train_sc_fno import PaddedFNO

    # Build Phase 3 structure
    fno_raw = load_phase2_fno(device=device)
    fno_wrapped = PaddedFNO(fno_raw, pad_size=0)  # 0 = identity for PI-FNO
    decoder = build_decoder(seed=42).to(device)
    model = SCFNOModel(fno_wrapped, decoder).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict load mismatch for {checkpoint_path}:\n"
            f"  missing ({len(missing)}): {missing[:5]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:5]}\n"
            f"  This indicates the model architecture does not match the "
            f"checkpoint. Expected PaddedFNO-wrapped SCFNOModel."
        )

    return model, ckpt
