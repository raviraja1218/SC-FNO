"""Adaptive lambda_phys — cosine-driven dynamic physics weighting.

Updates lambda_phys per epoch based on smoothed gradient cosine between
data and physics gradients. Bounded in log space to prevent runaway.

Frozen configuration (Step 4B.0 pre-registration):
    alpha = 0.05
    lambda_min = 1e-5
    lambda_max = 10.0
    lambda_init = 1e-4
    ema_beta = 0.1
    update_freq = per_epoch
    clip_method = log-clip-exp

Interaction with PCGrad (P4-D):
    The adaptive lambda update uses the RAW cosine (before PCGrad projection)
    because PCGrad projection makes post-cosine near-zero, which would freeze
    lambda. See implementation report section 3.
"""
from __future__ import annotations

import math


class AdaptiveLambda:
    """Adaptive physics loss weight updated per-epoch based on gradient cosine.

    Usage:
        al = AdaptiveLambda()  # defaults match frozen config
        for epoch in range(max_epochs):
            # ... compute raw_cosine from data+physics gradients
            al.update(epoch, raw_cosine)
            lambda_phys = al.current()
            # ... train with lambda_phys
    """

    def __init__(self, alpha: float = 0.05, lambda_min: float = 1e-5,
                 lambda_max: float = 10.0, lambda_init: float = 1e-4,
                 ema_beta: float = 0.1, ema_reference_window: int = 20):
        # Validate bounds
        if not (0 < lambda_min < lambda_max):
            raise ValueError(f"Invalid bounds: lambda_min={lambda_min}, lambda_max={lambda_max}")
        if not (lambda_min <= lambda_init <= lambda_max):
            raise ValueError(f"lambda_init={lambda_init} outside [{lambda_min}, {lambda_max}]")
        if not (0 < ema_beta < 1):
            raise ValueError(f"ema_beta must be in (0, 1), got {ema_beta}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.alpha = alpha
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lam = lambda_init
        self.lambda_init = lambda_init
        self.ema_beta = ema_beta
        self.ema_reference_window = ema_reference_window
        self.smoothed_cosine = None
        self._smoothed_cosine_history = []  # for look-back
        self.history = []

        self._log_lambda_min = math.log(lambda_min)
        self._log_lambda_max = math.log(lambda_max)

    def update(self, epoch: int, raw_cosine: float) -> float:
        """Update lambda based on raw cosine. Returns new lambda.

        Clarification B: raw_cosine is clipped to [-1, 1] before use; the
        original value is preserved in history for diagnostics.
        Clarification A: 20-epoch look-back of smoothed_cosine is recorded
        for interpretability only (not used in the update formula).
        """
        if not math.isfinite(raw_cosine):
            raise ValueError(f"Non-finite raw_cosine at epoch {epoch}: {raw_cosine}")

        # Clarification B: clip to [-1, 1] defensively
        raw_val = float(raw_cosine)
        clipped_val = max(-1.0, min(1.0, raw_val))
        cosine_was_clipped = (abs(raw_val - clipped_val) > 1e-12)

        # EMA smooth (on clipped value)
        if self.smoothed_cosine is None:
            self.smoothed_cosine = clipped_val
        else:
            self.smoothed_cosine = (
                self.ema_beta * clipped_val + (1.0 - self.ema_beta) * self.smoothed_cosine
            )

        # Clarification A: record smoothed cosine history for 20-epoch look-back
        self._smoothed_cosine_history.append(self.smoothed_cosine)
        if len(self._smoothed_cosine_history) > self.ema_reference_window:
            smoothed_20_ago = self._smoothed_cosine_history[-(self.ema_reference_window + 1)]
            delta_smoothed = self.smoothed_cosine - smoothed_20_ago
        else:
            smoothed_20_ago = None
            delta_smoothed = None

        log_lam_prev = math.log(self.lam)
        log_lam_raw = log_lam_prev + self.alpha * self.smoothed_cosine

        if log_lam_raw < self._log_lambda_min:
            log_lam_final = self._log_lambda_min
            clip_applied = "min"
        elif log_lam_raw > self._log_lambda_max:
            log_lam_final = self._log_lambda_max
            clip_applied = "max"
        else:
            log_lam_final = log_lam_raw
            clip_applied = ""

        self.lam = math.exp(log_lam_final)

        self.history.append({
            "epoch": epoch,
            "raw_cosine_input": raw_val,
            "raw_cosine_used": clipped_val,
            "cosine_clipped": cosine_was_clipped,
            "smoothed_cosine": self.smoothed_cosine,
            "smoothed_cosine_20_epochs_ago": smoothed_20_ago,
            "delta_smoothed_cosine": delta_smoothed,
            "log_lambda_prev": log_lam_prev,
            "log_lambda_raw_update": log_lam_raw,
            "log_lambda_after_clip": log_lam_final,
            "lambda": self.lam,
            "clip_applied": clip_applied,
            "alpha": self.alpha,
        })
        return self.lam

    def current(self) -> float:
        return self.lam

    def get_history(self) -> list:
        return list(self.history)

    def get_summary(self) -> dict:
        if not self.history:
            return {"n_updates": 0}
        lams = [h["lambda"] for h in self.history]
        n_min = sum(1 for h in self.history if h["clip_applied"] == "min")
        n_max = sum(1 for h in self.history if h["clip_applied"] == "max")
        n_cos_clipped = sum(1 for h in self.history if h.get("cosine_clipped", False))
        return {
            "n_updates": len(self.history),
            "lambda_init": self.lambda_init,
            "lambda_final": lams[-1],
            "lambda_min_seen": min(lams),
            "lambda_max_seen": max(lams),
            "n_clip_min": n_min,
            "n_clip_max": n_max,
            "clip_fraction": (n_min + n_max) / len(self.history),
            "n_cosine_clipped": n_cos_clipped,
            "cosine_clip_fraction": n_cos_clipped / len(self.history),
            "ema_reference_window": self.ema_reference_window,
        }
