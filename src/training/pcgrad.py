"""PCGrad — Projecting Conflicting Gradients.

Wraps a base optimizer and projects the physics gradient onto the orthogonal
complement of the data gradient when they conflict (cosine < 0).

Fixes applied after initial test failures:
  - _project uses exact projection (no eps in denominator, guards zero-norm)
  - _flatten respects per-parameter None patterns; only common valid
    parameters contribute to cosine computation
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PCGradStepMetrics:
    step: int
    epoch: int
    raw_cos_data_physics: float
    post_cos_data_physics_projected: float
    norm_data_grad: float
    norm_physics_grad: float
    norm_projected_physics_grad: float
    projection_applied: bool
    loss_data: float
    loss_physics: float
    loss_ic: float
    lambda_phys: float
    lambda_ic: float
    wall_time_ms: float


class PCGradOptimizer:
    """Wraps a base optimizer and applies PCGrad projection per step.

    Requires explicit gradient computation via torch.autograd.grad():
        grad_data = torch.autograd.grad(L_data, params, retain_graph=True, allow_unused=True)
        grad_physics = torch.autograd.grad(L_physics, params, retain_graph=True, allow_unused=True)
        grad_ic = torch.autograd.grad(L_ic, params, retain_graph=False, allow_unused=True)
        pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
        pcgrad.step()
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, eps: float = 1e-8):
        self.base_optimizer = base_optimizer
        self.eps = eps
        self._step_counter = 0
        self._current_epoch = 0
        self.grad_history: list = []
        self._loss_values = {"data": 0.0, "physics": 0.0, "ic": 0.0,
                             "lambda_phys": 0.0, "lambda_ic": 0.0}

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def set_loss_values(self, loss_data, loss_physics, loss_ic, lambda_phys, lambda_ic):
        self._loss_values = {
            "data": float(loss_data),
            "physics": float(loss_physics),
            "ic": float(loss_ic),
            "lambda_phys": float(lambda_phys),
            "lambda_ic": float(lambda_ic),
        }

    def set_gradients(self, grad_data, grad_physics, grad_ic):
        self._grad_data = grad_data
        self._grad_physics = grad_physics
        self._grad_ic = grad_ic

    def _project_exact(self, flat_d: torch.Tensor, flat_p: torch.Tensor) -> torch.Tensor:
        """Exact projection: g_p - (g_p·g_d)/(||g_d||²) · g_d.

        Guard against ||g_d||=0 by returning g_p unchanged if norm is tiny.
        """
        norm_sq = (flat_d ** 2).sum()
        if norm_sq.item() < self.eps:
            return flat_p
        coef = (flat_p * flat_d).sum() / norm_sq
        return flat_p - coef * flat_d

    def _unflatten_to_subset(self, flat_tensor, subset_ref_grads):
        """Un-flatten into a list matching subset_ref_grads (all non-None)."""
        result = []
        idx = 0
        for g in subset_ref_grads:
            n = g.numel()
            result.append(flat_tensor[idx:idx + n].reshape(g.shape).clone())
            idx += n
        return result

    def _param_index_map(self):
        if not hasattr(self, "_param_to_idx"):
            self._param_to_idx = {}
            for group in self.base_optimizer.param_groups:
                for param in group["params"]:
                    self._param_to_idx[id(param)] = len(self._param_to_idx)
        return self._param_to_idx

    def step(self):
        t_start = time.perf_counter()

        if not hasattr(self, "_grad_data"):
            raise RuntimeError(
                "PCGradOptimizer.step() called before set_gradients(). "
                "Compute gradients via torch.autograd.grad() and pass them in."
            )

        grad_data = self._grad_data
        grad_physics = self._grad_physics
        grad_ic = self._grad_ic
        n = len(grad_data)

        # Identify common valid parameter indices (both data + physics present)
        valid = [i for i in range(n)
                 if grad_data[i] is not None and grad_physics[i] is not None]

        if valid:
            flat_d = torch.cat([grad_data[i].flatten() for i in valid])
            flat_p = torch.cat([grad_physics[i].flatten() for i in valid])
            norm_d = torch.linalg.norm(flat_d)
            norm_p = torch.linalg.norm(flat_p)
            raw_cos = float((flat_d * flat_p).sum() / (norm_d * norm_p + self.eps))
            projection_applied = raw_cos < 0.0

            if projection_applied:
                flat_p_proj = self._project_exact(flat_d, flat_p)
                proj_subset = self._unflatten_to_subset(
                    flat_p_proj, [grad_physics[i] for i in valid]
                )
                # Rebuild full grad_physics_proj with None for non-common
                grad_physics_proj = [None] * n
                for j, i in enumerate(valid):
                    grad_physics_proj[i] = proj_subset[j]
            else:
                grad_physics_proj = list(grad_physics)

            # Post-projection cosine (on common subset)
            flat_p_proj_common = torch.cat([grad_physics_proj[i].flatten() for i in valid])
            norm_p_proj = torch.linalg.norm(flat_p_proj_common)
            post_cos = float((flat_d * flat_p_proj_common).sum() / (norm_d * norm_p_proj + self.eps))
        else:
            raw_cos = 0.0
            post_cos = 0.0
            projection_applied = False
            grad_physics_proj = list(grad_physics)
            flat_d = None
            flat_p = None
            norm_p_proj = torch.tensor(0.0)

        # Write per-parameter total gradient
        lambda_phys = self._loss_values["lambda_phys"]
        lambda_ic = self._loss_values["lambda_ic"]
        idx_map = self._param_index_map()

        with torch.no_grad():
            for group in self.base_optimizer.param_groups:
                for p in group["params"]:
                    idx = idx_map.get(id(p))
                    if idx is None:
                        continue
                    gd = grad_data[idx] if idx < len(grad_data) else None
                    gp = grad_physics_proj[idx] if idx < len(grad_physics_proj) else None
                    gi = grad_ic[idx] if idx < len(grad_ic) else None
                    total = torch.zeros_like(p)
                    if gd is not None:
                        total = total + gd
                    if gp is not None:
                        total = total + lambda_phys * gp
                    if gi is not None:
                        total = total + lambda_ic * gi
                    p.grad = total

        self.base_optimizer.step()

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        metrics = PCGradStepMetrics(
            step=self._step_counter,
            epoch=self._current_epoch,
            raw_cos_data_physics=raw_cos,
            post_cos_data_physics_projected=post_cos,
            norm_data_grad=float(torch.linalg.norm(flat_d)) if flat_d is not None else 0.0,
            norm_physics_grad=float(torch.linalg.norm(flat_p)) if flat_p is not None else 0.0,
            norm_projected_physics_grad=float(norm_p_proj),
            projection_applied=projection_applied,
            loss_data=self._loss_values["data"],
            loss_physics=self._loss_values["physics"],
            loss_ic=self._loss_values["ic"],
            lambda_phys=lambda_phys,
            lambda_ic=lambda_ic,
            wall_time_ms=wall_ms,
        )
        self.grad_history.append(metrics)
        self._step_counter += 1

    def get_logged_metrics(self):
        return self.grad_history

    def get_summary(self):
        if not self.grad_history:
            return {"n_steps_total": 0}
        raw_cos = [m.raw_cos_data_physics for m in self.grad_history]
        post_cos = [m.post_cos_data_physics_projected for m in self.grad_history]
        n_proj = sum(1 for m in self.grad_history if m.projection_applied)
        return {
            "n_steps_total": len(self.grad_history),
            "n_steps_with_projection": n_proj,
            "projection_fraction": n_proj / len(self.grad_history),
            "raw_cosine_mean": float(sum(raw_cos) / len(raw_cos)),
            "raw_cosine_min": float(min(raw_cos)),
            "raw_cosine_max": float(max(raw_cos)),
            "post_cosine_mean": float(sum(post_cos) / len(post_cos)),
            "post_cosine_min": float(min(post_cos)),
            "post_cosine_max": float(max(post_cos)),
        }
