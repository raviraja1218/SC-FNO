"""PCGrad — Projecting Conflicting Gradients.

Wraps a base optimizer and projects the physics gradient onto the orthogonal
complement of the data gradient when they conflict (cosine < 0).

Handles COMPLEX gradients (from FFT layers in the FNO) via Wirtinger inner
product: Re(conj(a) * b).

Frozen config (Phase 4B Step 4B.0 pre-registration):
    eps = 1e-8
    conflict trigger: cos(g_d, g_p) < 0
    log raw and post-projection cosine

Usage:
    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pcgrad = PCGradOptimizer(base_opt, eps=1e-8)
    # Compute separate gradients via torch.autograd.grad()
    pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
    pcgrad.set_loss_values(L_data, L_physics, L_ic, lam_phys, lam_ic)
    pcgrad.step()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

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


# ---------------- Complex-aware vector helpers ----------------

def _flat_and_norm(grads):
    """Flatten a list of (possibly complex) gradient tensors into one vector.

    Returns:
        flat_vec: 1-D tensor in FP64 (if all real) or complex128 (if any complex)
        norm_sq: scalar norm² as real FP64
    """
    has_complex = any(g is not None and g.is_complex() for g in grads)
    parts = []
    for g in grads:
        if g is None:
            continue
        if has_complex:
            if g.is_complex():
                parts.append(g.reshape(-1).to(torch.complex128))
            else:
                parts.append(g.reshape(-1).to(torch.float64).to(torch.complex128))
        else:
            parts.append(g.reshape(-1).to(torch.float64))
    if not parts:
        return None, None
    flat = torch.cat(parts)
    if has_complex:
        norm_sq = torch.real((flat.conj() * flat).sum())
    else:
        norm_sq = (flat * flat).sum()
    return flat, norm_sq


def _real_inner(a, b):
    """Real inner product. Handles complex via Re(conj(a)·b)."""
    if a.is_complex() or b.is_complex():
        return torch.real((a.conj() * b).sum())
    return (a * b).sum()


def _norm_from_flat(flat):
    """Real norm of flat vector (complex or real)."""
    if flat.is_complex():
        return torch.sqrt(torch.real((flat.conj() * flat).sum()))
    return torch.sqrt((flat * flat).sum())


class PCGradOptimizer:
    """PCGrad wrapper with complex-gradient support.

    Requires explicit gradient computation via torch.autograd.grad():
        grad_data = torch.autograd.grad(L_data, params, retain_graph=True, allow_unused=True)
        grad_physics = torch.autograd.grad(L_physics, params, retain_graph=True, allow_unused=True)
        grad_ic = torch.autograd.grad(L_ic, params, retain_graph=False, allow_unused=True)
        pcgrad.set_gradients(grad_data, grad_physics, grad_ic)
        pcgrad.set_loss_values(float(L_data), float(L_physics), float(L_ic), lam_phys, lam_ic)
        pcgrad.step()
    """

    def __init__(self, base_optimizer, eps: float = 1e-8):
        self.base_optimizer = base_optimizer
        self.eps = eps
        self._step_counter = 0
        self._current_epoch = 0
        self.grad_history = []
        self._loss_values = {"data": 0.0, "physics": 0.0, "ic": 0.0,
                             "lambda_phys": 0.0, "lambda_ic": 0.0}
        self._grad_data = None
        self._grad_physics = None
        self._grad_ic = None

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def set_loss_values(self, loss_data, loss_physics, loss_ic, lambda_phys, lambda_ic):
        self._loss_values = {
            "data": float(loss_data), "physics": float(loss_physics),
            "ic": float(loss_ic),
            "lambda_phys": float(lambda_phys), "lambda_ic": float(lambda_ic),
        }

    def set_gradients(self, grad_data, grad_physics, grad_ic):
        self._grad_data = grad_data
        self._grad_physics = grad_physics
        self._grad_ic = grad_ic

    def _param_index_map(self):
        if not hasattr(self, "_p2i"):
            self._p2i = {}
            for group in self.base_optimizer.param_groups:
                for p in group["params"]:
                    self._p2i[id(p)] = len(self._p2i)
        return self._p2i

    def step(self):
        t_start = time.perf_counter()

        if self._grad_data is None:
            raise RuntimeError("set_gradients() must be called before step()")

        gd_list = self._grad_data
        gp_list = self._grad_physics
        gi_list = self._grad_ic
        n_params = len(gd_list)

        # Valid = both gd and gp present
        valid = [i for i in range(n_params)
                 if gd_list[i] is not None and gp_list[i] is not None]

        lam_phys = self._loss_values["lambda_phys"]
        lam_ic = self._loss_values["lambda_ic"]

        # ---- Compute raw cosine and (optionally) projection ----
        if valid:
            gd_valid = [gd_list[i] for i in valid]
            gp_valid = [gp_list[i] for i in valid]

            flat_d, norm_d_sq = _flat_and_norm(gd_valid)
            flat_p, norm_p_sq = _flat_and_norm(gp_valid)

            norm_d = torch.sqrt(norm_d_sq)
            norm_p = torch.sqrt(norm_p_sq)
            raw_cos = float(_real_inner(flat_d, flat_p) / (norm_d * norm_p + self.eps))
            projection_applied = raw_cos < 0.0

            if projection_applied:
                # g_p* = g_p - Re(g_p^H g_d) / Re(g_d^H g_d) * g_d
                coef = _real_inner(flat_p, flat_d) / (norm_d_sq + self.eps)
                flat_p_proj = flat_p - coef * flat_d

                # Un-flatten to per-param tensors matching original shapes/dtypes
                proj_subset = []
                idx = 0
                for i in valid:
                    ref = gp_list[i]
                    numel = ref.numel()
                    chunk = flat_p_proj[idx:idx+numel]
                    if ref.is_complex():
                        chunk = chunk.to(torch.complex64)
                    else:
                        # flat_p_proj is complex128 if any param in valid is complex.
                        # For real params, imaginary part is 0j; .real discards it
                        # explicitly (suppresses "Casting complex to real" warning).
                        chunk = chunk.real.to(torch.float32) if chunk.is_complex() else chunk.to(torch.float32)
                    proj_subset.append(chunk.reshape(ref.shape))
                    idx += numel

                # Rebuild full list; non-valid entries keep original physics grad (or None)
                gp_proj_list = []
                valid_iter = iter(range(len(valid)))
                valid_pos = 0
                for i in range(n_params):
                    if i in valid:
                        gp_proj_list.append(proj_subset[valid_pos])
                        valid_pos += 1
                    else:
                        gp_proj_list.append(gp_list[i])
            else:
                gp_proj_list = gp_list

            # Post-projection cosine
            gp_proj_valid = [gp_proj_list[i] for i in valid]
            flat_p_proj_common, _ = _flat_and_norm(gp_proj_valid)
            norm_p_proj = _norm_from_flat(flat_p_proj_common)
            post_cos = float(_real_inner(flat_d, flat_p_proj_common) / (norm_d * norm_p_proj + self.eps))
        else:
            raw_cos = 0.0
            post_cos = 0.0
            projection_applied = False
            gp_proj_list = gp_list
            flat_d = None
            flat_p = None
            norm_d = torch.tensor(0.0, dtype=torch.float64)
            norm_p_proj = torch.tensor(0.0, dtype=torch.float64)

        # ---- Combine per-parameter gradients ----
        idx_map = self._param_index_map()
        with torch.no_grad():
            for group in self.base_optimizer.param_groups:
                for p in group["params"]:
                    i = idx_map.get(id(p))
                    if i is None:
                        continue
                    gd = gd_list[i] if i < len(gd_list) else None
                    gp = gp_proj_list[i] if i < len(gp_proj_list) else None
                    gi = gi_list[i] if i < len(gi_list) else None
                    has_any = False
                    total = torch.zeros_like(p)
                    if gd is not None:
                        total = total + gd.to(p.dtype)
                        has_any = True
                    if gp is not None:
                        total = total + lam_phys * gp.to(p.dtype)
                        has_any = True
                    if gi is not None:
                        total = total + lam_ic * gi.to(p.dtype)
                        has_any = True
                    p.grad = total if has_any else None

        self.base_optimizer.step()

        wall_ms = (time.perf_counter() - t_start) * 1000.0
        self.grad_history.append(PCGradStepMetrics(
            step=self._step_counter, epoch=self._current_epoch,
            raw_cos_data_physics=raw_cos,
            post_cos_data_physics_projected=post_cos,
            norm_data_grad=float(norm_d),
            norm_physics_grad=float(torch.sqrt(norm_p_sq)) if flat_p is not None else 0.0,
            norm_projected_physics_grad=float(norm_p_proj),
            projection_applied=projection_applied,
            loss_data=self._loss_values["data"],
            loss_physics=self._loss_values["physics"],
            loss_ic=self._loss_values["ic"],
            lambda_phys=lam_phys, lambda_ic=lam_ic,
            wall_time_ms=wall_ms,
        ))
        self._step_counter += 1

    def get_logged_metrics(self):
        return self.grad_history

    def get_summary(self):
        if not self.grad_history:
            return {"n_steps_total": 0}
        raw = [m.raw_cos_data_physics for m in self.grad_history]
        post = [m.post_cos_data_physics_projected for m in self.grad_history]
        n_proj = sum(1 for m in self.grad_history if m.projection_applied)
        return {
            "n_steps_total": len(self.grad_history),
            "n_steps_with_projection": n_proj,
            "projection_fraction": n_proj / len(self.grad_history),
            "raw_cosine_mean": float(sum(raw) / len(raw)),
            "raw_cosine_min": float(min(raw)),
            "raw_cosine_max": float(max(raw)),
            "post_cosine_mean": float(sum(post) / len(post)),
            "post_cosine_min": float(min(post)),
            "post_cosine_max": float(max(post)),
        }
