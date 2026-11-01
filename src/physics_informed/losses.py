"""Step 3.3 — Composite physics-informed loss.

Three losses with separated paths (frozen architecture, Step 3.2 v2.1 addendum):

  Data loss     : u_data = fno(rho_ext)   -> projection -> 9 states at 200 grid
                  Trains: lifting, fno_blocks, projection
  Physics loss  : u_phys = decoder(z, t_q), z = pre-projection latent
                  Trains: lifting, fno_blocks, decoder
  IC loss       : u_IC = decoder(z, t=0)
                  Trains: lifting, fno_blocks, decoder

Shared params: lifting + fno_blocks (36 tensors) — where physics informs data.

No .detach() or .no_grad() in forward path.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.derivatives import SCFNOModel, compute_du_dt
from physics_informed.residuals import compute_residuals


@dataclass
class LossOutput:
    data_loss:     torch.Tensor
    physics_loss:  torch.Tensor
    ic_loss:       torch.Tensor
    total_loss:    torch.Tensor
    diagnostics:   dict


class CompositeLoss(nn.Module):
    """Composite physics-informed loss for SC-FNO training.

    Args:
        lam_data:      weight on data loss (default 1.0)
        lam_phys:      weight on physics loss (default 1e-4 — ramped up during training)
        lam_IC:        weight on IC loss (default 1.0)
        residual_scales: dict {state: {"residual_p95": float, ...}} from per_state_scaling.json
        state_scales:    dict {state: {"normalization_std": float, ...}} from same
    """

    def __init__(
        self,
        lam_data: float = 1.0,
        lam_phys: float = 1e-4,
        lam_IC: float = 1.0,
        residual_scales: Optional[dict] = None,
        state_scales: Optional[dict] = None,
    ):
        super().__init__()
        self.lam_data = float(lam_data)
        self.lam_phys = float(lam_phys)
        self.lam_IC = float(lam_IC)
        self.residual_scales = residual_scales
        self.state_scales = state_scales
        self.STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

    # -------- data loss: fno(rho) -> projection -> 9 states at 200 grid --------
    def compute_data_loss(
        self,
        model: SCFNOModel,
        rho_ext: torch.Tensor,       # [B, 1, 200] normalized
        u_gt: torch.Tensor,          # [B, 9, 200] normalized
    ) -> torch.Tensor:
        """MSE on the FNO forward (with projection). Does NOT use decoder."""
        u_data = model.fno(rho_ext)  # [B, 9, 200]
        return F.mse_loss(u_data, u_gt)

    # -------- physics loss: decoder(z, t_q) -> residuals -> sum of squares --------
    def compute_physics_loss(
        self,
        model: SCFNOModel,
        rho_ext: torch.Tensor,       # [B, 1, 200] normalized
        t_q: torch.Tensor,           # [B, Nq] physical seconds
        rho_ext_fn,                  # callable for residual: t -> rho_ext(t)
        kin: KineticsParameters,
        th: THParameters,
        q_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scaled residual sum of squares over Nq query times."""
        z, u_hat, du_hat_dt = compute_du_dt(model, rho_ext, t_q)
        # u_hat: [B, 9, Nq], du_hat_dt: [B, 9, Nq]
        # Denominator normalization: u_hat is normalized. De-normalize before residual.
        u_phys = self._denorm_batch(u_hat) if self.state_scales else u_hat

        out = compute_residuals(
            u_hat=u_phys,
            du_hat_dt=du_hat_dt,  # NOTE: units — see scaling note below
            t_query=t_q,
            rho_ext_fn=rho_ext_fn,
            kin=kin, th=th, q_t=q_t,
        )
        # Residuals: R_n [B,Nq], R_C [B,6,Nq], R_Tf [B,Nq], R_Tc [B,Nq]
        stacked = out.stacked()  # [B, 9, Nq] in [n,C1..C6,Tf,Tc] order

        total = u_hat.new_zeros(())
        for j, state in enumerate(self.STATES):
            s_j = self.residual_scales[state]["residual_p95"]
            R_j = stacked[:, j, :] / s_j
            total = total + (R_j ** 2).mean()
        return total / len(self.STATES)

    # -------- IC loss: decoder(z, t=0) vs ground truth at t=0 --------
    def compute_ic_loss(
        self,
        model: SCFNOModel,
        rho_ext: torch.Tensor,       # [B, 1, 200] normalized
        u_gt_t0: torch.Tensor,       # [B, 9] normalized — first timestep of GT
    ) -> torch.Tensor:
        """MSE on decoder(z, t=0) vs u_gt at t=0, per-state scaled."""
        z = model._fno_latent(rho_ext)               # [B, 32, 200]
        t0 = torch.zeros(rho_ext.shape[0], 1, device=rho_ext.device, dtype=rho_ext.dtype)
        u_ic = model.decoder(z, t0)                  # [B, 9, 1]
        u_ic = u_ic[:, :, 0]                         # [B, 9]
        # Per-state scaled MSE
        total = u_ic.new_zeros(())
        for j, state in enumerate(self.STATES):
            s_uj = self.state_scales[state]["normalization_std"]
            diff = (u_ic[:, j] - u_gt_t0[:, j]) / s_uj
            total = total + (diff ** 2).mean()
        return total / len(self.STATES)

    # -------- combined --------
    def forward(
        self,
        model: SCFNOModel,
        rho_ext: torch.Tensor,
        u_gt: torch.Tensor,
        t_q: torch.Tensor,
        rho_ext_fn,
        kin: KineticsParameters,
        th: THParameters,
        q_t: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        u_gt_t0 = u_gt[:, :, 0]                      # [B, 9]
        L_data = self.compute_data_loss(model, rho_ext, u_gt)

        # Short-circuit physics/IC when their weights are zero (P-FNO variant):
        # avoids invoking physics-path code when it's not part of the loss.
        if self.lam_phys > 0:
            L_phys = self.compute_physics_loss(
                model, rho_ext, t_q, rho_ext_fn, kin, th, q_t
            )
        else:
            L_phys = L_data.new_zeros(())

        if self.lam_IC > 0:
            L_IC = self.compute_ic_loss(model, rho_ext, u_gt_t0)
        else:
            L_IC = L_data.new_zeros(())
        L_total = self.lam_data * L_data + self.lam_phys * L_phys + self.lam_IC * L_IC
        diag = {
            "data_loss":      float(L_data.item()),
            "physics_loss":   float(L_phys.item()),
            "ic_loss":        float(L_IC.item()),
            "total_loss":     float(L_total.item()),
            "lam_data":       self.lam_data,
            "lam_phys":       self.lam_phys,
            "lam_IC":         self.lam_IC,
        }
        return LossOutput(L_data, L_phys, L_IC, L_total, diag)

    # -------- helpers --------
    def _denorm_batch(self, u_norm: torch.Tensor) -> torch.Tensor:
        """Undo per-state normalization. u_norm: [B, 9, Nq] normalized."""
        out = u_norm.clone()
        for j, state in enumerate(self.STATES):
            mean = self.state_scales[state]["normalization_mean"]
            std  = self.state_scales[state]["normalization_std"]
            out[:, j, :] = u_norm[:, j, :] * std + mean
        return out


def load_composite_loss(
    per_state_scaling_path: str = "results/phase_3/loss_integration/per_state_scaling.json",
    lam_data: float = 1.0, lam_phys: float = 1e-4, lam_IC: float = 1.0,
) -> CompositeLoss:
    """Load CompositeLoss with scales from Step 3.3.A."""
    import json
    from pathlib import Path
    scaling = json.loads(Path(per_state_scaling_path).read_text())
    return CompositeLoss(
        lam_data=lam_data, lam_phys=lam_phys, lam_IC=lam_IC,
        residual_scales=scaling["per_state"],
        state_scales=scaling["per_state"],
    )
