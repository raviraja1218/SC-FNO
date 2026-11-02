"""Unified Latent FNO — boundary-aware wrapper for Phase 4B.

Design (fixed 2026-11-02):
    - Data path: pad → FNO(projection) → crop to 200 (Phase 3 accuracy preserved)
    - Physics path: pad → FNO(latent capture) → crop latent to 200 (padded context)
    - Both paths use the SAME padded FNO forward; only the head differs.
    - "Unified" means shared latent representation, not shared head.

    This preserves Phase 3 data-path accuracy (projection MLP unchanged)
    while giving the physics path padded spectral context.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedLatentFNO(nn.Module):
    """Wrapper around raw FNO providing:
        - forward(rho) → data path: [B, 9, 200] via projection MLP (Phase 3 style)
        - get_latent(rho) → physics path: [B, 32, 200] pre-projection latent, cropped

    Both paths run through the SAME padded FNO, so spectral context is identical.
    """

    def __init__(self, fno, pad_size: int = 20, pad_mode: str = "reflect"):
        super().__init__()
        if pad_mode not in ("reflect", "replicate", "circular"):
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")
        self.fno = fno
        self.pad_size = pad_size
        self.pad_mode = pad_mode
        # Proxy attributes for compatibility with SCFNOModel._fno_latent hook logic
        self.projection = getattr(fno, "projection", None)
        self.lifting = getattr(fno, "lifting", None)
        self.fno_blocks = getattr(fno, "fno_blocks", None)

    def forward(self, rho_ext: torch.Tensor) -> torch.Tensor:
        """Data path: pad → FNO (with projection MLP) → crop to 200.

        Args:
            rho_ext: [B, 1, 200]
        Returns:
            out: [B, 9, 200] — projected states via the trained MLP
        """
        if rho_ext.dim() != 3 or rho_ext.shape[1] != 1 or rho_ext.shape[2] != 200:
            raise ValueError(f"Expected [B, 1, 200], got {list(rho_ext.shape)}")
        rho_padded = F.pad(rho_ext, (self.pad_size, self.pad_size), mode=self.pad_mode)
        out_padded = self.fno(rho_padded)               # [B, 9, 240]
        out = out_padded[:, :, self.pad_size:-self.pad_size]   # [B, 9, 200]
        return out

    def get_latent(self, rho_ext: torch.Tensor) -> torch.Tensor:
        """Physics path: pad → FNO (capture pre-projection latent) → crop to 200.

        Args:
            rho_ext: [B, 1, 200]
        Returns:
            latent: [B, 32, 200]
        """
        if rho_ext.dim() != 3 or rho_ext.shape[1] != 1 or rho_ext.shape[2] != 200:
            raise ValueError(f"Expected [B, 1, 200], got {list(rho_ext.shape)}")
        rho_padded = F.pad(rho_ext, (self.pad_size, self.pad_size), mode=self.pad_mode)
        captured = {}
        def _hook(module, inputs, output):
            captured["latent"] = inputs[0]
        hook = self.fno.projection.register_forward_hook(_hook)
        try:
            _ = self.fno(rho_padded)
        finally:
            hook.remove()
        latent_padded = captured["latent"]              # [B, 32, 240]
        latent = latent_padded[:, :, self.pad_size:-self.pad_size]  # [B, 32, 200]
        return latent

    def data_path(self, rho_ext, decoder=None, t_q=None):
        """Alias for forward() — data path."""
        return self.forward(rho_ext)

    def physics_path(self, rho_ext, decoder, t_q):
        """Physics path: get_latent + decoder."""
        latent = self.get_latent(rho_ext)
        return decoder(latent, t_q)
