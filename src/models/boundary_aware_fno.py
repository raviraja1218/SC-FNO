"""Unified Latent FNO — boundary-aware wrapper ensuring data and physics
paths consume the identical padded FNO latent representation.

Fixes the Phase 3 architectural flaw:
    Phase 3 _fno_latent() used raw FNO on UNPADDED input, while the data path
    used PaddedFNO on padded input. The two paths therefore consumed different
    latent tensors even when padding was nominally enabled.

This wrapper provides a SINGLE forward pass that produces ONE latent, which
both the data loss and physics residual use. Bit-identity is guaranteed by
construction.

Usage in training loop:
    latent = model.fno.get_latent(rho_ext)          # [B, 32, 200]
    u_data = model.decoder(latent, t_data)          # [B, 9, 200]
    u_phys = model.decoder(latent, t_q)             # [B, 9, Nq]

Do NOT compute separate latents for the two paths.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedLatentFNO(nn.Module):
    """Wrapper around a raw FNO that guarantees data + physics use the same latent.

    Attributes:
        fno: the underlying FNO (raw, 63,634 params for Phase 3)
        pad_size: number of reflected timesteps to add on each side (frozen: 20)
        pad_mode: padding mode (frozen: 'reflect')
    """

    def __init__(self, fno, pad_size: int = 20, pad_mode: str = "reflect"):
        super().__init__()
        if pad_mode not in ("reflect", "replicate", "circular"):
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")
        self.fno = fno
        self.pad_size = pad_size
        self.pad_mode = pad_mode
        # Proxy attributes for compatibility with SCFNOModel._fno_latent hook logic
        self.projection = fno.projection if hasattr(fno, "projection") else None
        self.lifting = fno.lifting if hasattr(fno, "lifting") else None
        self.fno_blocks = fno.fno_blocks if hasattr(fno, "fno_blocks") else None

    def forward(self, rho_ext: torch.Tensor) -> torch.Tensor:
        """Data-path forward: pad → FNO → crop to 200 points.

        Args:
            rho_ext: [B, 1, 200]
        Returns:
            out: [B, 9, 200]
        """
        if rho_ext.dim() != 3 or rho_ext.shape[1] != 1 or rho_ext.shape[2] != 200:
            raise ValueError(f"Expected [B, 1, 200], got {list(rho_ext.shape)}")

        # 1. Pad
        rho_padded = F.pad(rho_ext, (self.pad_size, self.pad_size), mode=self.pad_mode)
        # [B, 1, 240]

        # 2. FNO forward on padded
        out_padded = self.fno(rho_padded)  # [B, 9, 240]

        # 3. Crop back to 200 physical points
        out = out_padded[:, :, self.pad_size:-self.pad_size]  # [B, 9, 200]
        return out

    def get_latent(self, rho_ext: torch.Tensor) -> torch.Tensor:
        """Compute the pre-projection latent, cropped to 200 physical points.

        THIS IS THE SINGLE SOURCE OF LATENT for both data and physics paths.
        Call once per batch; use the returned tensor for BOTH:
            - data path decoder evaluation
            - physics residual decoder evaluation

        Args:
            rho_ext: [B, 1, 200]
        Returns:
            latent: [B, 32, 200]
        """
        if rho_ext.dim() != 3 or rho_ext.shape[1] != 1 or rho_ext.shape[2] != 200:
            raise ValueError(f"Expected [B, 1, 200], got {list(rho_ext.shape)}")

        # 1. Pad
        rho_padded = F.pad(rho_ext, (self.pad_size, self.pad_size), mode=self.pad_mode)
        # [B, 1, 240]

        # 2. FNO forward on padded; hook projection to capture pre-projection latent
        captured = {}

        def _hook(module, inputs, output):
            captured["latent"] = inputs[0]

        hook = self.fno.projection.register_forward_hook(_hook)
        try:
            _ = self.fno(rho_padded)
        finally:
            hook.remove()

        latent_padded = captured["latent"]  # [B, 32, 240]

        # 3. Crop back to 200 physical points
        latent = latent_padded[:, :, self.pad_size:-self.pad_size]  # [B, 32, 200]
        return latent

    def data_path(self, rho_ext: torch.Tensor, decoder, t_q: torch.Tensor) -> torch.Tensor:
        """Convenience: get latent once, decode for data loss."""
        latent = self.get_latent(rho_ext)
        return decoder(latent, t_q)

    def physics_path(self, rho_ext: torch.Tensor, decoder, t_q: torch.Tensor) -> torch.Tensor:
        """Convenience: get latent once, decode for physics residual.

        WARNING: If called independently from data_path, this will run a
        SEPARATE forward pass and may produce a slightly different latent.
        For bit-identity in training, use the shared-latent pattern:
            latent = model.get_latent(rho)
            u_data = decoder(latent, t_data)
            u_phys = decoder(latent, t_q)
        This method exists for inference and diagnostics only.
        """
        latent = self.get_latent(rho_ext)
        return decoder(latent, t_q)
