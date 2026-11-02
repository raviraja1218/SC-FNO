"""Phase 5 Corrected — SCFNOModelV2 wrapper.

Exposes three entry points:
  - forward_data(rho_ext)          : FNO projection MLP → [B, 9, 200]   (data path)
  - forward_decoder(rho_ext, t_q)  : FNO latent → decoder → [B, 9, Nq] (physics + supervised)
  - _fno_latent(rho_ext)           : pre-projection latent              (helper)
"""
import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_informed.decoder import CoordinateDecoder


class SCFNOModelV2(nn.Module):
    """Phase 5 Corrected model.

    Args:
        fno: neuralop.FNO or wrapped (UnifiedLatentFNO / PaddedFNO)
        decoder: CoordinateDecoder (Phase 3 architecture)
        pad_size: 0 for unpadded, 20 for padded forward
    """

    def __init__(self, fno, decoder, pad_size: int = 0):
        super().__init__()
        self.fno = fno
        self.decoder = decoder
        self.pad_size = pad_size

    def _fno_latent(self, rho_ext):
        """Extract pre-projection latent [B, 32, Nz] via forward hook."""
        captured = {}
        def _hook(module, inputs, output):
            captured["latent"] = inputs[0]

        # Route through the correct FNO entry point
        if self.pad_size > 0:
            # Padded: wrap with reflect padding then crop
            rho_padded = F.pad(rho_ext, (self.pad_size, self.pad_size), mode="reflect")
            target = self.fno
            hook = target.projection.register_forward_hook(_hook)
            try:
                _ = target(rho_padded)
                latent_padded = captured["latent"]
                latent = latent_padded[:, :, self.pad_size:-self.pad_size]
            finally:
                hook.remove()
            return latent
        else:
            target = self.fno
            hook = target.projection.register_forward_hook(_hook)
            try:
                _ = target(rho_ext)
                latent = captured["latent"]
            finally:
                hook.remove()
            return latent

    def forward_data(self, rho_ext):
        """Data path: FNO projection MLP → [B, 9, 200]."""
        if self.pad_size > 0:
            rho_padded = F.pad(rho_ext, (self.pad_size, self.pad_size), mode="reflect")
            out_padded = self.fno(rho_padded)
            return out_padded[:, :, self.pad_size:-self.pad_size]
        return self.fno(rho_ext)

    def forward_decoder(self, rho_ext, t_q):
        """Decoder path: FNO latent → decoder → [B, 9, Nq]."""
        z = self._fno_latent(rho_ext)
        # t_q in physical seconds; decoder Fourier-embeds internally
        return self.decoder(z, t_q)
