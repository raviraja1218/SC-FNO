"""Step 3.2 — Autograd derivative wrapper + SCFNO composition.

Composes Phase 2 FNO backbone + new CoordinateDecoder, and provides
compute_du_dt that returns u_hat, du_hat_dt with FULL gradient flow
back to both FNO backbone and decoder params.

CRITICAL: The FNO backbone is used up to its PRE-PROJECTION latent
(32 channels), NOT its final 9-channel output. This lets the decoder
attend over the full latent trajectory z [B, 32, 200] and produce 9
states at arbitrary query times.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from neuralop.models import FNO

from physics_informed.decoder import CoordinateDecoder, build_decoder


class SCFNOModel(nn.Module):
    """Phase 2 FNO backbone (pre-projection latent) + temporal-attention decoder.

    The FNO backbone is decomposed as:
      - lifting       : [B, 1, 200] -> [B, 32, 200]
      - fno_blocks    : 4 x SpectralConv + local linear (keeps 32 channels)
      - projection    : [B, 32, 200] -> [B, 9, 200]   (SKIPPED here)

    We use lifting + fno_blocks to produce z [B, 32, 200], and skip the
    projection layer. The decoder then produces [B, 9, Nq] from z at
    arbitrary query times.

    Phase 2 projection weights remain in the FNO model (unused but intact)
    so the Phase 2 checkpoint SHA is unchanged.
    """

    def __init__(self, fno: FNO, decoder: CoordinateDecoder,
                 decoder_trainable: bool = True):
        super().__init__()
        self.fno = fno
        self.decoder = decoder
        if not decoder_trainable:
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    def _fno_latent(self, rho_ext: torch.Tensor) -> torch.Tensor:
        """Compute FNO latent z [B, 32, 200] via lifting + fno_blocks.

        Uses a forward hook on the projection layer to capture the 32-channel
        pre-projection tensor. This guarantees:
          - Correct numerical values (same as Phase 2 forward)
          - Correct gradient flow through ALL FNO blocks
          - Phase 2 checkpoint unchanged

        Rationale: manually calling `lifting` then `.fno_blocks` directly was
        observed to break gradient flow past block 0 in neuralop 2.0.0. The
        full `self.fno(rho_ext)` call is known-working (Phase 2 trained with
        it for 319 epochs). We use a hook to expose the intermediate without
        changing the forward computation itself.
        """
        captured: dict = {}

        def _capture_pre_projection(module, inputs, output):
            # inputs[0] is the tensor that feeds the projection layer
            captured["latent"] = inputs[0]

        hook = self.fno.projection.register_forward_hook(_capture_pre_projection)
        try:
            _ = self.fno(rho_ext)            # full forward (grad-preserving)
            latent = captured["latent"]
        finally:
            hook.remove()

        assert "latent" in captured, "Forward hook failed to capture latent"
        assert latent.shape == (rho_ext.shape[0], 32, rho_ext.shape[2]), \
            f"Latent shape {tuple(latent.shape)} != expected (B, 32, Nz)"
        return latent

    def forward(self, rho_ext: torch.Tensor, t_q: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z [B, 32, 200], u_hat [B, 9, Nq])."""
        z = self._fno_latent(rho_ext)                   # [B, 32, 200]
        u_hat = self.decoder(z, t_q)                    # [B, 9, Nq]
        return z, u_hat


def compute_du_dt(
    model: SCFNOModel,
    rho_ext: torch.Tensor,
    t_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute u_hat and du_hat/dt via autograd.

    Returns (z, u_hat, du_hat_dt) with FULL graph connectivity:
    du_hat_dt -> u_hat -> decoder -> z -> fno_backbone -> rho_ext.
    """
    if not t_q.requires_grad:
        t_q = t_q.detach().requires_grad_(True)

    z, u_hat = model(rho_ext, t_q)
    du_hat_dt = torch.autograd.grad(
        outputs=u_hat,
        inputs=t_q,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True,
        retain_graph=True,
    )[0]                                                 # [B, Nq]
    du_hat_dt = du_hat_dt.unsqueeze(1).expand(-1, u_hat.shape[1], -1)
    return z, u_hat, du_hat_dt


def load_phase2_fno(
    checkpoint_path: str = "models/baseline/fno_best.pt",
    device: str = "cuda",
) -> FNO:
    """Load Phase 2 FNO from checkpoint. Read-only — never modifies checkpoint."""
    model = FNO(
        n_modes=(16,), hidden_channels=32, in_channels=1, out_channels=9,
        n_layers=4, positional_embedding=None, domain_padding=None,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)
