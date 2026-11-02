"""Phase 5 Corrected — decoder re-export with supervised training pathway.

Architecture identical to Phase 3 (frozen in Step 3.0 ADR). Only the training
loop changes: L_data_decoder now anchors decoder output to ground truth at
query times (Bug J fix).

No architecture change — the fix is in losses_v2.py and trainer_v2.py.
This module exists so corrected code has its own namespace, per Step 5.A rules.
"""
import sys
sys.path.insert(0, "src")

# Re-export Phase 3 decoder unchanged
from physics_informed.decoder import CoordinateDecoder, FourierQueryEmbedding

def build_decoder_v2(latent_dim: int = 32, hidden_dim: int = 64,
                     out_dim: int = 9, n_freqs: int = 16,
                     T: float = 10.0, n_heads: int = 4, seed: int = 42):
    """Build decoder matching Phase 3 architecture exactly.

    Initialization seed fixed to 42 for reproducibility across variants.
    Same architecture as build_decoder (Phase 3) — decoder_v2 adds only
    the supervised-training pathway in the loss/trainer.
    """
    import torch
    torch.manual_seed(seed)
    return CoordinateDecoder(
        latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        n_freqs=n_freqs, T=T, n_heads=n_heads,
    )

__all__ = ["CoordinateDecoder", "FourierQueryEmbedding", "build_decoder_v2"]
