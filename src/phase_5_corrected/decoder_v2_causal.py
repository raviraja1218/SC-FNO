"""Phase 5 Corrected — causal decoder with temporal attention mask.

⚠ NOT guaranteed to fix FFT wraparound — the FNO backbone has already
mixed information globally. Causal decoder is one experiment; causal
backbone is the fallback if this fails.
"""
import sys; sys.path.insert(0, "src")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_informed.decoder import CoordinateDecoder


class CausalCoordinateDecoder(CoordinateDecoder):
    """CoordinateDecoder with causal attention mask.

    Reuses all parent modules (query_embed, z_proj, q_proj, k_proj, v_proj,
    head) but reimplements forward() to inject the causal mask into the
    cross-attention scores. Head input dimension matches Phase 3 exactly:
    [query_embed (33), attn_out (32)] = 65.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parent has: query_embed (33d), z_proj, q_proj, k_proj, v_proj,
        # n_heads, head. We reuse all of them.

    def forward(self, z, t_q, t_grid=None):
        """z: [B, C, Nz]; t_q: [B, Nq]; t_grid: [Nz] optional.

        Returns: u_hat [B, 9, Nq]
        """
        B, C, Nz = z.shape
        B2, Nq = t_q.shape
        assert B == B2

        if t_grid is None:
            t_grid = torch.linspace(0.0, self.T, Nz, device=z.device, dtype=z.dtype)
        else:
            t_grid = t_grid.to(z.device)

        # 1. Query embedding: [B, Nq, 33]
        q_embed = self.query_embed(t_q)
        # Also project to latent_dim for the attention query
        q = self.q_proj(q_embed)                 # [B, Nq, latent_dim]

        # 2. K/V from z
        z_proj = self.z_proj(z.transpose(1, 2))  # [B, Nz, latent_dim]
        k = self.k_proj(z_proj)                  # [B, Nz, latent_dim]
        v = self.v_proj(z_proj)                  # [B, Nz, latent_dim]

        # 3. Multi-head reshape
        H = self.n_heads
        D = self.latent_dim
        d_h = D // H
        q = q.view(B, Nq, H, d_h).transpose(1, 2)  # [B, H, Nq, d_h]
        k = k.view(B, Nz, H, d_h).transpose(1, 2)  # [B, H, Nz, d_h]
        v = v.view(B, Nz, H, d_h).transpose(1, 2)  # [B, H, Nz, d_h]

        # 4. Scaled dot-product attention
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)  # [B, H, Nq, Nz]

        # 5. CAUSAL MASK: forbid attending to latent j with t_grid[j] > t_q[i]
        tq_expand = t_q.unsqueeze(-1)              # [B, Nq, 1]
        tg_expand = t_grid.view(1, 1, 1, Nz)       # [1, 1, 1, Nz]
        causal_mask = (tg_expand > tq_expand.unsqueeze(1))  # [B, 1, Nq, Nz]
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 6. Softmax (handle any fully-masked rows — shouldn't occur for t_q >= t_grid[0])
        attn = F.softmax(scores, dim=-1)           # [B, H, Nq, Nz]
        attn = torch.nan_to_num(attn, nan=0.0)

        # 7. Apply attention
        out = attn @ v                             # [B, H, Nq, d_h]
        out = out.transpose(1, 2).reshape(B, Nq, D)  # [B, Nq, D=32]

        # 8. Head input: [query_embed (33), attn_out (32)] = 65
        #    This matches Phase 3's exact head input dimension.
        combined = torch.cat([out, q_embed], dim=-1)  # [B, Nq, 65]
        pred = self.head(combined)                    # [B, Nq, 9]
        return pred.transpose(1, 2)                   # [B, 9, Nq]


def build_causal_decoder(latent_dim=32, hidden_dim=64, out_dim=9,
                         n_freqs=16, T=10.0, n_heads=4, seed=42):
    torch.manual_seed(seed)
    return CausalCoordinateDecoder(
        latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        n_freqs=n_freqs, T=T, n_heads=n_heads,
    )

__all__ = ["CausalCoordinateDecoder", "build_causal_decoder"]
