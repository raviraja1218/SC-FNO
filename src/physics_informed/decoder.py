"""Step 3.2 — Coordinate-conditioned temporal-attention decoder.

Takes FNO latent z [B, 32, Nz] at grid times t_grid [Nz] and query times
t_q [B, Nq] in PHYSICAL SECONDS. Returns 9 predicted states at query times.

Architecture (frozen in Step 3.0 ADR):
  - Fourier query embedding (k=1..16, sin/cos -> dim 32)
  - Cross-attention over z (K, V from z, Q from query embeds)
  - 2-layer MLP head (32 + 32 -> 64 -> 64 -> 9)

No .detach(), no torch.no_grad() in forward path — gradient must flow
from output back to z (FNO latent) AND decoder params.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierQueryEmbedding(nn.Module):
    """Encode physical time t (seconds) as Fourier features.

    Features: [sin(2*pi*k*t/T), cos(2*pi*k*t/T)] for k=1..K, plus t/T itself.
    T = physical time window (seconds), frozen to 10.0 (Phase 1 horizon).
    """

    def __init__(self, n_freqs: int = 16, T: float = 10.0):
        super().__init__()
        self.n_freqs = n_freqs
        self.T = float(T)
        self.out_dim = 2 * n_freqs + 1   # 33 with defaults

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B, Nq] or [Nq]. Returns [B, Nq, out_dim]."""
        if t.dim() == 1:
            t = t.unsqueeze(0)
        # [B, Nq, 1] * [K]
        tau = t.unsqueeze(-1) / self.T
        k = torch.arange(1, self.n_freqs + 1,
                         device=t.device, dtype=t.dtype)  # [K]
        arg = 2.0 * math.pi * tau * k                     # [B, Nq, K]
        sin_f = torch.sin(arg)
        cos_f = torch.cos(arg)
        return torch.cat([tau, sin_f, cos_f], dim=-1)     # [B, Nq, 1+2K]


class CoordinateDecoder(nn.Module):
    """Temporal-attention coordinate decoder.

    z: FNO latent at grid times  [B, C, Nz]  (C = 32 in Phase 2)
    t_q: query times in seconds  [B, Nq]
    t_grid: grid times in seconds [Nz]

    Returns: u_hat [B, 9, Nq]
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 9,
        n_freqs: int = 16,
        T: float = 10.0,
        n_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.T = float(T)

        self.query_embed = FourierQueryEmbedding(n_freqs=n_freqs, T=T)
        q_dim = self.query_embed.out_dim

        # Project z into a K/V-friendly space (identity if latent_dim same)
        self.z_proj = nn.Linear(latent_dim, latent_dim)

        # Cross-attention: query from time embedding, K/V from z
        self.q_proj = nn.Linear(q_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.n_heads = n_heads
        self.attn_scale = (latent_dim // n_heads) ** -0.5

        # MLP head: [attended(z) ; query_embedding] -> 64 -> 64 -> 9
        head_in = latent_dim + q_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Zero-init final projection bias
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z: torch.Tensor, t_q: torch.Tensor,
                t_grid: torch.Tensor | None = None) -> torch.Tensor:
        """
        z      : [B, C, Nz]     (FNO latent, differentiable input)
        t_q    : [B, Nq]        (query times in seconds)
        t_grid : [Nz] or [1, Nz] grid times in seconds (optional, unused here
                 since we attend over the full z)

        Returns: u_hat [B, 9, Nq]
        """
        if z.dim() != 3:
            raise ValueError(f"z must be [B, C, Nz], got {z.shape}")
        B, C, Nz = z.shape
        if t_q.dim() == 1:
            t_q = t_q.unsqueeze(0)
        Nq = t_q.shape[1]

        # --- Project z to K/V space [B, Nz, C] ---
        z_t = z.transpose(1, 2)                          # [B, Nz, C]
        z_kv = self.z_proj(z_t)                          # [B, Nz, C]
        K = self.k_proj(z_kv)                            # [B, Nz, C]
        V = self.v_proj(z_kv)                            # [B, Nz, C]

        # --- Query embeds [B, Nq, q_dim] ---
        q_emb = self.query_embed(t_q)                    # [B, Nq, q_dim]
        Q = self.q_proj(q_emb)                           # [B, Nq, C]

        # --- Multi-head cross-attention: [B, Nq, C] x [B, Nz, C]^T ---
        # Simple single-head first; split to multi-head via reshape
        C_head = C // self.n_heads
        Qh = Q.view(B, Nq, self.n_heads, C_head).transpose(1, 2)  # [B, H, Nq, C_h]
        Kh = K.view(B, Nz, self.n_heads, C_head).transpose(1, 2)  # [B, H, Nz, C_h]
        Vh = V.view(B, Nz, self.n_heads, C_head).transpose(1, 2)  # [B, H, Nz, C_h]

        attn = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.attn_scale
        attn = F.softmax(attn, dim=-1)                   # [B, H, Nq, Nz]
        attended = torch.matmul(attn, Vh)                # [B, H, Nq, C_h]
        attended = attended.transpose(1, 2).reshape(B, Nq, C)  # [B, Nq, C]

        # --- Combine attended + query embeds, MLP head ---
        head_in = torch.cat([attended, q_emb], dim=-1)   # [B, Nq, C + q_dim]
        out = self.head(head_in)                         # [B, Nq, 9]
        return out.transpose(1, 2)                       # [B, 9, Nq]


def build_decoder(latent_dim: int = 32, hidden_dim: int = 64,
                  out_dim: int = 9, n_freqs: int = 16,
                  T: float = 10.0, seed: int = 42) -> CoordinateDecoder:
    """Build decoder with deterministic seed."""
    torch.manual_seed(seed)
    return CoordinateDecoder(
        latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        n_freqs=n_freqs, T=T,
    )
