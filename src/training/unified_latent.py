"""Unified-latent training helper.

Provides the canonical training-loop pattern that guarantees bit-identity
between data and physics paths:

    from training.unified_latent import unified_latent_step

    result = unified_latent_step(model, decoder, rho_ext, t_data, t_q,
                                 compute_data_loss, compute_physics_residuals)

The key guarantee: ONE forward through the FNO. Both data and physics losses
use the SAME latent tensor.

For non-padded variants (P4-A, P4-C, and legacy P3/P4A), the helper falls
back to the standard per-path forward.
"""
from __future__ import annotations

import torch


def unified_latent_forward(model, decoder, rho_ext, t_q):
    """Compute latent ONCE, decode for both data path (t_q=t_data implicitly)
    and physics path.

    Returns:
        latent: [B, 32, 200]
        u_at_t_q: [B, 9, Nq]
    """
    if hasattr(model.fno, "get_latent"):
        # UnifiedLatentFNO: single forward through padded FNO, cropped to 200
        latent = model.fno.get_latent(rho_ext)
    else:
        # Fallback: raw FNO (P4-A, P4-C, legacy P3/P4A)
        captured = {}
        def _hook(module, inputs, output):
            captured["latent"] = inputs[0]
        hook = model.fno.projection.register_forward_hook(_hook)
        try:
            _ = model.fno(rho_ext)
        finally:
            hook.remove()
        latent = captured["latent"]

    u_at_t_q = decoder(latent, t_q)
    return latent, u_at_t_q


def unified_latent_step(model, decoder, rho_ext, t_data, t_q, compute_data_loss_fn):
    """Canonical training step for the unified-latent pattern.

    Args:
        model: SCFNOModel with .fno (possibly UnifiedLatentFNO)
        decoder: CoordinateDecoder
        rho_ext: [B, 1, 200] normalized external reactivity
        t_data: [B, 1, 200] data path query times
        t_q: [B, Nq] physics query times
        compute_data_loss_fn: callable(latent, t_data) -> data loss

    Returns:
        dict with:
            - latent: [B, 32, 200] shared latent
            - u_data: [B, 9, 200] data path prediction
            - u_phys: [B, 9, Nq] physics path prediction
            - L_data: scalar data loss (from compute_data_loss_fn)
    """
    # Single forward through the FNO (or fallback)
    if hasattr(model.fno, "get_latent"):
        latent = model.fno.get_latent(rho_ext)
    else:
        captured = {}
        def _hook(module, inputs, output):
            captured["latent"] = inputs[0]
        hook = model.fno.projection.register_forward_hook(_hook)
        try:
            _ = model.fno(rho_ext)
        finally:
            hook.remove()
        latent = captured["latent"]

    # Both paths use the SAME latent
    u_data = decoder(latent, t_data)
    u_phys = decoder(latent, t_q)
    L_data = compute_data_loss_fn(u_data)

    return {
        "latent": latent,
        "u_data": u_data,
        "u_phys": u_phys,
        "L_data": L_data,
    }
