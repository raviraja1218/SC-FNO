"""Corrected per-state derivative: d(u_j)/dt for each channel j."""
import torch


def compute_du_dt_per_channel(model, rho_ext, t_q):
    """Per-state derivatives du_j/dt for j=0..8.

    Args:
        model: SCFNOModel
        rho_ext: [B, 1, T] normalized forcing
        t_q: [B, Nq] query times (requires_grad)

    Returns:
        z: latent
        u_hat: [B, 9, Nq] normalized states
        du_dt: [B, 9, Nq] per-state derivatives
    """
    if not t_q.requires_grad:
        t_q = t_q.detach().requires_grad_(True)

    z, u_hat = model(rho_ext, t_q)
    B, S, Nq = u_hat.shape
    assert S == 9

    du_dt = torch.zeros_like(u_hat)
    for j in range(S):
        grad_j = torch.autograd.grad(
            outputs=u_hat[:, j, :],
            inputs=t_q,
            grad_outputs=torch.ones_like(u_hat[:, j, :]),
            create_graph=True,
            retain_graph=True,
        )[0]
        du_dt[:, j, :] = grad_j

    return z, u_hat, du_dt
