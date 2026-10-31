# Architecture Decision Record — Phase 3 SC-FNO

**Date:** 2026-10-31
**Decision ID:** ADR-3.0-001
**Status:** FROZEN (implementation in Step 3.2)

## Context

Phase 2 FNO receives a 200-point forcing sequence and returns a 200-point trajectory.
It does not accept arbitrary continuous time `t` as input. Calling
`torch.autograd.grad(prediction, time)` alone does NOT establish a valid
continuous-time derivative pathway because the prediction does not
differentiably depend on an arbitrary `t`.

Phase 3 requires physics residuals of the form `d(u)/dt`, which demand
differentiable temporal derivatives in physical time units.

## Decision

**Adopt a coordinate-conditioned output decoder layered on top of the
unchanged Phase 2 FNO backbone.**

### Forward path (to be implemented in Step 3.2)

    z = FNO(rho_ext)                        # Phase 2 backbone, 200-pt latent
    u_hat(t) = D_theta(z, t)                # coordinate-conditioned decoder
    du_hat/dt = autograd(u_hat(t), t)       # per physical second

Where:
- `z` is the FNO latent representation (shape `[B, 32, 200]`)
- `t` is a continuous query time (physical seconds, scalar or vector)
- `D_theta` is the new differentiable decoder (parameters separate from FNO)
- `u_hat(t)` produces 9 states at query time(s) — shape `[B, 9, N_query]`

### Decoder architecture (freeze here; implementation details in Step 3.2)

- Concatenate `z` (interpolated to query times) with `t` coordinate
- MLP with hidden dim 64, 2 hidden layers, GELU activation
- Output projection → 9 channels
- Time coordinate passed as **physical seconds** (not normalized index)

### Initialization of new decoder parameters

- Deterministic seed 42 (matches Phase 2 baseline seed)
- Kaiming-uniform for MLP weights
- Zero init for final projection bias (so initial decoder output ≈ 0 before training)
- **Phase 2 checkpoint (`fno_best.pt`, SHA `24fce0ba...`) remains unchanged**

### Alternatives considered and rejected

| Alternative | Why rejected |
|-------------|--------------|
| `autograd(prediction, time)` on fixed 200-pt output | No continuous time dependency in FNO output |
| Piecewise-linear interpolation between grid points | Discontinuous derivatives at knots; inadequate for stiff PKE residuals |
| Cubic spline interpolation | Smooth but adds non-learned interpolation error; masks model error from physics loss |
| Finite-difference derivatives only | No true continuous-time loss backprop; cannot train end-to-end |

### Consequences

- **New trainable parameters** (decoder MLP) — parameter count recorded in Step 3.2
- **Phase 2 checkpoint preserved** at SHA `24fce0ba...` — no retraining overwrite
- **Training from Phase 2 weights** for FNO backbone + separately initialized decoder
- **Derivative is exact** through decoder (autograd) once decoder is in place
- **Boundary behavior** determined by decoder receptive field; test at t=0 and t=10 in Step 3.2

## Padding decision (separate)

See `padding_policy.md`. Padding is a **separate variant** — not combined
with physics by default. The 4-variant comparison matrix is in
`experimental_comparison_matrix.md`.

## References

- Phase 3 execution plan, Section 1.2 (derivative pathway)
- Phase 2 supplemental audit, Item 2 (FFT wraparound boundary artifact)
- Phase 2 completion report, Section 5.3 (step-family 28× worse than ramp)
