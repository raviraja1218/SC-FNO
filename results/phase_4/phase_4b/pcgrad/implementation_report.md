# Step 4B.1 — PCGrad Implementation Report

**Date:** 2026-11-02
**Status:** COMPLETE — 9/9 unit tests pass
**Predecessor:** Step 4B.0 LOCKED at `bec0f63`

## Overview

`src/training/pcgrad.py` implements `PCGradOptimizer`, a wrapper around any
`torch.optim.Optimizer` that projects the physics gradient onto the orthogonal
complement of the data gradient when they conflict (cosine < 0).

## Frozen Configuration (from Step 4B.0 pre-registration)

| Parameter | Value |
|-----------|-------|
| eps (projection denominator) | 1e-8 |
| Conflict trigger | cos < 0 |
| Log both raw and post-projection cosine | Required |

## Algorithm

Given:
- g_d = ∇_θ L_data
- g_p = ∇_θ L_physics
- g_ic = ∇_θ L_IC

1. Compute raw cosine: cos_raw = (g_d · g_p) / (||g_d|| ||g_p|| + eps)
2. If cos_raw < 0: project g_p* = g_p − (g_p · g_d)/(||g_d||²) · g_d (exact projection, no eps in denominator — eps-guarded zero-norm case)
3. Else: g_p* = g_p
4. Combined gradient: g_total = g_d + λ_phys · g_p* + λ_ic · g_ic
5. Assign g_total to parameter .grad fields, call base optimizer.step()
6. Log raw cosine, post-projection cosine, norms, projection flag, losses, λ, wall time

## Design Decisions

### Gradient handling
- Uses `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` for data and physics gradients, then final loss without retain
- Avoids `.backward()` accumulation which would sum gradients instead of keeping separate
- `create_graph=False` for memory efficiency (no second-order graph)

### Edge cases handled
- **Missing gradient** (parameter only trained by one loss): parameter excluded from projection computation, but receives the combined gradient it does have
- **Zero-norm gradient**: projection guarded by `norm_sq.item() < eps` check; no division crash
- **Subset mismatch** between data/physics None patterns: common valid index set used for cosine + projection; per-parameter combination still applies to all

### Memory safety
- Three gradient tensors per parameter (~254 KB each at FP32 for 63,634-param model)
- No `create_graph=True` → no second-order graph
- Measured peak on RTX 4050: see `test_T7_memory_measurement.json`

## Unit Tests (9)

| Test | Purpose |
|------|---------|
| T1 | No conflict → no projection applied |
| T2 | Conflict → projection applied; projected orthogonal to data |
| T3 | Post-projection cosine < 1e-5 (FP32 tolerance) |
| T4 | Combined update matches hand computation |
| T5 | Missing gradient handled (data-only / physics-only) |
| T6 | Zero-norm gradient handled safely |
| T7 | Memory < 3 GB on RTX 4050 |
| T8 | Numerical agreement with independent reference |
| T9 | Reproducibility (identical output on same inputs) |

**Result: 9/9 pass.**

## Scientific Note (per reviewer guidance)

The claim of "conflict mitigation" must be based on:
- **raw_cos distribution**: the input conflict level (can be negative — that's what PCGrad handles)
- **post_cos distribution when projection applied**: should be near zero (< 1e-5 in FP32)
- **projection_fraction**: how often conflicts triggered projection

**Do NOT claim "conflicts were eliminated."** Raw cosine remains what it is; PCGrad projects the conflicting component. Report both honestly.

## Files

- `src/training/pcgrad.py` — implementation (227 lines)
- `tests/phase_4/phase_4b/test_pcgrad.py` — 9 unit tests
- `results/phase_4/phase_4b/pcgrad/unit_test_results.log` — full pytest output
- `results/phase_4/phase_4b/pcgrad/test_T7_memory_measurement.json` — VRAM evidence
- `results/phase_4/phase_4b/pcgrad/pcgrad_verification.json` — gate status
