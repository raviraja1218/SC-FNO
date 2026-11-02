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


---

## Reviewer Addendum (Step 4B.1.R, 2026-11-02)

Four clarifications from Step 4B.1 review have been incorporated as additive
documentation. The implementation commit `b8ae156` is preserved; these
additions do not modify the algorithmic behavior.

### Qualification 1 — Asymmetric PCGrad

The implementation is **asymmetric PCGrad** — it projects ONLY the physics
gradient against the data gradient. This is by design (frozen at Step 4B.0
pre-registration), not the full symmetric multi-task PCGrad from Yu et al. 2020.

What is projected:
- Physics gradient `g_p` is projected against data gradient `g_d` when `cos < 0`

What is NOT projected:
- Data gradient `g_d` is not projected against `g_p`
- IC gradient `g_ic` is not projected against either `g_d` or `g_p`
- IC-involving conflicts are not resolved

**Terminology to use in all downstream references:** "asymmetric PCGrad" or
"physics-gradient projection against the data objective". Do NOT claim "all
task-gradient conflicts are resolved".

**Cosine computation:** both `raw_cos` and `post_cos` are computed on the
common valid parameter subset (params where BOTH data and physics gradients
exist). Verified by test T11.

### Qualification 2 — Memory Evidence Scope (Corrected)

Original Step 4B.1 report said: "PCGrad variants can run alongside normal
training with no memory penalty concern."

**Corrected statement:**

> The synthetic-gradient benchmark (T7) measures PCGrad memory overhead in
> isolation: 17.84 MB peak on a 3-layer MLP test model. This establishes
> low standalone PCGrad overhead. It does NOT establish total memory during
> actual FNO training (where the model, activations, residual calculation,
> and multiple gradient evaluations are present). Actual training-stage VRAM
> will be measured during Step 4B.4.

### Qualification 3 — Mixed-Gradient Behavior (Documented)

Explicit behavior for all 6 mixed-gradient cases:

| # | Case | Behavior |
|---|------|----------|
| 1 | Data-only parameter | `.grad` = `g_d + λ_ic·g_ic`; physics contribution skipped; no projection |
| 2 | Physics-only parameter | `.grad` = `λ_phys·g_p + λ_ic·g_ic`; NOT projected (no `g_d` to project against) |
| 3 | Shared parameter (data + physics) | `.grad` = `g_d + λ_phys·g_p* + λ_ic·g_ic`; `g_p*` is projected |
| 4 | IC gradient | Added unprojected always: `+ λ_ic·g_ic` |
| 5 | All gradients missing | `.grad = None` (not zero); parameter will NOT be updated this step |
| 6 | Between iterations | Training loop must call `pcgrad.zero_grad()` before next `backward`; verified by user code, not tested here |

Cases 1-5 are verified by tests T4, T5, T10. Case 6 is a training-loop contract.

### Qualification 4 — Same-Subset Cosine Verification

Verified by test T11: when some parameters are missing gradients,
both `raw_cos` and `post_cos` are computed on the same "common valid
parameter subset". No mismatched flattening.

### Test Suite Update

The original 9 tests remain. Two additional tests were added:
- **T10** — All gradients missing for a parameter → `.grad = None`
- **T11** — Same-subset cosine computation with mixed missing patterns

**Final: 11/11 unit tests pass.**

### Frozen Configuration Preserved

eps=1e-8, conflict trigger cos<0, raw+post-projection cosine logged — all
unchanged. No training, no locked paths modified. Commit `b8ae156` remains
the implementation anchor; this addendum documents the qualification
refinements.
