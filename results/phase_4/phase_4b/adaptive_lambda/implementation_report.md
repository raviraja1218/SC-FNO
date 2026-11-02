# Step 4B.2 — Adaptive Lambda_phys Implementation Report

**Date:** 2026-11-02
**Status:** COMPLETE — 10/10 unit tests pass
**Predecessor:** Step 4B.1 fully closed at `3fd0af0`

## Overview

`src/training/adaptive_lambda.py` implements `AdaptiveLambda`, a per-epoch
adaptive weighting module for the physics loss term. The weight is updated
based on the exponentially-smoothed cosine similarity between data and physics
gradients, bounded in log space.

## Frozen Configuration (from Step 4B.0 pre-registration)

| Parameter | Value |
|-----------|-------|
| alpha | 0.05 |
| lambda_min | 1e-5 |
| lambda_max | 10.0 |
| lambda_init | 1e-4 |
| ema_beta | 0.1 |
| update_freq | per_epoch |
| clip_method | log-clip-exp |

## Algorithm

### EMA smoothing of cosine

    smoothed_cosine_e = beta * raw_cosine_e + (1 - beta) * smoothed_cosine_{e-1}

Initialized at first epoch: `smoothed_cosine_0 = raw_cosine_0`.

### Log-space update with clipping

    log_lambda_prev = log(lambda_e)
    log_lambda_raw = log_lambda_prev + alpha * smoothed_cosine_e
    log_lambda_final = clip(log_lambda_raw, log(lambda_min), log(lambda_max))
    lambda_{e+1} = exp(log_lambda_final)

The clipping in log space ensures:
- Numerically stable (no underflow/overflow)
- Bounded in [lambda_min, lambda_max]
- Smooth exponential behavior preserved

## Behavior

- **Positive smoothed cosine** (gradients aligned): lambda increases → physics reinforced
- **Negative smoothed cosine** (conflicting): lambda decreases → data protected
- **Bounded** [1e-5, 10] via log-space clip

## PCGrad Interaction (P4-D Critical)

When both PCGrad and AdaptiveLambda are enabled (variant P4-D), the order of
operations is:

    1. Compute raw cosine = cos(g_data, g_physics) on common valid subset
    2. If PCGrad enabled: project g_physics onto orthogonal complement of g_data
       → g_physics* (post-projection cosine is ~0 by construction)
    3. AdaptiveLambda.update(epoch, raw_cosine) — uses RAW cosine, NOT post-projection
       Rationale: post-projection cosine is ~0 by design, which would freeze lambda
       at init value. The raw cosine measures actual task conflict, which is what
       should drive the adaptive weight.
    4. Combined gradient = g_data + lambda_phys * g_physics* + lambda_ic * g_ic

**CRITICAL:** Do NOT pass post-projection cosine to AdaptiveLambda in P4-D.
Document this in the P4-D training log so the lambda evolution is interpretable.

## Design Decisions

### Validation at construction

- `lambda_min < lambda_max` required
- `lambda_init` must be within `[lambda_min, lambda_max]`
- `ema_beta` in (0, 1)
- `alpha > 0`

Invalid construction raises `ValueError` immediately.

### Numerical stability

- Log-space arithmetic prevents underflow when lambda is very small
- Log-space arithmetic prevents overflow when lambda is very large
- `math.exp(math.log(x))` may return `x * (1 ± 1e-15)` — this is FP noise, not a bug (verified by T6 with tolerance)

### History tracking

Every update appends a record with:
- epoch
- raw_cosine
- smoothed_cosine
- log_lambda_prev
- log_lambda_raw_update
- log_lambda_after_clip
- lambda (final)
- clip_applied ("min", "max", or "")
- alpha (for provenance)

Summary statistics (`get_summary()`) include:
- n_updates
- lambda_init
- lambda_final
- lambda_min_seen
- lambda_max_seen
- n_clip_min
- n_clip_max
- clip_fraction

## Unit Tests (10)

| Test | Purpose |
|------|---------|
| T1 | Initialization (defaults, smoothed_cosine=None, empty history) |
| T2 | First update (EMA initialization, log update formula) |
| T3 | Subsequent update (EMA smoothing formula) |
| T4 | Lambda_min clip (strongly negative cosine) |
| T5 | Lambda_max clip (strongly positive cosine) |
| T6 | Numerical stability with extreme cosines (FP noise tolerated) |
| T7 | Reproducibility (identical histories) |
| T8 | Log history field correctness |
| T9 | Summary statistics accuracy |
| T10 | Frozen config defaults + invalid construction raises |

**Result: 10/10 pass.**

## Scientific Note

AdaptiveLambda provides the mechanism by which P4-C (adaptive lambda only) and
P4-D (PCGrad + adaptive lambda + unified padding) will dynamically balance
data and physics objectives. Per Step 4B.5 evaluation criteria, the success of
this intervention will be measured by:
- Whether lambda_phys converges to a stable value
- Whether the data accuracy vs physics loss trade-off improves over Phase 3 default
- Whether P4-C and P4-D beat P3-PI-FNO on the paired bootstrap

## Files

- `src/training/adaptive_lambda.py` — implementation (frozen config defaults)
- `tests/phase_4/phase_4b/test_adaptive_lambda.py` — 10 unit tests
- `results/phase_4/phase_4b/adaptive_lambda/unit_test_results.log` — pytest output
- `results/phase_4/phase_4b/adaptive_lambda/adaptive_lambda_verification.json` — gate status
