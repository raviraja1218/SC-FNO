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


---

## Reviewer Addendum (Step 4B.2.R, 2026-11-02)

Three clarifications from Step 4B.2 review have been incorporated. The
algorithmic behavior is unchanged in spirit; the changes add diagnostics and
defensive guards. Commit `b412f20` remains the implementation anchor.

### Clarification A — 20-Epoch Reference Window (Now Used)

**Original gap:** `ema_reference_window: 20` was frozen in Step 4B.0 but not
referenced in the Step 4B.2 implementation.

**Resolution:** The 20-epoch window is now used as a **diagnostic look-back**,
not as a controller gate. The update formula remains per-epoch; the first
update happens at epoch 0. The window provides:

- `smoothed_cosine_20_epochs_ago`: the EMA-smoothed cosine at epoch (current − 20)
- `delta_smoothed_cosine`: current smoothed cosine minus 20-epochs-ago value

Both fields are `None` until epoch ≥ 20. They are logged per-epoch in the
history for **post-hoc analysis** (e.g., "was the conflict sustained or
transient over the last 20 epochs?").

**Purpose:** interpretability and diagnostic transparency. Not a training-time
gate. The controller responds to the current smoothed cosine immediately.

**Verification:** test T11 confirms that for epochs 0-19 the look-back fields
are None, and for epochs 20+ they match the smoothed cosine at exactly the
20-epochs-earlier index.

### Clarification B — Cosine Numerical Bounds (Now Clipped)

**Original gap:** raw cosine was used as-is without validation.

**Resolution:** raw cosine is now **clipped to [-1, 1]** before EMA smoothing.
The original value is preserved in history for diagnostics.

Logging schema per epoch now includes:

- `raw_cosine_input`: the value passed to `update()` (may be outside [-1, 1])
- `raw_cosine_used`: the clipped value (in [-1, 1]) that entered the EMA
- `cosine_clipped`: boolean flag; True if `|raw_input − clipped| > 1e-12`

Summary statistics include:
- `n_cosine_clipped`: count of epochs where cosine was clipped
- `cosine_clip_fraction`: fraction of epochs with clipping

**Why this matters:** if PyTorch FP arithmetic ever produces a cosine slightly
outside [−1, 1] (e.g., 1.0000001 or −1.0000001 due to rounding), the clipping
prevents it from influencing the adaptive update in an unintended way. If a
genuine caller bug produces a wildly invalid cosine (e.g., 1.5), the clip
prevents runaway behavior and the `cosine_clipped=True` flag makes the
anomaly visible in logs.

**Verification:** test T12 confirms:
- `raw_cosine=1.5` → clipped to 1.0, flag=True
- `raw_cosine=-2.3` → clipped to -1.0, flag=True
- `raw_cosine=0.5` → unchanged, flag=False
- Summary records correct clip counts

### Clarification C — Adaptation Rate Analysis

**With frozen α=0.05, β=0.1, λ_init=1e-4:**

**Per-epoch maximum multiplicative change** (when smoothed cosine approaches +1):
$$\lambda_{e+1} / \lambda_e = e^{0.05} \approx 1.0513$$
i.e., ~5.13% increase per epoch at sustained maximum positive cosine.

**Per-epoch maximum decrease** (when smoothed cosine approaches −1):
$$\lambda_{e+1} / \lambda_e = e^{-0.05} \approx 0.9512$$
i.e., ~4.88% decrease per epoch at sustained maximum negative cosine.

**Maximum reachable λ over typical runs:**

| Epochs | Sustained max positive cosine | λ reachable from init 1e-4 |
|--------|-------------------------------|---------------------------|
| 55 (typical plateau) | +1.0 | ≈ 1.56e-3 |
| 200 (max epochs) | +1.0 | ≈ 2.20 |
| 250 | +1.0 | ≈ 5.75 (approaching λ_max) |
| 280 | +1.0 | ≈ 10.0 (clips at λ_max) |

**Maximum reachable λ from typical negative conflict:**

| Epochs | Sustained max negative cosine | λ reachable from init 1e-4 |
|--------|-------------------------------|---------------------------|
| 55 | −1.0 | ≈ 6.4e-6 |
| 100 | −1.0 | ≈ 6.7e-7 |
| 168 | −1.0 | ≈ 1e-5 (clips at λ_min) |

**Conclusion:** In a normal-length Phase 4B training run (55-200 epochs),
the λ_min/λ_max bounds are **safeguards, not expected operating values**.
The controller's practical range is bounded by the run length, not by the
clip events. Actual λ trajectory will be recorded in P4-C and P4-D training
logs and analyzed in Step 4B.5.

**Verification:** test T5 (extended to 500 epochs) confirms that clip events
do fire when sustained max cosine is held long enough to reach λ_max.

### Test Suite Update

The original 10 tests remain. Two new tests were added:

- **T11** — 20-epoch look-back correctness (Clarification A)
- **T12** — cosine clipping to [-1, 1] (Clarification B)

**Final: 12/12 unit tests pass.**

### Frozen Configuration Preserved

- α = 0.05: unchanged
- λ_min = 1e-5: unchanged
- λ_max = 10.0: unchanged
- λ_init = 1e-4: unchanged
- β (ema_alpha_cosine) = 0.1: unchanged
- **ema_reference_window = 20: now used as diagnostic look-back (was previously frozen but unused)**
- update_freq = per_epoch: unchanged
- clip_method = log-clip-exp: unchanged

### PCGrad Interaction Preserved

The critical ordering remains:
1. Compute data + physics gradients
2. Compute raw cosine on common valid subset
3. AdaptiveLambda.update(epoch, raw_cosine) ← uses raw cosine, NOT post-projection
4. Apply asymmetric PCGrad projection (P4-A, P4-D only)
5. Combine gradients with updated λ_phys

The cosine clipping happens **inside AdaptiveLambda.update()** — before EMA
smoothing — so it applies to the raw cosine regardless of whether PCGrad
is enabled. In P4-D, PCGrad runs AFTER AdaptiveLambda.update() in the
combined training loop.

### Commit `b412f20` Preserved

The corrections are additive documentation + small implementation
enhancements + 2 new tests. The original commit remains valid; this addendum
(`4B.2.R`) documents the qualifications.
