# Audit Item 3 — Prompt Dynamics Limitation (Documentation)

## Concern (Reviewer 2)

The Phase 2 completion report stated that the 16-mode Fourier truncation "acts as a
learned low-pass filter" and "makes the model less inclined to hallucinate
high-frequency artifacts." While regularization-benefit is real, this is NOT
automatically a physical advantage: the same truncation can suppress genuine
sharp transients.

The concern is sharpened by two facts:
- Prompt timescale in Phase 1 data: τ_prompt ≈ 0.031 s
- Output sample interval: Δt = 0.05 s
- Ratio: Δt/τ_prompt ≈ 1.63 — the sampling grid cannot resolve prompt dynamics
- FNO retains only 16 lowest Fourier modes out of 100 available (for N_t = 200)

The step family is the weakest family in test results (median rel L2 = 5.896e-3),
consistent with the hypothesis that sharp prompt-jump features are under-learned.

## Corrected Manuscript Wording

Replace the original statement with the following:

> "The baseline operates on a temporally sampled representation (Δ t = 0.05 s)
> with a limited spectral bandwidth (16 retained Fourier modes). The 16-mode
> truncation serves as a learned low-pass regularizer — a potential benefit for
> numerical stability — but it also constrains the model's ability to represent
> sub-grid sharp transients. The prompt timescale τ_prompt ≈ 0.031 s is
> comparable to the sampling interval, and the baseline's ability to reproduce
> sub-grid prompt dynamics has not been established by Phase 2 results. The
> per-family accuracy breakdown reflects this limitation: step-family median
> relative L2 (5.896e-3) is 28× the ramp-family median (2.115e-4). No claim is
> made that the FNO resolves prompt-jump physics; improving step-family accuracy
> is a specific research target for Phase 3."

## What Phase 2 Does and Does Not Establish

| Aspect | Established? |
|--------|-------------|
| Baseline predicts 9-state trajectories on training-like forcing | ✅ Yes |
| Baseline reproduces ramp/sinusoidal dynamics well | ✅ Yes |
| Baseline resolves sub-grid prompt dynamics at Δt < 0.05 s | ❌ Not established |
| 16-mode truncation is sufficient for step transients | ❌ Not demonstrated |
| Physics-informed constraint will improve step family | ❌ Open Phase 3 question |

## Phase 3 Implication

The step-family weakness (5.90e-3 vs 2.11e-4 ramp) is the specific, quantitative
signal that motivates the Phase 3 physics-informed residual approach. Phase 3
must report step-family improvement AND verify that ramp/sinusoidal performance
does not degrade.
