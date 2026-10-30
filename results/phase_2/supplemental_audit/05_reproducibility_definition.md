# Audit Item 5a — Reproducibility Tolerance Definition

## What the 3.4 × 10⁻⁵ Number Measures

The Phase 2 closure report documented "re-inference determinism verified,
max diff < 3.4 × 10⁻⁵" without specifying the formula. Reviewer 2 correctly
asked what this quantity is. Here is the exact answer from
`results/phase_2/closure/reproducibility.json`:

## Exact Formula (from closure_reproducibility.py)

For each of 5 test scenarios and each of 9 states:

    per_state_max_relative_diff[state] = max over the 5 scenarios of:
        max_t | ŷ_re_inference(t) − ŷ_stored(t) |
        ─────────────────────────────────────────
        max( max_t |ŷ_re_inference(t)|, max_t |ŷ_stored(t)| )

The reported "3.4 × 10⁻⁵" is the **maximum of these per-state values across all 9 states** — that is, the single worst-case per-state relative difference observed in the re-inference test.

## Measured Values (from reproducibility.json)

| State | Per-state max relative diff (across 5 scenarios) |
|-------|------------------------------------------------|
| n | 3.39 × 10⁻⁵ |
| C1 | 1.73 × 10⁻⁶ |
| C2 | 5.18 × 10⁻⁶ |
| C3 | 1.08 × 10⁻⁵ |
| C4 | 2.82 × 10⁻⁵ |
| C5 | 3.26 × 10⁻⁵ |
| C6 | 1.82 × 10⁻⁵ |
| Tf | 1.15 × 10⁻⁵ |
| Tc | 7.58 × 10⁻⁷ |

**Maximum across all states: 3.39 × 10⁻⁵ (state `n`)**

## What This Means

- The **reported 3.4 × 10⁻⁵ is the worst per-state relative difference** when re-running inference on 5 test scenarios
- It is **not** the maximum absolute difference, not the mean difference, not a per-timestep difference
- It is consistent with FP32 CUDA arithmetic non-determinism (~10⁻⁵ relative error on GPU matmul/FFT is expected)
- **No evidence of checkpoint corruption or weight drift** — differences are all within FP32 noise
- **Phase 2 prediction evidence remains valid** — the `predictions_summary.npz` file is the authoritative stored output; the re-inference just confirmed it's reproducible from `fno_best.pt`

## For the Manuscript

Precise wording:

> "Re-inference of the saved `fno_best.pt` checkpoint on 5 representative test
> scenarios reproduced the stored predictions to a **maximum per-state relative
> difference of 3.4 × 10⁻⁵** (formula: `max_t |ŷ_re − ŷ_stored| /
> max(max_t|ŷ_re|, max_t|ŷ_stored|)`, taken as the maximum over 9 states).
> This is consistent with FP32 CUDA arithmetic non-determinism and confirms the
> checkpoint reproduces the evaluated results."
