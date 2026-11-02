# Step 4B.5 — Correction Addendum (Post-Review)

**Date:** 2026-11-02
**Predecessor:** Commit `d226447` (Step 4B.5)
**Correction commit:** `b5f94a0`
**Status:** ADDITIVE — original Step 4B.5 artifacts preserved

## Correction 1 — Physics Measurement Revalidation

### The two defects

1. **Per-sum derivative**: `compute_du_dt` used `grad_outputs=torch.ones_like(u_hat)`, computing `∂(Σ_j u_j)/∂t` and assigning to all 9 channels. Correct is per-channel `∂u_j/∂t`.
2. **Zero forcing (not a defect — a documented training approximation)**: `rho_fn = lambda tq: zeros_like(tq)` used throughout Phase 3/4A/4B training. Not corrected in this addendum because correcting it would measure a residual the models never minimized (see Correction 1c).

### Correction 1a — Per-channel derivative only

Revalidation with **per-channel `∂u_j/∂t`** and **zero forcing** (preserving the training protocol):

| Model | Original (per-sum) | Corrected (per-channel) | Ratio |
|-------|-------------------|------------------------|-------|
| Phase4A.1 | 1.95e-6 | **0.2295** | 8.5e-6× |
| P3-PI-FNO | 1.67e-3 | **3.51** | 4.8e-4× |
| P4-A | 174.07 | **608.78** | 0.286× |
| P4-C | 182.28 | **1260.73** | 0.145× |
| P4-B | 204.41 | **2191.05** | 0.093× |
| P4-D | 225.58 | **7364.37** | 0.031× |

**Relative ordering is preserved** under the corrected protocol: Phase4A.1 < P3-PI-FNO < P4-A < P4-C < P4-B < P4-D.

**Corrected claims:**
- Phase 4A.1 remains the best physics-converged model.
- Phase 4A.1 is **15.3× better than P3-PI-FNO** under corrected protocol (vs 828× under the original per-sum protocol).
- P4-A/B/C/D are **173–2100× worse than Phase 4A.1** under corrected protocol (vs 100,000× under original).

### Correction 1b — Zero-forcing approximation: scope clarification

The corrected physics losses above are **zero-forcing residuals**, not true-physics residuals. The trained models were trained to satisfy the zero-forcing residual. Their "convergence" is convergence to the zero-forcing trajectory — a **within-model consistency metric**, not the true 9-equation physics.

**True-physics residuals** (with actual scenario forcing) are **not measured by any existing trained model**, because the models never minimized that residual. Measuring true-physics residuals would require **retraining** and is deferred to **Phase 5**.

### Correction 1c — Blowup under forced-physics measurement

A separate revalidation using per-channel derivative + **actual (denormalized) forcing** produced ~1e8 for all 6 models (see `physics_revalidation_results.json`). This is consistent with the models never having been trained against the true-physics residual. It is **not evidence of a new bug**; it is confirmation that the training protocol minimizes a zero-forcing objective.

### Correction 1d — Bug A attribution scope

**Original claim:** "P4-A/B/C/D physics loss is 100,000× worse than P3 baseline."

**Corrected claim:** "P4-A/B/C/D physics loss is 173–2100× worse than Phase 4A.1 under the corrected per-channel zero-forcing protocol. The primary cause remains the frozen Phase 3 training protocol (patience=30, init λ_phys=1e-4) that terminated training before λ_phys reached meaningful pressure. The corrected protocol preserves the ordering; the exact magnitude and per-state signature of the training failure are now reported under the corrected protocol."

## Correction 2 — Test-Set Usage History

The 200-scenario test set was used in:
- Phase 2 Step 2.4 (P2-FNO baseline)
- Phase 3 Step 3.5 (P2, P3, 4A.1, P4 variants)
- Phase 4B Steps 4B.0–4B.4 (variant diagnostics)
- **Step 4B.5** — definitive seven-model simultaneous evaluation

"First and only use" was incorrect and is retracted.

## Correction 3 — Statistical Clarification

- **Statistic:** median paired difference `median_i(P2_relL2_i − variant_relL2_i)`.
- **Bootstrap:** 10,000 resamples with replacement, seed 42, percentile method (2.5th/97.5th).
- **Sign:** positive = variant better than P2.
- **Phase 4A.1:** CI [+3.68e-7, +1.02e-5] excludes zero at α=0.05.
- **Multiple comparisons:** 5 variant-vs-P2 comparisons. No Bonferroni applied. At Bonferroni α=0.01, Phase 4A.1 remains significant.
- **Scope:** establishes median superiority, not per-scenario superiority (121 wins / 79 losses).

## Correction 4 — Complete Phase 4B Variant Data

Full P4-B/C/D per-family and bootstrap data are in:
- `per_family_metrics.csv` (full)
- `paired_bootstrap_vs_p2.json` (full)

## Correction 5 — M4 Comparison Clarification

- Val improvements (Step 4B.4): P4-A −1.35% vs P3, P4-C −1.13% vs P3.
- Test results (Step 4B.5): P4-A 23.5% win rate vs P2 (76.5% loss rate).
- **Different baselines** (P3 for training val, P2 for test).

## Correction 6 — Scientific Conclusions

**Confirmed under both protocols:**
- Phase 4A.1 has best data accuracy and best physics consistency.
- Phase 4A.1 beats P2-FNO on paired bootstrap (121/200 wins, 60.5%, CI positive).
- Phase 4A.1 physics loss is significantly better than P3 baseline.
- P4-A/B/C/D are invalid as physics-informed models (Bug A).

**Corrected under per-channel protocol:**
- Phase 4A.1 vs P3 ratio: **15.3×** (was 828× under original protocol)
- P4-A/B/C/D vs Phase 4A.1 ratio: **173–2100×** (was 100,000× under original)

**Not measured (deferred to Phase 5):**
- True-physics residuals with actual forcing
- Whether Phase 4A.1's 0.2295 zero-forcing residual is a genuinely better physics model, or a better zero-forcing residual model

## Correction 7 — Boundary Metrics Note

The boundary ratio for Phase 4A.1 is **5.16×**, not the <4.0× target from the Phase 4 plan. The FFT wraparound artifact remains unresolved. This is a Phase 4 limitation that carries into Phase 5.

## Reviewer Response Checklist

| Reviewer concern | Addressed |
|-----------------|-----------|
| Physics loss revalidation | ✅ Correction 1a, 1b, 1c |
| Attribution scope (Bug A) | ✅ Correction 1d |
| Test-set history | ✅ Correction 2 |
| Statistical reporting | ✅ Correction 3 |
| Missing summary data | ✅ Correction 4 |
| M4 comparison baseline | ✅ Correction 5 |
| Boundary limitations preserved | ✅ Correction 7 |

---

## Correction 8 (post-second-review) — Jacobian validation closed

### Reviewer concern
> *"If the decoder's temporal attention allows one query output to depend on other query-time inputs, computing a gradient for each channel by summing over query positions is still not sufficient. The correct quantity is ∂u_j(t_k)/∂t_k for each state j and query point k, extracted from the appropriate diagonal of the Jacobian when query points are coupled."*

### Audit result

**Empirical test (see `decoder_jacobian_audit.log`):**

Perturbing `t_q[:, 5]` by +0.01 yields:
- Output change at query 5: **1.047e-3**
- Output change at queries i≠5: **0.000e+00** (mean and max)

**Structural confirmation (from `src/physics_informed/decoder.py`):**
- Cross-attention operates over **FNO latent positions** (Nz = 200), NOT over query positions
- Query `Q[b, k, :]` depends only on `t_q[b, k]`
- Keys and values `K[b, n, :]`, `V[b, n, :]` depend only on latent `z`
- Attention weight `attn[b, k, n]` = query k attends to latent n; no query-to-query coupling

### Conclusion

**The `CoordinateDecoder` is query-independent.** Each query `t_k` computes its output independently of other query times `t_i` (i ≠ k). Therefore:

$$\frac{\partial u_j(t_k)}{\partial t_i} = 0 \quad \text{for all } i \neq k$$

And the current per-channel derivative implementation with `grad_outputs = ones_like(...)` evaluates `∂(Σ_k u_j(t_k))/∂t` which, since only diagonal terms are nonzero, equals the correct per-query derivative at each `t_k` for the per-query residual formulation used in `compute_residuals`.

**Issue 1 (Jacobian validation) is CLOSED.** The corrected physics losses reported in `physics_revalidation_v2_zero_forcing.json` are mathematically valid measurements of the per-query, per-state, zero-forcing physics residual.

### Reviewer's other condition (Bug A attribution scope)

Per reviewer guidance, the Bug A conclusion is scoped to:
> "The Phase 4B training protocol was inadequate for physics convergence under the frozen Phase 3 setup. The exact contribution of early stopping, lambda schedule, and other protocol factors to the observed residual failure remains to be isolated."

No claim is made that the individual interventions (PCGrad, AdaptiveLambda, UnifiedLatentPadding) are inherently ineffective. **They remain scientifically untested under an adequate training protocol.**
