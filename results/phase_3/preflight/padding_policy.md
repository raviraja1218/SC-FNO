# Padding Policy — Phase 3 SC-FNO

**Date:** 2026-10-31
**Decision ID:** ADR-3.0-002
**Status:** FROZEN (as candidate variant, not mandatory)

## Context

Phase 2 supplemental audit Item 2 (`results/phase_2/supplemental_audit/02_temporal_boundary_analysis.md`)
quantified an FFT wraparound boundary artifact:

- Neutron density n: start/interior MAE ratio = **8.05×**
- Coolant temperature Tc: start/interior MAE ratio = **8.66×**
- Median across 9 states: **4.07×**
- All 10 ratios > 2× occur on the **start** boundary (FFT periodic wrap)

Root cause: FFT assumes periodic input. Trajectory at t=10 ≠ trajectory at t=0,
so end-of-trajectory state wraps around to corrupt the start.

Step transients — Phase 2's weakest family (Rel L2 median 5.896e-3, 28× worse
than ramp 2.115e-4) — have their sharpest dynamics at t=0. This is the
documented explanation for the step-family weakness.

## Decision

**Test padding as an independent experimental variant. Do NOT assume
padding fixes the boundary artifact. Do NOT combine with physics by default.**

### Candidate padding configuration

- **Size:** 20 points each side (10% of 200) → 240 points total before spectral layers
- **Padding modes to test (each as separate variant):**
  1. Zero padding
  2. Reflection padding
  3. Replication (edge) padding
- **Crop back to 200 points** before computing any metric
- **Report boundary ratio** for each mode after cropping

### Justification for testing each mode

- **Zero padding:** simplest baseline, but introduces artificial discontinuity at boundary
- **Reflection padding:** preserves continuity at boundary; common in signal processing
- **Replication padding:** preserves DC level at boundary; may preserve physical continuity better than zero

**No mode is assumed correct.** All 3 measured in Step 3.4.

### Test protocol (Section 1.1 matrix)

- **Control:** padded FNO without physics loss (isolates padding effect)
- **Treatment:** padded SC-FNO with physics loss (tests combined intervention)
- **Baseline:** Phase 2 FNO (unpadded, no physics)

## What padding does NOT fix

- **Prompt-jump resolution** (Δt = 0.05 s > τ_prompt = 0.031 s) — padding does not add temporal resolution
- **Physics constraint violations** — padding alone does not enforce PKE residuals
- **Generalization outside training distribution** — padding alone does not add new information

## Consequences

- Additional variants added to Step 3.4 training (see experimental_comparison_matrix.md)
- Boundary ratio reported per mode; primary SC-FNO candidate selects mode based on evidence
- Target: neutron start/interior ratio < 2.0 (Phase 2 baseline = 8.05)
