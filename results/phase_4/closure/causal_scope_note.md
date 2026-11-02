# Causal Scope Note (Reviewer Correction D)

**Date:** 2026-11-02
**Status:** Confirmation — already applied in `phase_4a_completion_report.md` Section 8

## Reviewer-Approved Wording (verbatim in report)

> "The tested adaptive optimization procedure achieved a substantial reduction in the normalized nine-equation physics residual. The result demonstrates that the Phase 3 level of physics convergence was not an inherent limitation of the architecture. However, the individual contributions of training duration, physics-weight scheduling, and optimizer sequencing remain to be isolated."

## Limitations Explicitly Acknowledged

The Phase 4A.1 experiment tested a **combined procedure**:
- Fresh Adam from epoch-36 weights
- λ_phys ramp from 0.002754 to 0.398 over 54 epochs
- Plateau detection at epoch 54
- L-BFGS on fixed 16-scenario batch

**Individual contributions cannot be disentangled.** The 828× improvement is an **outcome of the combined procedure**, not attributable to any single component.

## Manuscript Guidance

Do NOT claim:
- "L-BFGS caused the improvement"
- "Extended training caused the improvement"
- "λ_phys ramp caused the improvement"

DO claim:
- "The tested combined adaptive optimization procedure achieved 828× physics residual reduction"

All Phase 4B design decisions must respect this limitation: individual interventions in Phase 4B will test each component separately to isolate causality.
