# Reviewer Decision — Bug J Response

**Date:** 2026-11-02
**Decision:** Option B — Preserve Phase 0–4 locks, build corrected architecture separately
**Status:** Phase 5.1 and 5.2 abandoned; fresh architecture work begins

## Reviewer Position

Direct quote:

> "My recommendation is choose the research path that produces the strongest
> scientifically defensible result, not the path that produces the most
> impressive-looking paper or the fastest publication.
>
> Choose Option B: preserve the Phase 0–4 locks and build a corrected,
> separately versioned architecture. I would not reopen the old phases just
> to make the original physics-informed results look successful.
>
> The research direction should be:
> - Preserve the original Phase 0–4 experiments as historical evidence.
> - Explicitly document Bug J: the decoder was not receiving meaningful
>   time-dependent supervised training.
> - Build a corrected decoder with a proper supervised data-loss pathway.
> - Retrain and evaluate it from a clean, traceable version.
> - Test unseen scenarios, all four forcing families, and fair
>   numerical-solver comparisons.
> - Compare the corrected model against the original data-only FNO and the
>   previous degenerate physics-informed model."

## Evidence Hierarchy (Reviewer Priority)

| Research Component | Priority |
|--------------------|----------|
| Corrected architecture and valid training | Essential |
| Independent test scenarios | Essential |
| Multiple random seeds and uncertainty analysis | Very high |
| Comparison against BDF and data-only FNO | Essential |
| Ablation studies isolating each component | High |
| Resolution-transfer experiments | High |
| Runtime and memory benchmarking | High |
| Novelty comparison with existing literature | Essential |

## Key Constraint

> "The reported 60.5% paired win rate for Phase 4A.1 is worth investigating,
> but it is not sufficient by itself to establish a research contribution.
> Confirm it independently and across multiple seeds."

## What This Means

1. **Phase 0–4 remains LOCKED.** No reopen. Bug J documented, not fixed-in-place.
2. **Phase 4A.1's 60.5% data-accuracy win rate** — investigate independently, ≥5 seeds.
3. **The 828× physics claim** — retracted. Not restated in Phase 5 report.
4. **Phase 5.1 and 5.2** — abandoned. They would measure degenerate physics.
5. **Fresh architecture** — separate path, separate name (`phase_5_corrected`).
6. **Publication** — only when scientifically defensible. Not before.
7. **MIT NSE application** — built on honest investigation, not on impressive-looking claims.

## Phase 5 Restructured Execution Order

| Step | Original Plan | Revised (Option B) |
|------|---------------|-------------------|
| 5.A | — | **NEW:** Corrected architecture (`SCFNOModelV2`) |
| 5.B | 5.3A True-forcing physics | Rewrite as fresh architecture verification |
| 5.C | 5.3B Causal boundary | Preserved (fresh architecture) |
| 5.D | 5.4 Unseen + resolution + benchmark | Preserved |
| 5.E | 5.5 Final validation | **Expanded:** ≥5 seeds mandatory |
| — | 5.1 Corrected Phase 4B retraining | **Abandoned** |
| — | 5.2 Optimizer isolation | **Abandoned** |

## Commitment

- No modifications to Phase 0–4 evidence
- No modifications to Phase 3/4 checkpoints
- Bug J documented as discovery, not as failure
- Corrected architecture is a **new research direction**, not a patch
- Report will include Bug J as a first-class finding

