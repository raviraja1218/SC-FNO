# SC-FNO Phase Summary - Quick Reference

## Phase 0 - Environment (e7e70c0)
Working WSL2 + RTX 4050 + CUDA PyTorch. All checks PASS.

## Phase 1 - Dataset (1d9b320)
2,000 validated reactor transients via SciPy BDF.
Aggregate SHA 9049c7b4..., 16/16 gate PASS.

## Phase 2 - FNO Baseline (phase-2-lock)
Standard FNO trained on Phase 1 data.
108/108 gate PASS, test rel L2 median 9.232e-4, 11.54x speedup.

## Phase 3 - SC-FNO (phase-3-closed)
Physics-informed FNO attempt.
Scientific hypothesis FALSIFIED (PI-FNO loses 76.5% paired).
4 diagnosed failure mechanisms (M1-M4).

## Phase 4 - Adaptive Optimization (phase-4-locked)
Adam-to-L-BFGS + 4 intervention variants.
Phase 4A.1 claimed 828x physics, RETRACTED in Phase 5 due to Bug J.

## Phase 5 - Corrected Architecture (phase-5-locked)
Bug J/E/B/K fixes. First valid physics-informed model.
Selected model: models/sc_fno/final_sc_fno.pt

### Phase 5 Sub-Steps

5.A (ec9029e): SCFNOModelV2, 4 bug fixes, decoder health verified.
    du_dt std = 30.13 vs Bug J 1.4e-3 (21,500x improvement).

5.B (68c1ce1): 5 seeds, CV 0.41%, mean val 0.066468.
    Physics loss decreases 234 to 2.66 during training.

5.C (6bb3d12): Causal mask, 5/5 seeds B_n < 4.0x (mean 3.75x).
    Interior accuracy degrades 1.71% (within 5% tolerance).

5.D (093e5c0): 100 unseen, in-dist 2.33e-3, OOD 2.22e-2 (10x worse).
    BDF 27.73 ms vs SC-FNO 3.22 ms = 8.6x end-to-end speedup.

5.E (03a7402): Final model selected, Gate 5 18/18 PASS.
    Selected: Phase 5.B non-causal seed 42.
    Rationale: 12% better test accuracy than 5.C per rule.
