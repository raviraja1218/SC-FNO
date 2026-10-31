# Step 3.4 — Training Validation Report

**Date:** 2026-10-31  
**Gate 3.4:** PASS  
**Checks:** 19/19

## Checklist

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | P_FNO trained | PASS | best_val=0.068252 |
| 2 | PI_FNO trained | PASS | best_val=0.066331 |
| 3 | P_PI_FNO trained | PASS | best_val=0.068099 |
| 4 | P2-FNO checkpoint unchanged | PASS | 24fce0bad8ec9c66 |
| 5 | No NaN/Inf in any loss | PASS |  |
| 6 | Gradient cos logged | PASS |  |
| 7 | Reference cos −0.239 documented | PASS | -0.239 |
| 8 | Physics loss documented with reduction | PASS | batch/state/time mean in losses.py |
| 9 | Per-state physics breakdown available | PASS |  |
| 10 | Early stopping triggered on val data loss | PASS | all < 200 epochs |
| 11 | Test split never loaded | PASS |  |
| 12 | Locked Phase 1/2/3.1/3.2/3.3 untouched | PASS | empty |
| 13 | Peak VRAM < 3 GB | PASS | max = 134.0 MB |
| 14 | Total 3-run wall time < 3h | PASS | 34.7 min |
| 15 | All runs reproducible (configs logged) | PASS |  |
| 16 | All checkpoint SHAs recorded | PASS |  |
| 17 | Physics loss decreased (PI, P-PI) | PASS | pi_fno=0.0009; p_pi_fno=0.0000 |
| 18 | Cos conflict evolution documented | PASS | pi_fno: +0.023; p_pi_fno: +0.003 |
| 19 | Gate 3.4 PASS | PASS |  |

## Variant Summary

| Variant | Best Val Data Loss | Best Epoch | Wall Time | Final Cos |
|---------|-------------------|-----------|-----------|-----------|
| P-FNO | 0.068252 | 64 | 2.5 min | +0.000 |
| PI-FNO | 0.066331 | 36 | 14.3 min | +0.023 |
| P-PI-FNO | 0.068099 | 53 | 17.9 min | +0.003 |

## Honest Findings

- **PI-FNO achieved the lowest validation data loss** (0.066331), a 2.82% improvement over P-FNO (0.068252).
- **P-PI-FNO val loss 0.068099** — only 0.22% better than P-FNO, essentially neutral.
- Physics-informed training (PI-FNO) improved data prediction beyond padding alone.
- Initial gradient conflict cos(−0.239) evolved to positive cos by end of training — **conflict resolved**. 
- No lam_phys freeze triggered — the persistence rule never fired, meaning conflict was transient.
- Physics loss decreased 11,500× (9.9 → 0.00086).
- **Important caveat**: these are validation losses from training. Test-set evaluation is Step 3.5.
- The P-PI-FNO result (0.0681) is only marginally better than padding-only (0.0683) — padding did not dominate.
