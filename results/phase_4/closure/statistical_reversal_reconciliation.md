# Statistical Reversal Reconciliation

**Status:** AUTHORITATIVE — supersedes commit `3a04b3d`

## Issue
- Step 4A.3 report (`3a04b3d`): "Phase 3 wins 162/200" — **WRONG interpretation**
- Step 4A.3.R.2 report (`f7474c0`): "Phase 4A.1 wins 81% (162/200)" — **CORRECT**

## Metric Definition
`delta = phase3_rel_l2 − phase4a1_rel_l2` — positive means Phase 4A.1 better.

## Corrected Interpretation (Authoritative)
| Metric | Value |
|--------|-------|
| Phase 4A.1 better (delta > 0) | **162/200** |
| Phase 3 better (delta < 0) | **38/200** |
| Ties | 0/200 |
| **Phase 4A.1 win rate** | **81.0%** |
| Mean paired delta | +3.9488e-03 |
| Mean 95% CI | [+2.0999e-03, +5.7646e-03] |
| Median paired delta | +4.2955e-03 |
| Median 95% CI | [+4.5245e-04, +6.2038e-03] |

**Both CIs exclude zero → statistically significant.**

## Supersession
`f7474c0` is **authoritative**. `3a04b3d` had the win/loss labels flipped. Data unchanged.

All downstream references use corrected interpretation: **Phase 4A.1 wins 162/200 scenarios (81%).**
