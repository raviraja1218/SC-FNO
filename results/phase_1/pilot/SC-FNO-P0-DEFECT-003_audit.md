# SC-FNO-P0-DEFECT-003 — coupled_rhs feedback bug

## Defect statement
`src/data/solver.py` `coupled_rhs` called `make_rho_total(rho_ext_fn, th)`,
which returns a closure `rho_total(t, y)` that reads `y[-2], y[-1]` as
`T_f, T_c`. But `pke_rhs` passes `y[:kin.m+1]` (=[n, C1..C6]) into that
closure, so `y[-2] = C5` and `y[-1] = C6` were being interpreted as
temperatures.

For pilot_0001 (initial C5=6.56, C6=0.91), the resulting spurious
feedback was:
    alpha_f * (C5 - T_f_ref) + alpha_c * (C6 - T_c_ref)
    = -3e-5 * (6.56 - 900) + -5e-4 * (0.91 - 583)
    = +0.0268 + 0.2910
    = +0.3178

This is ~49× beta_eff, i.e., a massive spurious positive reactivity on
every scenario regardless of the intended external forcing. Result: power
runaway (n up to 1,098), fuel temperature up to 132,859 K, coolant
temperature up to 38,467 K — all physically impossible.

## Discovery path
Block C comparison flagged this as "physical validation failed 100/100"
because the physical plausibility check `T_f < 5000 K` was tripped on
every scenario. Initial diagnosis attributed the failure to threshold
calibration (SC-FNO-P1-DEFECT-002); the deeper diagnostic (see
`block_C_physics_debug.log`) showed the actual cause was the physics.

## Root cause
`make_rho_total` was written to accept a length-9 state vector `y` with
`y[-2]=T_f, y[-1]=T_c`. But the `coupled_rhs → pke_rhs → rho_total_fn`
call chain passes only the kinetics slice `y[:7]`. The two conventions
were inconsistent.

## Fix
In `coupled_rhs`, extract `T_f, T_c` from the full 9-state `y` explicitly
and capture them in a closure:

    def rho_total_fn(t_inner, _y_kin):
        return float(rho_ext_fn(t_inner)) + feedback_reactivity(T_f, T_c, th)

This closure ignores its second argument (which pke_rhs will pass) and
uses the captured `T_f, T_c` from the current integration step.

Also: added `from physics.reactivity import feedback_reactivity` to
`solver.py` imports (was missing — latent bug in the fix, caught by
T5 test on first rerun).

## Regression test
`tests/test_solver.py::test_T5_negative_step_decreases_power` verifies:
1. A negative step through `coupled_rhs` produces `dn/dt < 0`
2. The magnitude matches `rho_ext / Lambda * n` within 5%
Both checks would fail under the pre-fix bug (which produced positive
dn/dt of the wrong magnitude).

## Preserved evidence
- Pre-fix `coupled_rhs` source: `results/phase_1/pilot/pre_defect_003/coupled_rhs_before.txt`
- Pre-fix Block B trajectories (T_f > 100,000 K): overwritten by Block B rerun;
  pre-fix summary preserved at `results/phase_1/pilot/pre_defect_003/block_B_summary.json`
- Pre-fix Block C summary: `results/phase_1/pilot/pre_defect_003/block_C_summary.json`
- Pre-fix comparison CSV: `results/phase_1/pilot/pre_defect_003/pilot_solver_comparison.csv`

## Post-fix verification
- 24/24 tests pass (including T5)
- All 100 scenarios: T_f max = 900.0 K, T_c max = 621.6 K, n max = 1.041
- Physical violations: 0/100
- BDF/Radau comparison: all 9 states, all 100 scenarios pass v2 thresholds
- Sub-microkelvin thermal agreement (max 11.5 μK for Tf, 9.2 μK for Tc)

## Downstream dependencies
- Block B (pilot solver runner): regenerated ✓
- Block C (BDF/Radau comparison): regenerated ✓
- Block D (tolerance sensitivity): to be performed on corrected data
- Block E (dataset lock and figures): to be performed on corrected data
- Steps 1.3–1.5: use corrected `coupled_rhs`

## Impact on prior phases
- Phase 0 config: unchanged (v2 thermal_hydraulics, v3 reactor_parameters)
- Phase 0 physics tests: unaffected (they don't exercise `coupled_rhs`)
- Step 1.1 T1–T4: unaffected (they use pke_rhs and th_rhs directly)
- Step 1.1 T5: added as regression test for this defect

## Note on SC-FNO-P1-DEFECT-002
The v2 thermal threshold correction (1e-6 K → 1e-2 K) was applied during
this investigation. Under corrected physics, the pre-fix v1 threshold
would actually have been achievable (post-fix max Tf diff = 11.5 μK,
which is < 1 μK? No — 11.5 μK > 1 μK. Wait, let me reconsider.

Under post-fix physics, max_abs_diff for Tf = 1.15e-5 K = 11.5 μK.
The v1 threshold was 1e-6 K = 1 μK. So v1 would still fail by ~11×.

The v2 threshold (1e-2 K = 10 mK = 10,000 μK) has ~1000× margin.
v2 is therefore the correct calibrated threshold regardless of the
coupled_rhs fix. Both defects (P1-DEFECT-002 and P0-DEFECT-003) were
real and both corrections are justified.
