# T2 Reviewer Correction Audit — SC-FNO-REV-001

## Defect ID
SC-FNO-REV-001

## Reviewer Finding
The original T2 test compared solver output to

    n(t) = n0 * exp((rho - beta_eff)/Lambda * t)

but with equilibrium initial precursors C_i(0) = beta_i*n0/(Lambda*lambda_i),
the frozen-precursor neutron equation still contains a nonzero delayed source

    S = sum_i lambda_i*C_i(0) = beta_eff*n0/Lambda != 0.

The exact solution of dn/dt = a*n + S (a constant, S constant) is

    n(t) = (n0 + S/a)*exp(a*t) - S/a,

NOT n0*exp(a*t). The naive exponential is only correct when S = 0, which
is not the case here. Therefore the original T2 could not be a genuine test
of the frozen-precursor ODE solution — it compared against a formula that
does not match the system being integrated.

## Resolution
T2 was rewritten to:
1. Construct the frozen-precursor RHS as dn/dt = a*n + S with S computed
   from C_i(0) explicitly (never hardcoded).
2. Integrate with BDF over 3 prompt time constants (tau = 1/|a|).
3. Compare the numeric trajectory to the exact analytical
   n(t) = (n0 + S/a)*exp(a*t) - S/a over the full window, not to a
   near-zero limit.
4. Report relative L2 error over the whole trajectory with tolerance 1e-6.

Additionally, the algebraic prompt-jump ratio beta_eff/(beta_eff - rho)
is verified independently as a parameter check (no ODE involved).

## Verification
- pytest tests/ -v -> 9/9 PASS.
- analytical_test_results.json records SC-FNO-REV-001 in corrections_applied.

## Downstream Impact
- T1, T3, T4 unchanged — they do not depend on this defect.
- Phase 0 correction (SC-FNO-P0-DEFECT-001) unaffected.
- Step 1.1 evidence package updated; prior commit b8c3676 / 49c67f0 preserved
  in git history as the pre-correction state.

## Approval
Reviewer audit accepted; correction executed per project owner's instruction
"follow the plan, produce results, no options."
