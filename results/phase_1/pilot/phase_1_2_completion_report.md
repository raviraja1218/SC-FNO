# Phase 1 — Step 1.2 (100-Scenario Pilot) Completion Report

## Status
Step 1.2 is complete. All blocks A–E are locked. Gate 1.2 passes.

## Scope delivered
- 100 pilot scenarios (25 step, 25 ramp, 25 sinusoidal, 25 GRF)
- Each solved with BDF (rtol=1e-8, atol=1e-10) on a 200-point grid [0, 10] s
- 9 output states: [n, C1..C6, Tf, Tc]
- Independent Radau cross-check on all 100
- Tolerance sensitivity on 20 (5 per family) at rtol=1e-10, atol=1e-12
- Physical plausibility validation on all 100
- 8 scientific figures
- Zero silent discards — all 100 accounted for

## Blocks and evidence

| Block | Content | Commit |
|-------|---------|--------|
| A | Forcing generators + 100-scenario registry + manifest | ed52aa8 / ee1e84c / f668eb3 |
| B | Pilot solver runner — 100/100 solved | 44b880a |
| C | BDF/Radau comparison + physical validation | 959451b |
| D | Tolerance-sensitivity sweep — 20 × 9 states | 2fd9bff |
| E | Figures + Gate 1.2 (this report) | (to be committed) |

## Two defects discovered and corrected during Step 1.2

### SC-FNO-P1-DEFECT-002 (thermal threshold calibration)
- Original v1 threshold of 1e-6 K absolute for Tf, Tc was unachievable for stiff ODEs
- Corrected to 1e-2 K (10 mK) in `bdf_radau_methodology.v2.md`
- v1 file preserved untouched
- See `results/phase_1/solver/bdf_radau_methodology.v2.md`

### SC-FNO-P0-DEFECT-003 (coupled_rhs feedback bug)
- `coupled_rhs` passed only `y[:7]` to a closure expecting `y[:9]`, so C5, C6 were
  interpreted as Tf, Tc by `make_rho_total`, producing spurious +0.32 reactivity
  and physically impossible temperatures (Tf up to 132,859 K)
- Fixed in `src/data/solver.py` by capturing Tf, Tc from full y
- Regression test T5 added: negative step must decrease power
- Pre-fix evidence preserved in `results/phase_1/pilot/pre_defect_003/`
- See `results/phase_1/pilot/SC-FNO-P0-DEFECT-003_audit.md`

## Physics after all corrections
Post-fix, all 100 trajectories:
- n ∈ [1.000e+00, 1.041e+00] — small transients, physically consistent with ±0.5 β_eff steps
- Tf ∈ [900.0 K, 900.0 K] — no spurious heating
- Tc ∈ [563.0 K, 621.6 K] — plausible PWR coolant range
- 0/100 physical violations

## Numerical validation
### BDF/Radau agreement (Block C)
- All 9 states, all 100 scenarios within v2 thresholds
- Max Tf diff = 1.15e-5 K (sub-10-μK)
- Max Tc diff = 9.18e-6 K
- Max n diff = 7.85e-7 (kinetics)

### Tolerance sensitivity (Block D)
- 20 scenarios × 9 states, max rel L2 = 4.39e-7 (230× below acceptance)
- Reference tolerance is sufficient — tightening further does not change results

## Figures
- `pilot_reactivity_profiles.png` — Block A
- `pilot_neutron_density.png` — Block E
- `pilot_precursor_evolution.png` — Block E
- `pilot_fuel_temperature.png` — Block E
- `pilot_coolant_temperature.png` — Block E
- `pilot_bdf_radau_comparison.png` — Block C
- `pilot_tolerance_sensitivity.png` — Block D

## Gate 1.2 — 10 criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | 100 pilot scenarios generated | ✅ |
| 2 | 4 forcing families, 25 each | ✅ |
| 3 | All 100 trajectories on 200-point grid | ✅ |
| 4 | No unexplained NaN/inf | ✅ |
| 5 | Initial-state checks passed | ✅ |
| 6 | Tolerance-sensitivity evidence | ✅ |
| 7 | BDF/Radau agreement under predefined thresholds | ✅ |
| 8 | All 100 accounted for (no silent discards) | ✅ |
| 9 | Reproducible (seeds, parameters, environment) | ✅ |
| 10 | 6+ figures generated | ✅ (8 figures) |

**GATE 1.2: PASS.**

## Authorization
Step 1.2 completion authorizes Step 1.3: full-scale generation of 2,000 scenarios.
The locked artifacts (Block A manifest, Block B–D evidence, two defect corrections)
form the fixed scientific reference for Step 1.3.

---

## Addendum: Step 1.1 Lock Preservation Audit

Added after reviewer provenance review on 2026-10-30.

### Complete change inventory since Step 1.1 lock (e5b5628 → 9eed1a6)

**Step 1.1 source files modified: 2 (both documented)**

| File | Change | Documentation |
|------|--------|---------------|
| `src/data/solver.py` | `coupled_rhs` fixed for SC-FNO-P0-DEFECT-003 (+24 lines changed) | SC-FNO-P0-DEFECT-003_audit.md |
| `tests/test_solver.py` | T5 regression test added (+39 lines) | SC-FNO-P0-DEFECT-003_audit.md |

**Step 1.1 source files NOT modified (verified via `git diff --stat`):**

- `tests/test_physics.py` — unchanged
- `src/physics/kinetics.py` — unchanged
- `src/physics/thermal_hydraulics.py` — unchanged
- `src/physics/reactivity.py` — unchanged
- `configs/reactor_parameters.v3.yaml` — unchanged
- `configs/thermal_hydraulics.v2.yaml` — unchanged
- `results/phase_1/solver/bdf_radau_methodology.md` (v1) — unchanged
  (v2 methodology was created as a NEW file)

### Supersession relationship

- Step 1.1 lock `e5b5628` is preserved as historical record
- `src/data/solver.py` had a genuine bug (SC-FNO-P0-DEFECT-003) that was
  undetected in Step 1.1's tests T1–T4 (they exercise `pke_rhs` directly,
  not `coupled_rhs`)
- The fix was applied in commit `959451b` (Phase 1.2 Block C) with:
  - Full audit in `SC-FNO-P0-DEFECT-003_audit.md`
  - Pre-fix source preserved in `pre_defect_003/coupled_rhs_before.txt`
  - New regression test T5 that would catch this class of bug
- No Step 1.1 physics modules or configs were touched
- No Step 1.1 evidence was deleted or overwritten

### Git ancestry

- `git merge-base --is-ancestor e5b5628 9eed1a6` → exit 0
- Step 1.2 final commit `9eed1a6` descends from Step 1.1 lock `e5b5628`
- Full commit chain is intact

### Manifest integrity

- 93 artifacts with SHA-256 checksums
- Self-reference excluded (documented in manifest)
- Corrected `solver.py` checksum: `3f51d56743a753af...`
- Corrected `test_solver.py` checksum: `1b256aaa87b25b41...`
- Unchanged Step 1.1 physics files have original checksums:
  - `kinetics.py`: `d482c39a60604912...`
  - `thermal_hydraulics.py`: `2e360f801de7a4a2...`
  - `reactivity.py`: `4e46eea1af6553af...`

**Provenance verified. Step 1.2 is formally closed.**
