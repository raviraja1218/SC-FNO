# Phase 1 — Step 1.3 (2,000-Scenario Production Dataset) Completion Report

## Status
Step 1.3 is complete. All blocks A–F are locked. Gate 1.3 passes.

## Scope delivered
- 2,000 production scenarios (500 step, 500 ramp, 500 sinusoidal, 500 GRF)
- Each solved with BDF (rtol=1e-8, atol=1e-10) on 200-point grid [0, 10] s
- 9 output states: [n, C1..C6, Tf, Tc]
- Radau cross-check on 200-scenario sample (10%)
- Tolerance sensitivity on 40-scenario sample (2%)
- Physical plausibility validation on all 2,000
- 7 production figures + 1 from Block A
- Dataset summary JSON + manuscript Table 3 data
- Zero silent discards

## Blocks and evidence

| Block | Content | Commit |
|-------|---------|--------|
| A | Production registry (2,000 scenarios, 500/family) | 7d00c3f |
| B | Production solver — 2,000/2,000 BDF trajectories | ec14693 |
| C | Radau cross-check (200) + tolerance sweep (40) | e7a2b42 |
| D | Full physical validation (2,000) + runtime benchmark | 7c6c20c |
| E | 7 production figures + dataset summary + Table 3 data | 064b214 |
| F | Gate 1.3 verification + completion report + manifest | (this commit) |

## Defects carried forward from Step 1.2

Both defects from Step 1.2 remain resolved with evidence preserved:
- **SC-FNO-P1-DEFECT-002** — thermal threshold calibration (v2 methodology)
- **SC-FNO-P0-DEFECT-003** — `coupled_rhs` feedback bug (physics corrected, T5 regression test in place)

**No new defects discovered in Step 1.3.** The corrected `coupled_rhs` (checksum `3f51d56743a753af...`) was used throughout all 2,000 scenarios and verified at Block A start.

## Dataset characteristics

### Composition
| Family | Count | Parameter ranges |
|--------|-------|------------------|
| Step | 500 | ρ_ext ∈ [−0.5, +0.5] β_eff, t_insert = 0 |
| Ramp | 500 | ρ ∈ [−0.3, +0.3] β_eff, duration log-uniform [0.5, 5.0] s |
| Sinusoidal | 500 | ρ ∈ [0.05, 0.30] β_eff, ω log-uniform [0.05, 20] rad/s |
| GRF | 500 | length_scale log-uniform [0.05, 3.0] s, amplitude_scale [0.02, 0.25] β_eff |
| **Total** | **2,000** | |

### Numerical configuration
- Primary solver: BDF, rtol=1e-8, atol=1e-10
- Cross-check: Radau, same tolerances, 200-sample
- Time grid: 200 uniform points on [0, 10] s
- Configs: reactor_parameters v3, thermal_hydraulics v2
- Source term: q(t) = 0, include_q=True
- Initial condition: Phase 0 locked state (n₀=1, Tf₀=900K, Tc₀=583K, C_i0 equilibrium)

### Physical bounds (all 2,000 scenarios)
| State | Global range |
|-------|--------------|
| n | [0.241, 1.287] |
| C_i | [0.276, 467.2] |
| Tf | [673.7 K, 900.0 K] |
| Tc | [583.0 K, 622.8 K] |

### Physical validation
- **2,000/2,000 scenarios pass** physical plausibility (n ≥ 0, C_i ≥ 0, Tf < 5000 K, Tc < 2000 K)
- **0 violations**

## Numerical validation

### BDF/Radau agreement (200-scenario sample)
- All 9 states, all 200 scenarios: **PASS** v2 thresholds
- Max Tf abs diff: 1.74e-5 K (17.4 μK)
- Max Tc abs diff: 8.50e-6 K (8.5 μK)
- Max n abs diff: 9.19e-7
- All thermal margins >500× below threshold

### Tolerance sensitivity (40-scenario sample)
- All 9 states, all 40 scenarios: **PASS** 1e-4 rel L2 acceptance
- Max relative L2: 5.21e-7 (n), **192× below threshold**
- Reference tolerance (1e-8, 1e-10) confirmed sufficient

## Runtime benchmark

### Classical numerical baseline (for Phase 5 comparison)
| Metric | Value |
|--------|-------|
| Total BDF wall time | **788.2 s** (~13.1 min) |
| Mean per scenario | 394 ms |
| Median per scenario | **61.3 ms** |
| Min / max | 42 ms / 1.53 s |
| p90 / p95 | 1.12 s / 1.21 s |
| Mean nfev | 7,841 |
| Median nfev | 1,241 |

### Per-family runtime
| Family | Mean (s) | Median (s) | Total (s) |
|--------|----------|------------|-----------|
| Step | 0.048 | 0.048 | 24.1 |
| Ramp | 0.056 | 0.056 | 27.9 |
| Sinusoidal | 0.596 | 0.560 | **298.1** |
| GRF | 0.876 | 0.886 | **438.1** |

**Sinusoidal and GRF families dominate (94% of total runtime)** — expected due to higher-frequency forcing driving more BDF steps.

## Manuscript extracts

### Table 3 — Dataset composition and characteristics
Data in `manuscript_table_3_data.csv` (5 rows: 4 families + ALL aggregate).
Columns: family, n_scenarios, n bounds, Tf bounds, Tc bounds, runtime stats.

### Figure 2 — Representative input reactivity + 9-state responses
- `production_responses_composite.png` — 4 families × 3 panels (n, ΣC, Tf+Tc)
- `production_input_reactivity.png` — matching input ρ_ext(t)/β_eff

## Gate 1.3 — 15 criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Exactly 2,000 scenarios | ✅ |
| 2 | 4 families × 500 each | ✅ |
| 3 | 2,000 matched input-output pairs | ✅ |
| 4 | 9 output states per scenario | ✅ |
| 5 | 200 time points per trajectory | ✅ |
| 6 | Reproducible metadata (seeds) | ✅ |
| 7 | Original solver outputs preserved | ✅ |
| 8 | No missing/mismatched IDs | ✅ |
| 9 | Per-scenario diagnostics + timing | ✅ |
| 10 | No silent discards | ✅ |
| 11 | Physical plausibility (2,000) | ✅ |
| 12 | BDF/Radau agreement (200 sample) | ✅ |
| 13 | Tolerance sensitivity (40 sample) | ✅ |
| 14 | Runtime benchmark recorded | ✅ |
| 15 | Manifest with SHA-256 | ✅ (see step_1_3_manifest.json) |

**GATE 1.3: PASS.**

## Authorization

Step 1.3 completion authorizes **Step 1.4**: dataset validation, processing, and scenario-level splitting (1,600 train / 200 val / 200 test). The locked 2,000-scenario dataset (manifest SHA-256 to be recorded) becomes the fixed scientific reference for all downstream training and evaluation.

## Preservation notes

- **Pilot data** (`~/Research/Datasets/SC-FNO/raw/pilot_100/`) — untouched, preserved for reference
- **Step 1.2 evidence** (`results/phase_1/pilot/`) — untouched, preserved
- **Two Step 1.2 defects** — resolved with audit trails preserved
- **No modifications to Step 1.2 pilot data, solver, or evidence** during Step 1.3

The 2,000-scenario production dataset is a **new artifact** generated on the corrected physics foundation. It does not overwrite or modify any prior phase evidence.
