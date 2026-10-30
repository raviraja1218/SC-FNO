# Phase 1 — Scientific Limitations

**Purpose:** Honest, complete documentation of what Phase 1's dataset can and cannot be used to claim. Written for the eventual manuscript's "Limitations" section and for Phase 2+ developers who will consume the dataset.

---

## 1. Model scope

### 1.1 Spatial dimensionality
The model is **lumped-parameter, zero-dimensional (0D)**. Fuel and coolant are treated as single thermal nodes. There is:
- **No** spatial dependence of neutron flux
- **No** spatial dependence of temperature
- **No** distributed fuel pin model
- **No** core-wide power distribution

**Consequence:** Phase 1 cannot be used to study spatial oscillations, xenon-induced power tilting, or localized thermal phenomena. Predictions are core-average only.

### 1.2 Neutronics
The model uses **point kinetics with 6 delayed-neutron groups**:
- **Not** neutron transport
- **Not** multi-group diffusion
- **Not** spectral effects beyond 6-group lumping
- **No** control rod worth curves (reactivity is treated as an external forcing)

**Consequence:** Cannot be used for detailed core design, spatial flux reconstruction, or analysis requiring energy-dependent neutron spectra.

### 1.3 Thermal-hydraulics
The model uses **two lumped thermal nodes (fuel + coolant)** with:
- **No** coolant sub-channel geometry
- **No** flow redistribution
- **No** two-phase flow or boiling
- **No** critical heat flux
- **No** multi-pass coolant flow

**Consequence:** Cannot be used for DNB analysis, two-phase instability studies, or detailed TH safety analysis.

### 1.4 Reactivity feedback
Feedback is **linear** with respect to temperature:
$$\rho_{\text{feedback}} = \alpha_f (T_f - T_{f,\text{ref}}) + \alpha_c (T_c - T_{c,\text{ref}})$$

**No** nonlinear reactivity effects, **no** burnup, **no** fuel depletion, **no** xenon dynamics, **no** samarium dynamics.

**Consequence:** Cannot be used for fuel cycle analysis or long-timescale operation simulation.

### 1.5 Time horizon
All scenarios use **t ∈ [0, 10] s**. This is appropriate for:
- Prompt-jump transients
- Fast reactivity insertion
- Initial power excursions

**Not** appropriate for:
- Thermal-hydraulic steady-state approach (timescale hours)
- Xenon transients (timescale hours)
- Fuel depletion (timescale years)

**Consequence:** Cannot be used for operational or fuel-cycle analysis.

---

## 2. Forcing coverage

### 2.1 Forcing families
Four families: **step, ramp, sinusoidal, Gaussian random field (GRF)**.

Each family uses specific parameter ranges (see `split_manifest.json`):
- Step amplitude: ±0.5 β_eff
- Ramp amplitude: ±0.3 β_eff, duration 0.5–5.0 s
- Sinusoidal amplitude: 0.05–0.30 β_eff, ω 0.05–20 rad/s
- GRF: RBF kernel, ℓ 0.05–3.0 s, amplitude_scale 0.02–0.25 β_eff

### 2.2 Family-specific limitations

**Step family:** Only t=0 insertions. Does not cover delayed step insertion (e.g., control rod drop at t > 0).

**Ramp family:** Only linear ramps from t=0. Does not cover piecewise ramps, exponential ramps, or delayed ramps.

**Sinusoidal family:** Single-frequency sinusoids with phase = 0. Does not cover multi-frequency, chirp, or modulated sinusoids.

**GRF family:** RBF kernel only. Biases toward smooth profiles. Does not cover:
- White noise (would need ℓ → 0)
- Power-law spectra
- Multi-scale variability
- Nonstationary processes

### 2.3 Only reactivity forcing
All forcing is via **external reactivity**. No scenarios perturb:
- Initial state
- Coolant inlet temperature
- Coolant flow rate
- Precursor concentrations independently

**Consequence:** Cannot be used to study forcing modes other than reactivity insertion.

---

## 3. Numerical limitations

### 3.1 Reference solver
- **BDF** at `rtol=1e-8, atol=1e-10` is the reference for all 2,000 scenarios
- **Radau** cross-check validates the reference on a **200-scenario sample (10%)**, not all 2,000
- Sub-20-μK thermal agreement between BDF and Radau across the sample

### 3.2 Tolerance sensitivity
- **40 scenarios (2%)** were rerun at `rtol=1e-10, atol=1e-12`
- Max relative L2 difference: **5.21e-7** (n) — **192× margin below 1e-4 acceptance**
- Confirms reference tolerance is sufficient **on the sample**; does not prove it for all 2,000

### 3.3 Numerical precision
- All arrays are `float64` (IEEE 754 double)
- Absolute error floors determined by `atol=1e-10`
- For thermal states at ~600 K, this corresponds to relative precision ~1.7e-13

**Consequence:** Trajectories are numerically converged but not exact. Downstream phases should treat the reference as one valid solution to the stiff system, not as ground truth in the strict sense.

---

## 4. Split limitations

### 4.1 Deterministic split
The split is **deterministic** based on a canonical parameter per family and seed `split_v1_2026`. Different seeds would produce different train/test sets.

**Consequence:** Reported test metrics depend on the split seed. Phase 2 must use the provided split files (`train_ids.csv` etc.) — not the seed alone — to reproduce.

### 4.2 Stratification by input parameter
The split is stratified by **input parameter** (amplitude, ω, or ℓ). It is **not** stratified by output trajectory statistics.

**Consequence:** The train/val/test sets have slightly different distributions of nonlinear-composite outputs (n, Tf, Tc). KS tests show statistically significant differences for these variables (p → 0), even though parameter ranges are nearly identical. This is **expected and desirable** — it means the test set genuinely tests generalization across the parameter space, not memorization.

### 4.3 Per-family stratification
Each family has 400/50/50 split. The test set has exactly 50 per family, giving 200 test scenarios total.

**Consequence:** Any family-specific analysis has only 50 test samples. Statistical power is limited for family-specific claims.

### 4.4 No cross-validation
The split is a single train/val/test fold. No k-fold cross-validation is provided.

**Consequence:** Reported test metrics carry split-seed variance. Manuscript claims should state this.

---

## 5. Normalization limitations

### 5.1 Training-only
Normalization statistics are computed from **training data only** (1,600 scenarios).

### 5.2 Global + per-time-point
Both global (mean/std over all time points) and per-time-point (mean/std at each t) statistics are provided.

### 5.3 Near-zero variance
**No variables have near-zero variance** (all std > 1e-6).

### 5.4 Method
z-score `(x - mean) / std` is the primary method. Min-max is available as fallback.

**Consequence:** Phase 2 must apply normalization **using the provided statistics only**. Recomputing normalization from val/test would violate the frozen evidence.

---

## 6. What CAN be claimed

Phase 1 supports the following claims:

1. **"A verified 9-state coupled reactor model was implemented and validated"** — four analytical tests pass to specified tolerances.
2. **"A 2,000-scenario dataset was generated using BDF with cross-check on 200 scenarios"** — both BDF and Radau solutions exist; Radau agrees with BDF on the sample.
3. **"The dataset has 500 scenarios per family across 4 forcing families"** — verified.
4. **"A stratified 1,600/200/200 split was created without time-point leakage"** — verified.
5. **"Training-only normalization statistics are available"** — verified.
6. **"All 2,000 trajectories are physically plausible within the model's operating envelope"** — verified.

## 7. What CANNOT be claimed

Phase 1 does **not** support:

1. **Spatial predictions** — this is a 0D model.
2. **Multi-group neutronics** — point kinetics only.
3. **Two-phase thermal-hydraulics** — single-phase, 2-node.
4. **Long-timescale dynamics** — only 10-second horizon.
5. **Families beyond the 4 defined** — no other forcing types.
6. **Ground-truth beyond numerical accuracy** — the reference is one numerically converged solution, not exact physics.
7. **Generalization across split seeds** — the split is deterministic; different seeds would give different metrics.
8. **Fuel cycle / burnup** — no depletion.
9. **Control rod worth analysis** — external reactivity only.
10. **Cross-validation variance** — single fold only.

---

## 8. Manuscript-ready limitations paragraph

For the eventual SC-FNO manuscript, the Limitations section should state:

> "The Phase 1 dataset was generated using a lumped-parameter 0D model: point kinetics with 6 delayed-neutron groups coupled to a two-node thermal-hydraulics representation. Spatial effects, neutron transport, multi-phase flow, and nonlinear feedback mechanisms are outside the model scope. Forcing is restricted to four families (step, ramp, sinusoidal, and Gaussian random field) of external reactivity within specified amplitude ranges, with a fixed 10-second simulation horizon. The numerical reference is BDF at rtol=1e-8, atol=1e-10, cross-checked against Radau on a 10% sample (200 scenarios); tolerance sensitivity was validated on a 2% sample (40 scenarios) with 192× margin below the acceptance threshold. The 1,600/200/200 split is deterministic with seed `split_v1_2026` and stratified by input parameter per family. Reported test metrics are specific to this seed and split; family-specific analyses have n=50 test scenarios, limiting statistical power. All 2,000 trajectories are physically plausible within the model's defined operating envelope. Downstream claims about spatial fidelity, two-phase behavior, long-timescale dynamics, or forcing families beyond the four defined are not supported by this dataset."

---

## 9. Honest scientific position

Phase 1 establishes a **verified numerical solver** and a **numerically converged, physically plausible dataset** for a **restricted model class**: 0D coupled point kinetics + 2-node thermal-hydraulics with linear reactivity feedback over a 10-second horizon. Within this class, the dataset is a defensible ground truth for training and evaluating the SC-FNO operator.

Outside this class, the dataset is not a substitute for full-core, multi-physics, or multi-phase analysis.

All downstream phases should:
- Reference the frozen dataset via the authoritative manifest
- Use the provided split and normalization without modification
- Interpret model performance in the context of Phase 1's model class
- Not overstate the physical fidelity of the underlying model in the manuscript
