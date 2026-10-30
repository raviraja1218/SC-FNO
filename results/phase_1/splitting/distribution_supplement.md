# Step 1.4 Block D — Supplementary Analysis (Plan-Completeness)

Added after reviewing plan §1.4.4 completeness: this supplement provides the
analyses the plan asked for that were not fully delivered in Block D's first
pass, and documents the literal-plan metrics side-by-side with corrected
metrics for honest reporting.

## Analysis 1 — Per-family FFT spectra (plan requirement)

Previously delivered: aggregate PSD only.
This supplement: per-family PSD.

| Family | N profiles | Dominant freq (Hz) | Total power |
|--------|------------|-------------------|-------------|
| step | 500 | 0.5000 | 1.4079e-01 |
| ramp | 500 | 0.1000 | 4.2645e-02 |
| sinusoidal | 500 | 0.1000 | 1.8114e-02 |
| grf | 500 | 0.1000 | 3.1737e-02 |

Full spectra in `per_family_fft.json`.

## Analysis 2 — Scenario-level extrema (plan requirement)

Previously: not extracted.
This supplement: `scenario_extrema.csv` contains, per scenario:
- min, max, mean of n, C1–C6, Tf, Tc

Total rows: 2000. Columns: 29.

## Analysis 3 — Duplicate detection: literal vs corrected

### Literal plan metric (cosine > 0.99, all families)
- Total pairs flagged: **106434**
- Per-family pair breakdown:
  - grf-grf: 1
  - ramp-grf: 99
  - ramp-ramp: 31712
  - ramp-sinusoidal: 1182
  - sinusoidal-grf: 40
  - sinusoidal-sinusoidal: 590
  - step-grf: 480
  - step-ramp: 2940
  - step-sinusoidal: 7040
  - step-step: 62350

### Corrected metric (family-aware)
- Total pairs flagged: **17**

### Explanation
The literal plan metric (cosine > 0.99 across all families) counts proportional step/ramp profiles as duplicates because cosine similarity is 1.0 for any two proportional vectors. This is not a data defect — it is an artifact of using an angular metric on DC-dominated signals. The corrected metric uses L2 for step/ramp and cosine only for sinusoidal/GRF.

## Analysis 4 — Outlier detection: literal vs corrected

### Literal plan metric (any time point |x − mean| > 5σ)
- Scenarios flagged: **180**

### Corrected metric (scenario aggregate |mean_t(x) − mean| > 5σ)
- Scenarios flagged: **0**

### Explanation
The literal plan metric (any time point exceeding ±5σ) flags every transient that sweeps a range — which is every non-trivial transient. This is not a data defect; it is a metric that does not distinguish normal transients from anomalous scenarios. The corrected metric flags scenarios whose TIME-AVERAGED value is anomalous, which detects the genuine outliers the plan intended.

## Wall time
1.9 s
