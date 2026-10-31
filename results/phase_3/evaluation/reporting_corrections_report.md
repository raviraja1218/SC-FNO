# Phase 3.5 v2.1 — Reporting Corrections (Reviewer Response)

**Purpose:** Apply the five reporting corrections requested by the reviewer of the Step 3.5 completion report. No new evidence, no retraining, no test-set tuning. All analysis is computed from already-locked predictions.

---

## Correction 1 — Statistical Support for the "16.81%" Claim

### Original claim (retracted)

The Step 3.5 report stated:

> "PI-FNO achieves a 16.81% improvement over Phase 2 baseline on the untouched test set."

### What the paired bootstrap reveals

**Paired comparison on scenario-level aggregate Rel L2:**

| Metric | Value |
|--------|-------|
| PI-FNO wins | **47 / 200** |
| P2-FNO wins | **153 / 200** |
| Ties | 0 |
| **PI-FNO win rate** | **23.5%** |
| Median paired delta (P2 − PI) | **−1.595 × 10⁻⁵** (negative) |
| Mean paired delta | −1.888 × 10⁻⁵ |
| **95% CI of median paired delta** | **[−2.106 × 10⁻⁵, −1.938 × 10⁻⁶]** |

**The 95% CI excludes zero, but the sign is NEGATIVE.** This means:

> **PI-FNO loses to P2-FNO on 76.5% of test scenarios. The paired median delta is negative: PI-FNO is WORSE than P2-FNO.**

### Why the "16.81%" was misleading

- **Phase 2's 9.232 × 10⁻⁴** = median over 1,800 (scenario, state) pair-level Rel L2 values.
- **Step 3.5's 7.680 × 10⁻⁴** = median over 200 scenario-level aggregate Rel L2 values.
- **These are two different statistical quantities.**

The "16.81% improvement" compared incompatible statistics. Under the correct metric (paired per-scenario aggregate), **PI-FNO is worse than P2-FNO**, not better.

### Retracted claim

> **Step 3.5 does NOT demonstrate a statistically significant improvement over Phase 2. It demonstrates the opposite: PI-FNO underperforms P2-FNO on 76.5% of paired scenarios.**

---

## Correction 2 — Full Per-Family Table (All 4 Models)

### Complete table (P2-FNO now included)

| Family | P2-FNO | P-FNO | PI-FNO | P-PI-FNO |
|--------|--------|-------|--------|----------|
| step | 3.890 × 10⁻³ | 3.895 × 10⁻³ | 3.890 × 10⁻³ | 3.903 × 10⁻³ |
| **ramp** | **1.069 × 10⁻⁴** | 1.479 × 10⁻⁴ | 1.643 × 10⁻⁴ | 2.544 × 10⁻⁴ |
| **sinusoidal** | **3.946 × 10⁻⁴** | 5.221 × 10⁻⁴ | 4.473 × 10⁻⁴ | 6.058 × 10⁻⁴ |
| **grf** | **1.377 × 10⁻³** | 1.389 × 10⁻³ | 1.446 × 10⁻³ | 1.608 × 10⁻³ |

### Four isolated comparisons (as reviewer requested)

| Comparison | Isolates | Effect |
|-----------|----------|--------|
| **P2-FNO → PI-FNO** | Physics effect, no padding | ramp: **−53.7%** (worse); sinusoidal: **−13.4%** (worse); grf: **−5.0%** (worse); step: −0.02% (tied) |
| **P-FNO → P-PI-FNO** | Physics effect, with padding | ramp: −72.1%; sinusoidal: −16.0%; grf: −15.8%; step: −0.22% |
| **P2-FNO → P-FNO** | Padding effect, no physics | ramp: −38.3%; sinusoidal: −32.3%; grf: −0.8%; step: −0.14% |
| **PI-FNO → P-PI-FNO** | Padding effect, with physics | ramp: −54.8%; sinusoidal: −35.4%; grf: −11.2%; step: −0.34% |

### Honest interpretation

**Every isolated intervention (physics or padding) makes the test-set error WORSE on ramp, sinusoidal, and GRF.** The only family where effects are ~neutral is step.

**Scientific conclusion:** neither physics-informed training nor temporal padding improved test-set accuracy in these experiments. The step family is essentially unchanged (differences < 0.3%).

### Hypothesis (not established)

The claim that "physics helps sinusoidal because of delayed-neutron response" is a **plausible hypothesis**, not a mechanism established by these measurements. Family-level aggregates cannot isolate mechanism. Only per-scenario per-state residual diagnostics could, and those are not available at test time for the SC-FNO variants.

---

## Correction 3 — P2-FNO scenario_0458 Recomputed

### Previous claim (retracted)

> "scenario_0458 improves 55% with SC-FNO"

This compared Phase 2's 1.85e-2 (computed under a different aggregation convention) with Step 3.5's freshly computed ~8.17e-3.

### Recomputed under identical pipeline

| Model | scenario_0458 rel L2 |
|-------|---------------------|
| **P2-FNO** (recomputed same pipeline) | **8.1699 × 10⁻³** |
| P-FNO | 8.1791 × 10⁻³ |
| PI-FNO | 8.1700 × 10⁻³ |
| P-PI-FNO | 8.1805 × 10⁻³ |

### Honest interpretation

**All four models achieve ~8.17 × 10⁻³ on scenario_0458** — differences are within 1.3 × 10⁻⁵ (0.16% relative). **There is no meaningful improvement.**

**The 1.85e-2 Phase 2 value must be marked as "from Phase 2 aggregate metrics file, different computation convention"** — it is not comparable to the Step 3.5 numbers.

---

## Correction 4 — Throughput Reconciliation

### Phase 2 throughput (64,418 scenarios/s)

Source: `results/phase_2/closure/speedup_decomposition.json`
- **Batch size: 200**
- **Model: pure FNO forward**
- **Timing: batch total, model-only**
- Result: 64,418 scenarios/s

### Phase 3 throughput (1,863 scenarios/s for P2-FNO)

Source: `results/phase_3/evaluation/speedup_benchmark.json`
- **Batch size: 100**
- **Model: SCFNOModel wrapper** (FNO + decoder composition)
- **Timing: `predict_data_path` per-batch, includes Python overhead + sync**
- Result: 1,863 scenarios/s for P2-FNO

### Differences explaining the 35× discrepancy

| Factor | Phase 2 | Phase 3 |
|--------|---------|---------|
| Batch size | 200 | 100 |
| Model wrapper | Pure FNO | SCFNOModel + decoder |
| Sync per batch | once | per call |
| Python overhead | minimal | `predict_data_path` wrapper + numpy ↔ tensor conversion |
| Padding wrapper (for P/P-PI) | none | PaddedFNO adds `F.pad` + crop |

**The throughput numbers measure different things and must not be directly compared.**

### Authoritative reporting for manuscript

**Primary:** single-scenario latency speedup (~11× vs BDF) — **consistent between Phase 2 (11.54×) and Phase 3 (11.23×)**.
**Qualified:** throughput must always state batch size, model wrapper, and whether pipeline is included. Use Phase 3 numbers only for intra-Phase-3 comparisons (P2 vs P vs PI vs P-PI); do not compare to Phase 2's 64,418/s.

---

## Correction 5 — Resolution Terminology

### Renamed section

`Resolution Generalization` → **`Dense-Query Evaluation Stability`**

### Terminology note (added to report and JSON)

> This demonstrates that the coordinate decoder can evaluate a continuous-time representation at any query density without degradation. It does NOT demonstrate resolution-invariant physical accuracy.
>
> - The 3.48e-2 rel L2 measured on the physics path (decoder) is against a cubic-spline-interpolated reference derived from the 200-point Phase 1 ground truth.
> - Physics-path rel L2 (3.48e-2) is 45× higher than data-path rel L2 (7.68e-4). **These are different output paths and are not comparable.**
> - The Phase 1 reference grid is Δt = 0.050251 s; prompt timescale ≈ 0.031 s; **sub-grid dynamics remain unresolved.**

---

## Correction 6 — Gradient Conflict Evidence

### Actual numerical trajectory (not "resolved")

| Metric | PI-FNO | P-PI-FNO |
|--------|--------|----------|
| Reference cos (Step 3.3 init) | −0.239 | −0.239 |
| Min cos during training | **−0.239** | **−0.201** |
| Max cos during training | +0.259 | +0.333 |
| Final cos | +0.023 | +0.003 |
| Epochs below −0.15 | 7 / 66 | 3 / 83 |
| Epochs below −0.25 | 0 | 0 |

### Honest interpretation

**The gradient conflict did NOT resolve.** It oscillated between approximately −0.24 and +0.33 throughout training. Neither the warning threshold (−0.15, 3 consecutive epochs) nor the intervention threshold (−0.25, 3 consecutive epochs) triggered. The cos ended slightly positive but is within oscillation range.

**Retracted wording:** "conflict resolved through training" — this is unsupported. The correct statement is:

> "The gradient cosine oscillated between roughly −0.24 and +0.33 during training and ended near +0.02 for PI-FNO and +0.00 for P-PI-FNO. It did not monotonically improve. The conflict was not resolved; the persistence thresholds for intervention were not triggered."

---

## Step 3.5 Verdict — Corrected

**Step 3.5 remains operationally locked at `f6afd44`** (evaluation harness worked correctly; all predictions saved; all metrics computed). **The evaluation itself is sound; only the interpretation was corrected.**

### Corrected scientific conclusion

| Claim | Corrected status |
|-------|-----------------|
| PI-FNO improves test-set median Rel L2 by 16.81% | ❌ **RETRACTED** — compared incompatible statistics |
| PI-FNO is statistically better than P2-FNO | ❌ **FALSE** — PI-FNO loses on 153/200 scenarios (76.5%), negative median paired delta |
| Physics helps sinusoidal family | ❌ **FALSE** — P2-FNO (no physics) is best on sinusoidal |
| scenario_0458 improves 55% | ❌ **FALSE** — all four models achieve ~8.17e-3 on this scenario |
| Resolution invariance demonstrated | ❌ **RETRACTED** — renamed to "dense-query evaluation stability" |
| Gradient conflict resolved | ❌ **RETRACTED** — oscillated, did not resolve |
| Single-scenario BDF speedup ~11× | ✅ **RETAINED** — consistent across Phase 2 and Phase 3 |
| P-PI-FNO physics loss anomaly was a logging artifact | ✅ **RETAINED** — confirmed with 6-hypothesis audit |

### What Phase 3 actually demonstrates

**Physics-informed training, as implemented in Phase 3, does NOT improve test-set accuracy over the Phase 2 FNO baseline.**

- **P2-FNO (baseline, no physics) is the best model on the test set** by most robust criteria.
- **Physics training and padding each independently make ramp, sinusoidal, and GRF performance worse.**
- **The step family is essentially unchanged** across all variants.
- **The "16.81% improvement" was a metric mismatch artifact.**

**This is the honest scientific result.** It is a **negative result** on the central hypothesis that physics-informed training improves test-set generalization. It is still valuable as engineering — six gates of verified infrastructure — and as a documented negative finding for the research literature.

---

## Files Added by Correction v2.1

- `statistical_significance.json` — paired bootstrap + CI + win rates
- `per_family_full.json` — 4-model family table + 4 isolated comparisons
- `scenario_0458_recomputed.json` — P2 recomputed on identical pipeline
- `gradient_conflict_evidence.json` — actual cos trajectory numbers
- `p2_scenario_metrics.csv` — per-scenario metrics for P2-FNO (was missing)
- `throughput_protocols.log` — Phase 2 vs Phase 3 benchmark protocol comparison
- `reporting_corrections_report.md` — this document
- `phase_3_final_report.md` (updated) — resolution section renamed, terminology caveat added
- `resolution_generalization.json` (updated) — terminology note added
