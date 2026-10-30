"""Step 1.4 Block D supplement — add missing plan analyses.

Adds:
  1. Per-family FFT spectra (plan §1.4.4 "frequency content per family")
  2. Scenario-level min/max table per state (plan §1.4.4 "scenario-level extrema")
  3. Literal-plan metrics (cosine-all-families, per-time-point ±5σ) side-by-side
     with corrected metrics, for honest reporting
"""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path('results/phase_1/splitting')
RAW = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"
N_TIME = 200
STATE_NAMES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
FAMILIES = ["step", "ramp", "sinusoidal", "grf"]


def load_all():
    with (R.parent / "generation" / "scenario_manifest_2000.csv").open() as f:
        meta = {r["scenario_id"]: r["family"] for r in csv.DictReader(f)}
    ids = sorted(meta)
    forcing = np.empty((len(ids), N_TIME))
    states = np.empty((len(ids), N_TIME, 9))
    fam = np.empty(len(ids), dtype=object)
    for i, sid in enumerate(ids):
        d = RAW / sid
        forcing[i] = np.load(d / "forcing.npy")
        states[i] = np.load(d / "trajectory.npy")
        fam[i] = meta[sid]
    return ids, forcing, states, fam


def per_family_fft(forcing, fam):
    """Per-family aggregate FFT spectra."""
    out = {}
    dt = 0.05
    freqs = np.fft.rfftfreq(N_TIME, d=dt)
    for f in FAMILIES:
        mask = fam == f
        fft_vals = np.fft.rfft(forcing[mask], axis=1)
        power = np.mean(np.abs(fft_vals) ** 2, axis=0)
        total = float(np.sum(power))
        # Normalized power per family
        out[f] = {
            "n_profiles": int(mask.sum()),
            "dominant_freq_hz": float(freqs[1 + int(np.argmax(power[1:]))]),
            "total_power": total,
            "power_spectrum": power.tolist(),
        }
    return {
        "frequencies_hz": freqs.tolist(),
        "per_family": out,
    }


def scenario_extrema(states, ids, fam):
    """Per-scenario min/max for each state (plan §1.4.4 "scenario-level extrema")."""
    rows = []
    for i, sid in enumerate(ids):
        row = {"scenario_id": sid, "family": fam[i]}
        for j, name in enumerate(STATE_NAMES):
            arr = states[i, :, j]
            row[f"{name}_min"] = float(np.min(arr))
            row[f"{name}_max"] = float(np.max(arr))
            row[f"{name}_mean"] = float(np.mean(arr))
        rows.append(row)
    return rows


def literal_vs_corrected_duplicates(forcing, ids, fam):
    """Literal plan metric (cosine all families, >0.99) vs corrected (family-aware)."""
    # Literal: cosine all
    norms = np.linalg.norm(forcing, axis=1)
    norms[norms < 1e-30] = 1e-30
    nz = forcing / norms[:, None]
    sim = nz @ nz.T
    # Count above 0.99 in upper triangle
    mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
    literal_count = int(np.sum((sim > 0.99) & mask))

    # Per-family breakdown of the literal count
    literal_by_fam = defaultdict(int)
    idx_upper = np.argwhere(mask)
    for i, j in idx_upper:
        if sim[i, j] > 0.99:
            literal_by_fam[f"{fam[i]}-{fam[j]}"] += 1

    return {
        "literal_metric_cosine_all_gt_0_99": {
            "total_pairs": literal_count,
            "by_family_pair": dict(literal_by_fam),
        },
        "corrected_metric_family_aware": json.load(open(R / "block_D_summary.json"))["n_duplicate_pairs"],
        "note": (
            "The literal plan metric (cosine > 0.99 across all families) "
            "counts proportional step/ramp profiles as duplicates because "
            "cosine similarity is 1.0 for any two proportional vectors. "
            "This is not a data defect — it is an artifact of using an "
            "angular metric on DC-dominated signals. The corrected metric "
            "uses L2 for step/ramp and cosine only for sinusoidal/GRF."
        ),
    }


def literal_vs_corrected_outliers(forcing, states, ids, fam):
    """Literal plan metric (per-time-point ±5σ) vs corrected (scenario aggregate)."""
    # Get training stats
    with (R / "train_ids.csv").open() as f:
        train_set = set(r["scenario_id"] for r in csv.DictReader(f))
    train_idx = [i for i, sid in enumerate(ids) if sid in train_set]

    f_train = forcing[train_idx]
    s_train = states[train_idx]

    train_stats = {"rho_ext": (float(np.mean(f_train)), float(np.std(f_train)))}
    for j, name in enumerate(STATE_NAMES):
        train_stats[name] = (
            float(np.mean(s_train[:, :, j])),
            float(np.std(s_train[:, :, j])),
        )

    # Literal metric: any time point where |x - mean| > 5σ (across all trajectories)
    literal_count = 0
    for i in range(len(ids)):
        hit = False
        m, sd = train_stats["rho_ext"]
        sd = max(sd, 1e-30)
        if np.max(np.abs(forcing[i] - m)) > 5 * sd:
            hit = True
        if not hit:
            for j, name in enumerate(STATE_NAMES):
                m, sd = train_stats[name]
                sd = max(sd, 1e-30)
                if np.max(np.abs(states[i, :, j] - m)) > 5 * sd:
                    hit = True
                    break
        if hit:
            literal_count += 1

    # Corrected metric already computed
    corrected = json.load(open(R / "block_D_summary.json"))["n_outlier_flagged"]

    return {
        "literal_metric_per_time_point_5sigma": literal_count,
        "corrected_metric_scenario_aggregate_5sigma": corrected,
        "note": (
            "The literal plan metric (any time point exceeding ±5σ) flags "
            "every transient that sweeps a range — which is every "
            "non-trivial transient. This is not a data defect; it is a "
            "metric that does not distinguish normal transients from "
            "anomalous scenarios. The corrected metric flags scenarios "
            "whose TIME-AVERAGED value is anomalous, which detects the "
            "genuine outliers the plan intended."
        ),
    }


def main():
    t0 = time.perf_counter()
    print("Loading 2,000 scenarios...")
    ids, forcing, states, fam = load_all()
    print(f"  Loaded {len(ids)}")

    print("Per-family FFT spectra...")
    fft_data = per_family_fft(forcing, fam)
    (R / "per_family_fft.json").write_text(json.dumps(fft_data, indent=2))

    print("Scenario-level extrema...")
    extrema = scenario_extrema(states, ids, fam)
    with (R / "scenario_extrema.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(extrema[0].keys()))
        w.writeheader()
        w.writerows(extrema)

    print("Literal vs corrected metric comparison...")
    dup_cmp = literal_vs_corrected_duplicates(forcing, ids, fam)
    out_cmp = literal_vs_corrected_outliers(forcing, states, ids, fam)

    # Supplementary report
    report = f"""# Step 1.4 Block D — Supplementary Analysis (Plan-Completeness)

Added after reviewing plan §1.4.4 completeness: this supplement provides the
analyses the plan asked for that were not fully delivered in Block D's first
pass, and documents the literal-plan metrics side-by-side with corrected
metrics for honest reporting.

## Analysis 1 — Per-family FFT spectra (plan requirement)

Previously delivered: aggregate PSD only.
This supplement: per-family PSD.

| Family | N profiles | Dominant freq (Hz) | Total power |
|--------|------------|-------------------|-------------|
"""
    for f_, v in fft_data["per_family"].items():
        report += (f"| {f_} | {v['n_profiles']} | {v['dominant_freq_hz']:.4f} | "
                   f"{v['total_power']:.4e} |\n")

    report += f"""
Full spectra in `per_family_fft.json`.

## Analysis 2 — Scenario-level extrema (plan requirement)

Previously: not extracted.
This supplement: `scenario_extrema.csv` contains, per scenario:
- min, max, mean of n, C1–C6, Tf, Tc

Total rows: {len(extrema)}. Columns: {len(extrema[0])}.

## Analysis 3 — Duplicate detection: literal vs corrected

### Literal plan metric (cosine > 0.99, all families)
- Total pairs flagged: **{dup_cmp['literal_metric_cosine_all_gt_0_99']['total_pairs']}**
- Per-family pair breakdown:
"""
    for pair, cnt in sorted(dup_cmp['literal_metric_cosine_all_gt_0_99']['by_family_pair'].items()):
        report += f"  - {pair}: {cnt}\n"

    report += f"""
### Corrected metric (family-aware)
- Total pairs flagged: **{dup_cmp['corrected_metric_family_aware']}**

### Explanation
{dup_cmp['note']}

## Analysis 4 — Outlier detection: literal vs corrected

### Literal plan metric (any time point |x − mean| > 5σ)
- Scenarios flagged: **{out_cmp['literal_metric_per_time_point_5sigma']}**

### Corrected metric (scenario aggregate |mean_t(x) − mean| > 5σ)
- Scenarios flagged: **{out_cmp['corrected_metric_scenario_aggregate_5sigma']}**

### Explanation
{out_cmp['note']}

## Wall time
{time.perf_counter() - t0:.1f} s
"""
    (R / "distribution_supplement.md").write_text(report)

    summary = {
        "block": "D-supplement (Step 1.4)",
        "wall_time_s": time.perf_counter() - t0,
        "per_family_fft": {k: {"dominant_freq": v["dominant_freq_hz"]}
                            for k, v in fft_data["per_family"].items()},
        "scenario_extrema_rows": len(extrema),
        "duplicate_literal": dup_cmp['literal_metric_cosine_all_gt_0_99']['total_pairs'],
        "duplicate_corrected": dup_cmp['corrected_metric_family_aware'],
        "outlier_literal": out_cmp['literal_metric_per_time_point_5sigma'],
        "outlier_corrected": out_cmp['corrected_metric_scenario_aggregate_5sigma'],
    }
    (R / "block_D_supplement_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
