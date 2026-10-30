"""Step 1.4 Block D — distribution + outlier analysis.

Read-only across train/val/test splits and all 2,000 scenarios.
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
from scipy.stats import ks_2samp

R = Path('results/phase_1/splitting')
RAW = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"
N_TIME = 200
STATE_NAMES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
FAMILIES = ["step", "ramp", "sinusoidal", "grf"]


def load_split_ids():
    def _load(p):
        with p.open() as f:
            return [r["scenario_id"] for r in csv.DictReader(f)]
    return {
        "train": _load(R / "train_ids.csv"),
        "val": _load(R / "validation_ids.csv"),
        "test": _load(R / "test_ids.csv"),
    }


def load_metadata_family():
    """Map scenario_id -> family from Step 1.3 manifest."""
    m = {}
    with (R.parent / "generation" / "scenario_manifest_2000.csv").open() as f:
        for r in csv.DictReader(f):
            m[r["scenario_id"]] = r["family"]
    return m


def load_all_forcing_and_states():
    """Load 2000 forcing arrays + 2000 trajectories."""
    meta = load_metadata_family()
    ids = sorted(meta.keys())
    forcing = np.empty((len(ids), N_TIME), dtype=np.float64)
    states = np.empty((len(ids), N_TIME, 9), dtype=np.float64)
    fam_arr = []
    for i, sid in enumerate(ids):
        d = RAW / sid
        forcing[i] = np.load(d / "forcing.npy")
        states[i] = np.load(d / "trajectory.npy")
        fam_arr.append(meta[sid])
    return ids, forcing, states, np.array(fam_arr), meta


def global_stats_by_family(forcing, states, fam_arr):
    """Per-family min/max/mean/std for each variable."""
    out = {}
    for fam in FAMILIES:
        mask = fam_arr == fam
        fam_stats = {}
        # forcing
        f = forcing[mask]
        fam_stats["rho_ext"] = {
            "mean": float(np.mean(f)),
            "std": float(np.std(f)),
            "min": float(np.min(f)),
            "max": float(np.max(f)),
        }
        for j, name in enumerate(STATE_NAMES):
            s = states[mask, :, j]
            fam_stats[name] = {
                "mean": float(np.mean(s)),
                "std": float(np.std(s)),
                "min": float(np.min(s)),
                "max": float(np.max(s)),
            }
        out[fam] = fam_stats
    return out


def fft_frequency_analysis(forcing):
    """FFT of forcing profiles: aggregate power spectrum."""
    # Mean power spectral density across all 2000
    ffts = np.fft.rfft(forcing, axis=1)
    power = np.mean(np.abs(ffts) ** 2, axis=0)  # (n_freq,)
    freqs = np.fft.rfftfreq(forcing.shape[1], d=0.05)  # sampling interval = 0.05 s
    return {
        "n_freq_bins": int(len(freqs)),
        "freq_min": float(freqs[1]) if len(freqs) > 1 else 0.0,
        "freq_max": float(freqs[-1]),
        "dominant_freq": float(freqs[1 + int(np.argmax(power[1:]))]),
        "power_spectrum": power.tolist(),
        "frequencies": freqs.tolist(),
    }


def ks_cross_split(train_vals, val_vals, test_vals):
    """KS statistic + p-value for train vs val, train vs test."""
    stat_v, p_v = ks_2samp(train_vals, val_vals)
    stat_t, p_t = ks_2samp(train_vals, test_vals)
    return {
        "train_vs_val": {"ks_stat": float(stat_v), "p_value": float(p_v)},
        "train_vs_test": {"ks_stat": float(stat_t), "p_value": float(p_t)},
    }


def cosine_duplicates(forcing, ids, threshold=0.99):
    """Find pairs of forcing profiles with cosine similarity > threshold."""
    # Normalize each row
    norms = np.linalg.norm(forcing, axis=1)
    norms[norms < 1e-30] = 1e-30
    normalized = forcing / norms[:, None]
    # Gram matrix
    sim = normalized @ normalized.T
    # Get pairs above threshold, excluding diagonal and (i<j) to avoid duplicates
    n = len(ids)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > threshold:
                pairs.append({
                    "id_i": ids[i], "id_j": ids[j],
                    "cosine": float(sim[i, j]),
                })
    return pairs


def flag_outliers(forcing, states, ids, fam_arr, threshold_sigma=5.0):
    """Flag scenarios where any state exceeds ±threshold_sigma from training mean."""
    # Use global training stats
    train_mask = np.ones(len(ids), dtype=bool)  # placeholder; will filter below
    # Simpler: use overall per-variable mean/std (post-split, "training-like")
    # Better: use actual training split IDs
    with (R / "train_ids.csv").open() as f:
        train_set = set(r["scenario_id"] for r in csv.DictReader(f))

    train_idx = [i for i, sid in enumerate(ids) if sid in train_set]
    f_train = forcing[train_idx]
    s_train = states[train_idx]

    # Per-variable train mean/std
    train_stats = {"rho_ext": (float(np.mean(f_train)), float(np.std(f_train)))}
    for j, name in enumerate(STATE_NAMES):
        train_stats[name] = (
            float(np.mean(s_train[:, :, j])),
            float(np.std(s_train[:, :, j])),
        )

    flags = []
    for i, sid in enumerate(ids):
        # Check each variable
        outliers = []
        # forcing
        m, sd = train_stats["rho_ext"]
        sd_safe = max(sd, 1e-30)
        z_max = float(np.max(np.abs(forcing[i] - m))) / sd_safe
        if z_max > threshold_sigma:
            outliers.append(f"rho_ext: z_max={z_max:.2f}")

        # states
        for j, name in enumerate(STATE_NAMES):
            m, sd = train_stats[name]
            sd_safe = max(sd, 1e-30)
            z = np.abs(states[i, :, j] - m) / sd_safe
            z_max = float(np.max(z))
            if z_max > threshold_sigma:
                outliers.append(f"{name}: z_max={z_max:.2f}")

        if outliers:
            flags.append({
                "scenario_id": sid,
                "family": fam_arr[i],
                "outlier_flags": "; ".join(outliers),
            })

    return flags, train_stats


def main():
    t0 = time.perf_counter()

    print("Loading all 2,000 scenarios...")
    ids, forcing, states, fam_arr, meta = load_all_forcing_and_states()
    print(f"  Loaded {len(ids)} scenarios")

    print("Loading split IDs...")
    split_ids = load_split_ids()
    train_set = set(split_ids["train"])
    val_set = set(split_ids["val"])
    test_set = set(split_ids["test"])

    # --- 1. Per-family distributions
    print("Computing per-family distributions...")
    family_distributions = global_stats_by_family(forcing, states, fam_arr)

    # --- 2. FFT frequency content
    print("Computing FFT frequency content...")
    freq_analysis = fft_frequency_analysis(forcing)
    # Keep only summary in report; save full spectrum in JSON
    freq_summary = {
        "n_freq_bins": freq_analysis["n_freq_bins"],
        "freq_min": freq_analysis["freq_min"],
        "freq_max": freq_analysis["freq_max"],
        "dominant_freq": freq_analysis["dominant_freq"],
    }

    # --- 3. Cross-split consistency (KS tests)
    print("Running KS tests for cross-split consistency...")
    forcing_train = forcing[[i for i, sid in enumerate(ids) if sid in train_set]]
    forcing_val = forcing[[i for i, sid in enumerate(ids) if sid in val_set]]
    forcing_test = forcing[[i for i, sid in enumerate(ids) if sid in test_set]]

    # Per-state KS too
    ks_results = {}
    for j, name in enumerate(STATE_NAMES):
        s_train = states[[i for i, sid in enumerate(ids) if sid in train_set], :, j].ravel()
        s_val = states[[i for i, sid in enumerate(ids) if sid in val_set], :, j].ravel()
        s_test = states[[i for i, sid in enumerate(ids) if sid in test_set], :, j].ravel()
        ks_results[name] = ks_cross_split(s_train, s_val, s_test)
    ks_results["rho_ext"] = ks_cross_split(
        forcing_train.ravel(), forcing_val.ravel(), forcing_test.ravel()
    )

    # --- 4. Duplicate detection
    print("Detecting near-duplicate forcing profiles...")
    duplicates = cosine_duplicates(forcing, ids, threshold=0.99)

    # --- 5. Outlier flagging (do NOT remove)
    print("Flagging outliers (±5σ)...")
    outlier_flags, train_stats = flag_outliers(
        forcing, states, ids, fam_arr, threshold_sigma=5.0
    )

    # --- Write outputs
    print("Writing outputs...")

    # Outliers CSV (informational only)
    with (R / "outliers_flagged.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "family", "outlier_flags"])
        w.writeheader()
        w.writerows(outlier_flags)

    # Duplicates CSV
    with (R / "duplicates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id_i", "id_j", "cosine"])
        w.writeheader()
        w.writerows(duplicates)

    # Distribution analysis CSV (long format: family × variable × stat)
    dist_rows = []
    for fam, fam_stats in family_distributions.items():
        for var, s in fam_stats.items():
            for k, v in s.items():
                dist_rows.append({
                    "family": fam, "variable": var, "stat": k, "value": v,
                })
    with (R / "distribution_analysis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "variable", "stat", "value"])
        w.writeheader()
        w.writerows(dist_rows)

    # Distribution report
    report = f"""# Step 1.4 Block D — Distribution + Outlier Analysis

## Summary
- Scenarios loaded: {len(ids)}
- Train: {len(split_ids['train'])} / Val: {len(split_ids['val'])} / Test: {len(split_ids['test'])}
- Wall time: {time.perf_counter() - t0:.1f} s

## FFT frequency content (aggregate over 2,000 profiles)
- Frequency bins: {freq_summary['n_freq_bins']}
- Range: [{freq_summary['freq_min']:.4f}, {freq_summary['freq_max']:.4f}] Hz
- Dominant frequency: {freq_summary['dominant_freq']:.4f} Hz

## Cross-split consistency (KS test)
Kolmogorov–Smirnov: 2-sample test. p > 0.05 = no significant difference (good).

| Variable | KS train vs val | p | KS train vs test | p |
|----------|-----------------|---|------------------|---|
"""
    for var, ks in ks_results.items():
        v = ks["train_vs_val"]
        t = ks["train_vs_test"]
        report += (f"| {var} | {v['ks_stat']:.4f} | {v['p_value']:.4f} | "
                   f"{t['ks_stat']:.4f} | {t['p_value']:.4f} |\n")

    report += f"""
## Duplicate detection
- Threshold: cosine similarity > 0.99
- Near-duplicate pairs found: **{len(duplicates)}**
{"- None — all forcing profiles are distinct" if len(duplicates) == 0 else "- See `duplicates.csv` for details"}

## Outliers flagged (±5σ from training mean)
- Scenarios with at least one outlier flag: **{len(outlier_flags)}**
- **Not removed** — informational only
- See `outliers_flagged.csv` for details

## Per-family ranges (a sample)
"""
    for fam in FAMILIES:
        fs = family_distributions[fam]
        report += f"\n### {fam}\n"
        for var in ["rho_ext", "n", "Tf", "Tc"]:
            s = fs[var]
            report += (f"- {var}: mean={s['mean']:.4e}, std={s['std']:.4e}, "
                       f"range=[{s['min']:.4e}, {s['max']:.4e}]\n")

    (R / "distribution_report.md").write_text(report)

    # Summary JSON
    summary = {
        "block": "D (Step 1.4)",
        "wall_time_s": time.perf_counter() - t0,
        "n_scenarios": len(ids),
        "fft_summary": freq_summary,
        "n_duplicate_pairs": len(duplicates),
        "n_outlier_flagged": len(outlier_flags),
        "ks_results": ks_results,
        "outlier_threshold_sigma": 5.0,
    }
    (R / "block_D_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(json.dumps({
        "wall_time_s": summary["wall_time_s"],
        "n_duplicate_pairs": summary["n_duplicate_pairs"],
        "n_outlier_flagged": summary["n_outlier_flagged"],
        "fft_summary": summary["fft_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
