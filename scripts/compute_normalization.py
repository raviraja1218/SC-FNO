"""Step 1.4 Block C — training-only normalization statistics.

Computes global + per-time-point stats for:
  - rho_ext (forcing, 200 values per scenario)
  - n, C1..C6, Tf, Tc (9 states, 200 values per scenario)

Training set only (1600 scenarios from train_ids.csv).
Validation and test are NOT loaded.
"""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')

import csv
import json
import time
from pathlib import Path

import numpy as np

R = Path('results/phase_1/splitting')
RAW = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"
N_TIME = 200

VAR_NAMES = ["rho_ext", "n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]


def load_training_arrays():
    """Load all 1600 training scenarios into stacked arrays."""
    with (R / "train_ids.csv").open() as f:
        train_ids = [r["scenario_id"] for r in csv.DictReader(f)]

    n_scen = len(train_ids)
    # (n_scen, N_TIME) for forcing; (n_scen, N_TIME, 9) for states
    forcing_arr = np.empty((n_scen, N_TIME), dtype=np.float64)
    states_arr = np.empty((n_scen, N_TIME, 9), dtype=np.float64)

    for i, sid in enumerate(train_ids):
        d = RAW / sid
        forcing_arr[i] = np.load(d / "forcing.npy")
        states_arr[i] = np.load(d / "trajectory.npy")

    return train_ids, forcing_arr, states_arr


def stats_1d(arr, name):
    """Compute global stats for a 1-D array of values."""
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    var = std ** 2
    return {
        "name": name,
        "n_samples": int(arr.size),
        "mean": mean,
        "std": std,
        "variance": var,
        "min": mn,
        "max": mx,
        "p5": p5,
        "p95": p95,
        "near_zero_variance": bool(std < 1e-6),
    }


def main():
    t0 = time.perf_counter()
    train_ids, forcing, states = load_training_arrays()
    n_scen = len(train_ids)
    assert n_scen == 1600, f"expected 1600 train scenarios, got {n_scen}"
    print(f"Loaded {n_scen} training scenarios")
    print(f"  forcing shape: {forcing.shape}")
    print(f"  states shape:  {states.shape}")

    # Global stats
    global_stats = {}
    for name, arr in [("rho_ext", forcing.ravel())] + [
        (VAR_NAMES[1 + i], states[:, :, i].ravel()) for i in range(9)
    ]:
        global_stats[name] = stats_1d(arr, name)
        print(f"  {name}: mean={global_stats[name]['mean']:.6e}, "
              f"std={global_stats[name]['std']:.6e}")

    # Per-time-point stats
    per_time_point = {}
    for name, arr2d in [("rho_ext", forcing)] + [
        (VAR_NAMES[1 + i], states[:, :, i]) for i in range(9)
    ]:
        # arr2d shape (n_scen, N_TIME) -> per-time stats over scenarios
        means = np.mean(arr2d, axis=0)  # (N_TIME,)
        stds = np.std(arr2d, axis=0)
        mins = np.min(arr2d, axis=0)
        maxs = np.max(arr2d, axis=0)
        per_time_point[name] = {
            "n_time": int(N_TIME),
            "mean": means.tolist(),
            "std": stds.tolist(),
            "min": mins.tolist(),
            "max": maxs.tolist(),
        }

    # Normalization method + near-zero flags
    near_zero_vars = [k for k, v in global_stats.items() if v["near_zero_variance"]]

    stats = {
        "source": "training only (1600 scenarios from train_ids.csv)",
        "split_seed": "split_v1_2026",
        "n_training_scenarios": n_scen,
        "n_time_points": N_TIME,
        "variables": VAR_NAMES,
        "method": {
            "global": "z-score: x_norm = (x - mean) / std",
            "per_time_point": "optional: x_norm(t) = (x(t) - mean(t)) / std(t)",
            "fallback": "min-max available from min/max fields",
        },
        "near_zero_variance_variables": near_zero_vars,
        "global": global_stats,
        "per_time_point": per_time_point,
    }
    (R / "normalization_statistics.json").write_text(json.dumps(stats, indent=2))

    # Report
    report = f"""# Step 1.4 Block C — Normalization Statistics Report

## Source
- Training set only: **{n_scen}** scenarios (from `train_ids.csv`)
- Split seed: `split_v1_2026`
- Validation and test data: **NOT loaded** (guaranteed no leakage)

## Method
- Primary: **z-score** — `x_norm = (x − mean) / std`
- Optional per-time-point: `x_norm(t) = (x(t) − mean(t)) / std(t)`
- Fallback: min-max available from `min/max` fields
- Raw data is **not modified** — normalization is applied at training time by Phase 2 dataloaders

## Variables
{', '.join(VAR_NAMES)}

## Global statistics (training set, all time points)

| Variable | mean | std | min | max | p5 | p95 | near-zero std? |
|----------|------|-----|-----|-----|-----|-----|----------------|
"""
    for name in VAR_NAMES:
        s = global_stats[name]
        nz = "⚠️ YES" if s["near_zero_variance"] else "no"
        report += (f"| {name} | {s['mean']:.6e} | {s['std']:.6e} | "
                   f"{s['min']:.6e} | {s['max']:.6e} | {s['p5']:.6e} | "
                   f"{s['p95']:.6e} | {nz} |\n")

    report += f"""
## Per-time-point statistics
- Computed at each of {N_TIME} time points across the training set
- Stored in `normalization_statistics.json` under `per_time_point`
- Useful for time-conditioned normalization if Phase 2 requires it

## Near-zero variance flags
"""
    if near_zero_vars:
        report += f"⚠️ Variables with std < 1e-6: {near_zero_vars}\n"
        report += "These require special handling in Phase 2 normalization.\n"
    else:
        report += "None — all variables have std > 1e-6.\n"

    report += f"""
## Validation
- Loaded scenarios: {n_scen} (exactly the training split)
- Variables with stats: {len(global_stats)} (all required)
- Per-time-point arrays: {len(per_time_point)} × {N_TIME} values each
- JSON file written: `normalization_statistics.json`
"""
    (R / "normalization_report.md").write_text(report)

    wall = time.perf_counter() - t0
    summary = {
        "block": "C (Step 1.4)",
        "n_training_scenarios": n_scen,
        "n_variables": len(global_stats),
        "n_time_points": N_TIME,
        "near_zero_variance_variables": near_zero_vars,
        "wall_time_s": wall,
        "source": "training only",
    }
    (R / "block_C_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
