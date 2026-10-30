"""Audit item 2 — Temporal boundary vs interior error (read-only)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/phase_2/supplemental_audit")
FIG = Path("reports/figures/phase_2")
FIG.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]

data = np.load("results/phase_2/evaluation/predictions_summary.npz", allow_pickle=True)
pred = data["pred_phys"]   # (200, 9, 200)
true = data["true_phys"]
t = np.linspace(0, 10, 200)

# Segment definitions
# Boundary: t in [0, 0.5] and [9.5, 10] (10 indices each end)
# Interior: t in [0.5, 9.5] (180 indices)
idx_start = np.arange(0, 10)
idx_end = np.arange(190, 200)
idx_interior = np.arange(10, 190)

results = {}
for j, state in enumerate(STATE_ORDER):
    y = true[:, j, :]
    yh = pred[:, j, :]
    err = np.abs(yh - y)     # (200, 200)

    start_mae = float(err[:, idx_start].mean())
    end_mae = float(err[:, idx_end].mean())
    interior_mae = float(err[:, idx_interior].mean())
    overall_mae = float(err.mean())

    results[state] = {
        "start_region_mae": start_mae,
        "end_region_mae": end_mae,
        "interior_mae": interior_mae,
        "overall_mae": overall_mae,
        "start_over_interior_ratio": start_mae / interior_mae if interior_mae > 0 else None,
        "end_over_interior_ratio": end_mae / interior_mae if interior_mae > 0 else None,
    }

# Aggregate stats
ratios_start = [r["start_over_interior_ratio"] for r in results.values() if r["start_over_interior_ratio"] is not None]
ratios_end = [r["end_over_interior_ratio"] for r in results.values() if r["end_over_interior_ratio"] is not None]

boundary_flagged = []
for state, r in results.items():
    sr = r["start_over_interior_ratio"]
    er = r["end_over_interior_ratio"]
    if sr is not None and sr > 2.0:
        boundary_flagged.append(f"{state}_start (ratio={sr:.2f})")
    if er is not None and er > 2.0:
        boundary_flagged.append(f"{state}_end (ratio={er:.2f})")

summary = {
    "results_per_state": results,
    "aggregate": {
        "median_start_over_interior_ratio": float(np.median(ratios_start)),
        "median_end_over_interior_ratio": float(np.median(ratios_end)),
        "max_start_over_interior_ratio": float(np.max(ratios_start)),
        "max_end_over_interior_ratio": float(np.max(ratios_end)),
    },
    "boundary_flags": boundary_flagged,
    "interpretation": (
        "Boundary error within 2× of interior error is not flagged (FFT wraparound "
        "is not materially impacting accuracy). Boundary error >2× interior error "
        "indicates a temporal boundary artifact worth noting for Phase 3."
    ),
}
json.dump(summary, open(OUT / "02_temporal_boundary_analysis.json", "w"), indent=2)

print("=== Per-state boundary vs interior MAE ===")
print(f"{'State':>4} | {'start MAE':>12} | {'end MAE':>12} | {'interior MAE':>14} | {'start/int':>10} | {'end/int':>10}")
print("-" * 90)
for state in STATE_ORDER:
    r = results[state]
    print(f"{state:>4} | {r['start_region_mae']:>12.4e} | {r['end_region_mae']:>12.4e} | "
          f"{r['interior_mae']:>14.4e} | {r['start_over_interior_ratio']:>10.3f} | "
          f"{r['end_over_interior_ratio']:>10.3f}")

print(f"\n=== Aggregate ===")
print(f"Median start/interior ratio: {summary['aggregate']['median_start_over_interior_ratio']:.3f}")
print(f"Median end/interior ratio: {summary['aggregate']['median_end_over_interior_ratio']:.3f}")
print(f"Max start/interior ratio: {summary['aggregate']['max_start_over_interior_ratio']:.3f}")
print(f"Max end/interior ratio: {summary['aggregate']['max_end_over_interior_ratio']:.3f}")
print(f"\nBoundary flags (>2×): {boundary_flagged if boundary_flagged else 'None'}")

# Figure: MAE vs time index, mean over scenarios, per state
fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
for j, state in enumerate(STATE_ORDER):
    ax = axes.flat[j]
    err = np.abs(pred[:, j, :] - true[:, j, :])
    mae_t = err.mean(axis=0)   # mean across 200 scenarios, per timestep
    ax.plot(t, mae_t, color="tab:blue", lw=1.5)
    ax.axvspan(0, 0.5, color="tab:red", alpha=0.15, label="boundary t<0.5s")
    ax.axvspan(9.5, 10, color="tab:red", alpha=0.15)
    ax.axvspan(0.5, 9.5, color="tab:green", alpha=0.05, label="interior")
    ax.set_title(f"{state}: MAE vs time (mean over 200 scenarios)")
    ax.set_yscale("log")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Time [s]")

fig.suptitle("Temporal Boundary Analysis: is there FFT wraparound artifact?", y=1.01)
fig.tight_layout()
fig.savefig(FIG / "boundary_error_profile.png", dpi=150)
plt.close(fig)
print(f"\nFigure saved: {FIG}/boundary_error_profile.png")
