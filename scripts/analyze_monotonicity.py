"""Step 2.5 addendum — Analyze val MSE monotonicity at multiple smoothing windows.
Read-only analysis. Writes nothing except analysis JSON.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/phase_2/closure")
em = pd.read_csv("results/phase_2/training/epoch_metrics.csv")

epochs = em["epoch"].values
val = em["val_mse"].values
train = em["train_mse"].values
best_epoch = int(em.loc[em["val_mse"].idxmin(), "epoch"])
best_val = float(em["val_mse"].min())
n_epochs = len(epochs)

# Raw monotonicity stats
raw_diffs = np.diff(val)
raw_increases = raw_diffs > 0
n_raw_increases = int(raw_increases.sum())
max_raw_increase = float(raw_diffs[raw_increases].max()) if n_raw_increases else 0.0
pct_raw_increases = 100.0 * n_raw_increases / len(raw_diffs)

print(f"=== Raw validation MSE curve ===")
print(f"Epochs: {n_epochs}, best at epoch {best_epoch} = {best_val:.6e}")
print(f"Raw increases (val[i+1] > val[i]): {n_raw_increases} / {len(raw_diffs)} "
      f"({pct_raw_increases:.1f}%)")
print(f"Max single-epoch increase: {max_raw_increase:.6e}")
print(f"First epoch val: {val[0]:.6e}, best epoch val: {best_val:.6e}, "
      f"reduction: {val[0]/best_val:.3f}×")

# Test moving-average windows
def moving_average(x, w):
    """Trailing moving average; first (w-1) entries are NaN (not enough history)."""
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(w - 1, len(x)):
        out[i] = x[i - w + 1:i + 1].mean()
    return out

# Monotonicity of the smoothed curve, over the epochs where it's defined
# AND up to best_epoch (we don't care about monotonicity after overfitting starts)
def check_monotonic(smoothed, upto):
    """Check if smoothed[first_defined : upto+1] is non-increasing."""
    first = np.argmax(~np.isnan(smoothed))
    seg = smoothed[first:upto + 1]
    diffs = np.diff(seg)
    n_viol = int((diffs > 0).sum())
    max_viol = float(diffs[diffs > 0].max()) if n_viol else 0.0
    return {
        "segment_start_epoch": int(epochs[first]),
        "segment_end_epoch": int(epochs[upto]),
        "n_epochs": len(seg),
        "n_increases": n_viol,
        "monotonic": n_viol == 0,
        "max_increase": max_viol,
        "smoothed_start": float(seg[0]),
        "smoothed_end": float(seg[-1]),
        "reduction_x": float(seg[0] / seg[-1]) if seg[-1] > 0 else None,
    }

windows = [5, 10, 15, 20, 25, 30]
results = {"raw": {
    "n_epochs": n_epochs,
    "best_epoch": best_epoch,
    "best_val_mse": best_val,
    "n_raw_increases": n_raw_increases,
    "pct_raw_increases": pct_raw_increases,
    "max_single_epoch_increase": max_raw_increase,
    "reduction_from_first_to_best": float(val[0] / best_val),
}}

for w in windows:
    sm = moving_average(val, w)
    # Monotonicity up to best_epoch (main convergence phase)
    mono = check_monotonic(sm, best_epoch - 1)
    # Store smoothed values in results for reference
    results[f"window_{w}"] = {
        **mono,
        "window": w,
        "smoothed_values_json_size": len(sm.tolist()),
    }
    print(f"\n=== {w}-epoch moving average ===")
    print(f"  Segment: epochs {mono['segment_start_epoch']} → {mono['segment_end_epoch']}")
    print(f"  Increases in smoothed curve: {mono['n_increases']} "
          f"({'MONOTONIC ✓' if mono['monotonic'] else 'NOT monotonic ✗'})")
    print(f"  Smoothed start: {mono['smoothed_start']:.6e}, "
          f"end: {mono['smoothed_end']:.6e}, "
          f"reduction {mono['reduction_x']:.3f}×")
    if not mono["monotonic"]:
        print(f"  Max increase: {mono['max_increase']:.6e}")

# Save the full analysis for the report
OUT.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT / "val_mse_smoothing_analysis.json", "w"), indent=2)
print(f"\n=== Analysis saved to {OUT}/val_mse_smoothing_analysis.json ===")

# Print the recommended window
best_monotonic = None
for w in windows:
    if results[f"window_{w}"]["monotonic"]:
        best_monotonic = w
        break  # smallest monotonic window

if best_monotonic:
    print(f"\n✓ RECOMMENDED: {best_monotonic}-epoch moving average is monotonic "
          f"from epoch {results[f'window_{best_monotonic}']['segment_start_epoch']} "
          f"to epoch {best_epoch}.")
else:
    print(f"\n⚠ No window is strictly monotonic. Using window 10 with "
          f"violation count reported.")
