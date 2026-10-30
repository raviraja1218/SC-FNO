"""Generate f1b_val_mse_moving_average.png + monotonicity metrics JSON."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/phase_2/closure")
FIG = Path("reports/figures/phase_2")
FIG.mkdir(parents=True, exist_ok=True)

em = pd.read_csv("results/phase_2/training/epoch_metrics.csv")
epochs = em["epoch"].values
val = em["val_mse"].values
train = em["train_mse"].values
best_epoch = int(em.loc[em["val_mse"].idxmin(), "epoch"])
best_val = float(em["val_mse"].min())


def moving_average(x, w):
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(w - 1, len(x)):
        out[i] = x[i - w + 1:i + 1].mean()
    return out


ma10 = moving_average(val, 10)
best_so_far = np.minimum.accumulate(val)
new_minima_count = int((val == best_so_far).sum())

raw_diffs = np.diff(val)
n_raw_increases = int((raw_diffs > 0).sum())
max_raw_increase = float(raw_diffs[raw_diffs > 0].max()) if n_raw_increases else 0.0
pct_raw_increases = 100.0 * n_raw_increases / len(raw_diffs)

seg = ma10[9:best_epoch]
seg_diffs = np.diff(seg)
n_sm_increases = int((seg_diffs > 0).sum())
max_sm_increase = float(seg_diffs[seg_diffs > 0].max()) if n_sm_increases else 0.0

# Build sub-dicts FIRST so we can reference them in interpretation string
reduction_first_to_best = float(val[0] / best_val)
ma_sub = {
    "segment": f"epochs 10 to {best_epoch}",
    "start_value": float(ma10[9]),
    "end_value": float(ma10[best_epoch - 1]),
    "reduction_x": round(float(ma10[9] / ma10[best_epoch - 1]), 3),
    "n_smoothed_increases": n_sm_increases,
    "max_smoothed_increase": max_sm_increase,
    "max_smoothed_increase_pct_of_best": round(100.0 * max_sm_increase / best_val, 3),
}

interpretation = (
    f"Raw per-epoch val MSE fluctuates in {n_raw_increases}/{len(raw_diffs)} "
    f"({pct_raw_increases:.1f}%) of consecutive-epoch comparisons, with maximum "
    f"single-epoch increase {max_raw_increase:.3e} ({100.0*max_raw_increase/best_val:.2f}% "
    f"of best val MSE) — consistent with mini-batch SGD stochasticity. "
    f"The 10-epoch moving average decays with reduction {ma_sub['reduction_x']:.3f}×; "
    f"the best-so-far curve is strictly monotonic by construction with "
    f"{new_minima_count} new minima recorded. Convergence from "
    f"{val[0]:.3e} to {best_val:.3e} is a {reduction_first_to_best:.2f}× reduction."
)

metrics = {
    "n_epochs": int(len(epochs)),
    "best_epoch": best_epoch,
    "best_val_mse": best_val,
    "first_val_mse": float(val[0]),
    "reduction_first_to_best_x": reduction_first_to_best,
    "raw": {
        "n_increases": n_raw_increases,
        "pct_increases": round(pct_raw_increases, 1),
        "max_single_increase": max_raw_increase,
        "max_single_increase_pct_of_best": round(100.0 * max_raw_increase / best_val, 2),
    },
    "moving_average_10ep": ma_sub,
    "best_so_far_curve": {
        "strictly_monotonic_by_construction": True,
        "n_new_minima": new_minima_count,
        "first_value": float(best_so_far[0]),
        "last_value": float(best_so_far[-1]),
        "reduction_x": round(float(best_so_far[0] / best_so_far[-1]), 3),
    },
    "stability": {
        "n_nan": int(np.isnan(val).sum() + np.isnan(train).sum()),
        "n_inf": int(np.isinf(val).sum() + np.isinf(train).sum()),
        "max_train_mse_increase": float(np.diff(train)[np.diff(train) > 0].max())
        if (np.diff(train) > 0).any() else 0.0,
    },
    "interpretation": interpretation,
}

json.dump(metrics, open(OUT / "val_mse_smoothing_analysis.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))

# ----- Figure -----
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(epochs, val, color="lightgray", lw=1.0, label="raw per-epoch val MSE")
ax.plot(epochs, train, color="peachpuff", lw=1.0, alpha=0.7, label="raw per-epoch train MSE")
ax.plot(epochs, ma10, color="tab:blue", lw=2.0, label="10-epoch moving average (val)")
ax.plot(epochs, best_so_far, color="tab:green", lw=2.2,
        label=f"best-so-far (val), monotonic by construction ({new_minima_count} new minima)")
ax.axvline(best_epoch, ls="--", color="tab:green", alpha=0.6,
           label=f"best epoch {best_epoch} = {best_val:.4e}")
ax.axvline(epochs[-1], ls=":", color="tab:red", alpha=0.6,
           label=f"early stop at epoch {epochs[-1]}")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Normalized MSE (log scale)")
ax.set_title(
    "F1b — Validation MSE: raw, 10-epoch moving average, and monotonic best-so-far\n"
    f"Raw reductions: {pct_raw_increases:.1f}% of epochs go up "
    f"(max +{max_raw_increase:.2e}); best-so-far is strictly non-increasing"
)
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "f1b_val_mse_moving_average.png", dpi=150)
plt.close(fig)
print(f"\nFigure saved: {FIG / 'f1b_val_mse_moving_average.png'}")
