"""Generate F4b — physics loss comparison under CORRECTED per-channel protocol.

Preserves original F4 (per-sum protocol). Adds F4b as additive correction.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results/phase_4/phase_4b/evaluation")
CORR = OUT / "physics_revalidation_addendum"
FIG = Path("reports/figures/phase_4b")
TAB = Path("reports/tables/phase_4b")

models = ["P3-PI-FNO", "Phase4A.1", "P4-A", "P4-B", "P4-C", "P4-D"]
colors = {"P3-PI-FNO": "#A23B72", "Phase4A.1": "#18A558",
          "P4-A": "#E67E22", "P4-B": "#C0392B", "P4-C": "#8E44AD", "P4-D": "#D35400"}

# Load corrected results (per-channel, zero-forcing)
v2 = json.load(open(CORR / "physics_revalidation_v2_zero_forcing.json"))
orig = json.load(open(CORR / "physics_revalidation_results.json"))

# ---- F4b — Corrected physics loss vs original ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: original per-sum
vals_orig = [orig[m]["physics_loss_mean"] if m in orig else np.nan for m in models]
bars = axes[0].bar(models, vals_orig, color=[colors[m] for m in models])
axes[0].axhline(1e-5, color="red", linestyle="--", linewidth=2, label="10^-5 target")
axes[0].set_yscale("log")
axes[0].set_ylabel("Physics loss (per-sum derivative, log scale)")
axes[0].set_title("F4b-left — Original per-sum protocol (Phase 3/4A/4B training)")
axes[0].tick_params(axis='x', rotation=45)
for b, v in zip(bars, vals_orig):
    if not np.isnan(v):
        axes[0].text(b.get_x() + b.get_width()/2, v * 1.5, f"{v:.2e}", ha="center", fontsize=8)
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

# Right panel: corrected per-channel
vals_corr = [v2[m]["physics_loss_mean"] if m in v2 else np.nan for m in models]
bars = axes[1].bar(models, vals_corr, color=[colors[m] for m in models])
axes[1].axhline(1e-5, color="red", linestyle="--", linewidth=2, label="10^-5 target")
axes[1].set_yscale("log")
axes[1].set_ylabel("Physics loss (per-channel derivative, log scale)")
axes[1].set_title("F4b-right — Corrected per-channel protocol")
axes[1].tick_params(axis='x', rotation=45)
for b, v in zip(bars, vals_corr):
    if not np.isnan(v):
        axes[1].text(b.get_x() + b.get_width()/2, v * 1.5, f"{v:.2e}", ha="center", fontsize=8)
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "F4b_physics_loss_corrected.png", dpi=120)
plt.close()
print(f"[OK] F4b written to {FIG}/F4b_physics_loss_corrected.png")

# ---- T1b — Physics loss table (original vs corrected) ----
rows = []
for m in models:
    orig_val = orig[m]["physics_loss_mean"] if m in orig else None
    corr_val = v2[m]["physics_loss_mean"] if m in v2 else None
    ratio = orig_val / corr_val if orig_val and corr_val else None
    rows.append({
        "model": m,
        "physics_loss_original_per_sum": orig_val,
        "physics_loss_corrected_per_channel": corr_val,
        "ratio_original_over_corrected": ratio,
    })
import pandas as pd
pd.DataFrame(rows).to_csv(TAB / "T1b_physics_loss_corrected.csv", index=False)
print(f"[OK] T1b written to {TAB}/T1b_physics_loss_corrected.csv")

# ---- F4c — Per-state corrected physics loss comparison (Phase 4A.1 vs P3) ----
STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
p3_ps = v2["P3-PI-FNO"]["per_state_physics_mean"]
p4a_ps = v2["Phase4A.1"]["per_state_physics_mean"]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(9)
width = 0.35
p3_vals = [p3_ps.get(k, np.nan) for k in STATES]
p4a_vals = [p4a_ps.get(k, np.nan) for k in STATES]
ax.bar(x - width/2, p3_vals, width, label="P3-PI-FNO", color=colors["P3-PI-FNO"])
ax.bar(x + width/2, p4a_vals, width, label="Phase4A.1", color=colors["Phase4A.1"])
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(STATES)
ax.set_ylabel("Per-state physics loss (log scale)")
ax.set_title("F4c — Per-state corrected physics loss: Phase4A.1 vs P3")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# Annotate improvement ratio
for i, (p3v, p4v) in enumerate(zip(p3_vals, p4a_vals)):
    if p3v and p4v and p4v > 0:
        ratio = p3v / p4v
        ax.text(i, max(p3v, p4v) * 1.5, f"{ratio:.1f}x", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "F4c_per_state_corrected.png", dpi=120)
plt.close()
print(f"[OK] F4c written to {FIG}/F4c_per_state_corrected.png")
