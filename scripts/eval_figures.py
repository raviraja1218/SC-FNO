"""Step 2.4.E — Required manuscript figures."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/phase_2/evaluation"
FIG = PROJECT / "reports/figures/phase_2"
FIG.mkdir(parents=True, exist_ok=True)

STATE_ORDER = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
STATE_UNITS = {"n": "[-]", "C1": "[-]", "C2": "[-]", "C3": "[-]", "C4": "[-]",
               "C5": "[-]", "C6": "[-]", "Tf": "[K]", "Tc": "[K]"}

data = np.load(OUT / "predictions_summary.npz", allow_pickle=True)
sids = data["sids"]
pred_p = data["pred_phys"]
true_p = data["true_phys"]
t = np.linspace(0, 10, 200)
scen_df = pd.read_csv(OUT / "scenario_level_metrics.csv")
per_state = pd.read_csv(OUT / "metrics_per_state.csv")

# Determine one representative scenario per family (first one in each)
test_df = pd.read_csv(PROJECT / "results/phase_1/splitting/test_ids.csv")
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
fam_of = {}
for sid in sids:
    meta = json.load(open(RAW / sid / "metadata.json"))
    fam_of[sid] = meta.get("forcing_family") or meta.get("family") or "unknown"

rep_by_family = {}
for sid, fam in fam_of.items():
    if fam not in rep_by_family:
        rep_by_family[fam] = sid
print(f"Representative scenarios: {rep_by_family}")

def idx_of(sid):
    return list(sids).index(sid)


# --- F1: training vs validation loss (from epoch_metrics.csv) ---
em = pd.read_csv(PROJECT / "results/phase_2/training/epoch_metrics.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(em["epoch"], em["train_mse"], label="train MSE", lw=1.5)
ax.plot(em["epoch"], em["val_mse"], label="val MSE", lw=1.5)
best_epoch = int(em.loc[em["val_mse"].idxmin(), "epoch"])
best_val = float(em["val_mse"].min())
ax.axvline(best_epoch, ls="--", color="green", alpha=0.7, label=f"best epoch {best_epoch} (val {best_val:.4e})")
ax.axvline(em["epoch"].max(), ls=":", color="red", alpha=0.6, label=f"early stop @ {em['epoch'].max()}")
ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Normalized MSE")
ax.set_title("F1 — Training vs Validation Loss"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "f1_training_validation_loss.png", dpi=150); plt.close(fig)
print("F1 saved")

# --- F2: neutron density prediction (one per family) ---
fig, ax = plt.subplots(figsize=(9, 4.8))
colors = plt.cm.tab10.colors
for i, (fam, sid) in enumerate(sorted(rep_by_family.items())):
    k = idx_of(sid)
    ax.plot(t, true_p[k, 0], "-", color=colors[i], lw=1.8, label=f"{fam} truth")
    ax.plot(t, pred_p[k, 0], "--", color=colors[i], lw=1.3, label=f"{fam} pred")
ax.set_xlabel("Time [s]"); ax.set_ylabel("n(t) [-]")
ax.set_title("F2 — Neutron Density: Reference vs Prediction (one scenario per family)")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "f2_neutron_density_prediction.png", dpi=150); plt.close(fig)
print("F2 saved")

# --- F3: precursor predictions (6 subplots, one representative scenario) ---
rep_sid = rep_by_family["step"] if "step" in rep_by_family else list(rep_by_family.values())[0]
k = idx_of(rep_sid)
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for j in range(6):
    ax = axes.flat[j]
    state = STATE_ORDER[1 + j]  # C1..C6
    ax.plot(t, true_p[k, 1+j], "-", lw=1.8, label="truth")
    ax.plot(t, pred_p[k, 1+j], "--", lw=1.3, label="pred")
    ax.set_title(f"{state} — {rep_sid}")
    ax.set_ylabel(f"C_{j+1} [-]"); ax.grid(alpha=0.3)
for ax in axes[1]: ax.set_xlabel("Time [s]")
axes.flat[0].legend()
fig.suptitle(f"F3 — Precursor Concentrations ({rep_sid}, family={fam_of[rep_sid]})", y=1.02)
fig.tight_layout(); fig.savefig(FIG / "f3_precursor_predictions.png", dpi=150); plt.close(fig)
print("F3 saved")

# --- F4: fuel temp (one per family) ---
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (fam, sid) in enumerate(sorted(rep_by_family.items())):
    k = idx_of(sid)
    ax.plot(t, true_p[k, 7], "-", color=colors[i], lw=1.8, label=f"{fam} truth")
    ax.plot(t, pred_p[k, 7], "--", color=colors[i], lw=1.3, label=f"{fam} pred")
ax.set_xlabel("Time [s]"); ax.set_ylabel("T_f [K]")
ax.set_title("F4 — Fuel Temperature: Reference vs Prediction")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "f4_fuel_temperature_prediction.png", dpi=150); plt.close(fig)
print("F4 saved")

# --- F5: coolant temp (one per family) ---
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (fam, sid) in enumerate(sorted(rep_by_family.items())):
    k = idx_of(sid)
    ax.plot(t, true_p[k, 8], "-", color=colors[i], lw=1.8, label=f"{fam} truth")
    ax.plot(t, pred_p[k, 8], "--", color=colors[i], lw=1.3, label=f"{fam} pred")
ax.set_xlabel("Time [s]"); ax.set_ylabel("T_c [K]")
ax.set_title("F5 — Coolant Temperature: Reference vs Prediction")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "f5_coolant_temperature_prediction.png", dpi=150); plt.close(fig)
print("F5 saved")

# --- F6: state-wise relative L2 bar chart ---
fig, ax = plt.subplots(figsize=(9, 4.5))
ps = per_state.set_index("state").loc[STATE_ORDER].reset_index()
xpos = np.arange(9)
ax.bar(xpos - 0.2, ps["rel_l2_mean"], width=0.4, label="mean")
ax.bar(xpos + 0.2, ps["rel_l2_median"], width=0.4, label="median")
ax.set_xticks(xpos); ax.set_xticklabels(STATE_ORDER)
ax.set_ylabel("Relative L2"); ax.set_yscale("log")
ax.set_title("F6 — State-wise Relative L2 (median across 200 test scenarios)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "f6_statewise_error_comparison.png", dpi=150); plt.close(fig)
print("F6 saved")

# --- Worst-case aggregate ---
wc = json.load(open(OUT / "worst_case_aggregate.json"))
sid = wc["scenario_id"]; k = idx_of(sid)
fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
for j, state in enumerate(STATE_ORDER):
    ax = axes.flat[j]
    ax.plot(t, true_p[k, j], "-", lw=1.8, label="truth")
    ax.plot(t, pred_p[k, j], "--", lw=1.3, label="pred")
    ax.set_title(f"{state}"); ax.set_ylabel(STATE_UNITS.get(state, "")); ax.grid(alpha=0.3)
for ax in axes[2]: ax.set_xlabel("Time [s]")
axes.flat[0].legend()
fig.suptitle(f"Worst-case aggregate: {sid} (family={wc['family']}, relL2={wc['aggregate_rel_l2_mean']:.3e})", y=1.01)
fig.tight_layout(); fig.savefig(FIG / "worst_case_prediction.png", dpi=150); plt.close(fig)
print("Worst-case figure saved")

# --- Manuscript Table 4 (per-state metrics) ---
per_state.to_csv(PROJECT / "reports/tables/phase_2/table_4_metrics_per_state.csv", index=False)
per_family = pd.read_csv(OUT / "metrics_per_family.csv")
per_family.to_csv(PROJECT / "reports/tables/phase_2/table_family_performance.csv", index=False)
print("Tables saved")

print("\n=== 2.4.E DONE ===")
