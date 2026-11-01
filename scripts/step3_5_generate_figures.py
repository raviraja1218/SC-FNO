"""Missing Item 2 — Generate required Phase 3 figures."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from step3_5_eval_utils import load_test_data, STATES

base = Path("results/phase_3/evaluation")
fig_dir = Path("results/phase_3/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIGURE 1: Dual-channel loss convergence
# ============================================================
print("Generating Figure 1: dual-channel loss convergence...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, variant, title in zip(
    axes,
    ["p_fno", "pi_fno", "p_pi_fno"],
    ["P-FNO (padding only)", "PI-FNO (physics only)", "P-PI-FNO (physics+padding)"],
):
    log_path = f"results/phase_3/training/{variant}/training_log.csv"
    if not Path(log_path).exists():
        continue
    df = pd.read_csv(log_path)
    epochs = df["epoch"]
    ax.plot(epochs, df["train_data_loss"], label="L_data", color="C0", linewidth=1.5)
    if "train_phys_loss" in df.columns:
        phys = df["train_phys_loss"]
        if phys.max() > 1e-6:
            ax.plot(epochs, phys, label="L_physics", color="C1", linewidth=1.5)
            ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(fig_dir / "f1_dual_channel_loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved f1_dual_channel_loss.png")

# ============================================================
# FIGURE 2: Boundary artifact comparison (P2 vs PI-FNO for neutron density n)
# ============================================================
print("Generating Figure 2: boundary artifact comparison...")
boundary = json.load(open(base / "boundary_comparison.json"))
p2_preds = json.load(open(base / "p2_fno_predictions.json"))["predictions"]
pi_preds = json.load(open(base / "pi_fno_predictions.json"))["predictions"]
test_data = load_test_data()

# For each model, compute per-timestep MAE for neutron density
def per_timestep_mae(preds_dict, state_idx=0):
    maes = []
    for sid, pred in preds_dict.items():
        pred_arr = np.array(pred)
        gt = test_data[sid]["trajectory"].T
        maes.append(np.abs(pred_arr[state_idx] - gt[state_idx]))
    return np.mean(np.array(maes), axis=0)  # [200] mean MAE per timestep

p2_n_mae = per_timestep_mae(p2_preds, state_idx=0)
pi_n_mae = per_timestep_mae(pi_preds, state_idx=0)
time_grid = np.linspace(0, 10, 200)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(time_grid, p2_n_mae, label="P2-FNO", color="C0", linewidth=1.5)
ax.plot(time_grid, pi_n_mae, label="PI-FNO", color="C1", linewidth=1.5)
ax.axvline(x=time_grid[10], color="red", linestyle="--", alpha=0.5,
           label=f"Start window cutoff (t={time_grid[10]:.2f}s)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean absolute error (neutron density n)")
ax.set_title("Neutron density boundary artifact: P2-FNO vs PI-FNO")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(fig_dir / "f2_boundary_comparison_neutron.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved f2_boundary_comparison_neutron.png")

# ============================================================
# FIGURE 3: Worst-case scenario_0458 all 4 models
# ============================================================
print("Generating Figure 3: scenario_0458 comparison...")
sid = "scenario_0458"
gt_458 = test_data[sid]["trajectory"].T  # [9, 200]
time_458 = test_data[sid]["time"]  # [200]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for j, state in enumerate(STATES):
    ax = axes[j]
    ax.plot(time_458, gt_458[j], "k-", label="Ground truth", linewidth=2)
    for v, name, color in [
        ("p2_fno", "P2-FNO", "C0"),
        ("p_fno", "P-FNO", "C1"),
        ("pi_fno", "PI-FNO", "C2"),
        ("p_pi_fno", "P-PI-FNO", "C3"),
    ]:
        preds = json.load(open(base / f"{v}_predictions.json"))["predictions"]
        pred = np.array(preds[sid])
        ax.plot(time_458, pred[j], "--", label=name, color=color, linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(state)
    ax.set_title(f"Scenario 0458 — {state}")
    if j == 0:
        ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Scenario 0458 — all 4 models vs ground truth\n(Honest finding: nearly indistinguishable — physics did NOT correct worst-case)",
             y=1.001, fontsize=12)
plt.tight_layout()
fig.savefig(fig_dir / "f3_scenario_0458_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved f3_scenario_0458_comparison.png")

# ============================================================
# FIGURE 4: Per-family comparison bar chart (4 models)
# ============================================================
print("Generating Figure 4: per-family comparison...")
per_family = json.load(open(base / "per_family_full.json"))["per_family_full"]

fig, ax = plt.subplots(figsize=(10, 5))
families = ["step", "ramp", "sinusoidal", "grf"]
models = ["P2-FNO", "P-FNO", "PI-FNO", "P-PI-FNO"]
colors = ["C0", "C1", "C2", "C3"]
x = np.arange(len(families))
width = 0.2
for i, (model, color) in enumerate(zip(models, colors)):
    vals = [per_family[fam][model] for fam in families]
    ax.bar(x + i*width - 1.5*width, vals, width, label=model, color=color)
ax.set_xticks(x)
ax.set_xticklabels(families)
ax.set_ylabel("Aggregate Rel L2 median")
ax.set_title("Per-family Rel L2 comparison — P2-FNO wins all families")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(fig_dir / "f4_per_family_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved f4_per_family_comparison.png")

print("\n=== All figures generated ===")
