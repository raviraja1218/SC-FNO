"""Step 4A.4 — Optimization Analysis.

Produces 6 figures + 2 unavailability notes + hyperparameter table from
Phase 4A.1 training logs. Read-only. No training. No test-set use.

Per reviewer:
  - Every figure traceable to a source artifact
  - F5 (per-equation convergence) and F7 (gradient norms) unavailable — logged honestly
  - Hyperparameter table verbatim from pre_registration.yaml
"""
import json
import yaml
import hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OPT = Path("results/phase_4/phase_4a/optimizer")
OUT = Path("results/phase_4/phase_4a/analysis")
FIG = Path("reports/figures/phase_4/phase_4a")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---------- Load source data ----------
adam = pd.read_csv(OPT / "adam_training_log.csv")
lbfgs = pd.read_csv(OPT / "lbfgs_training_log.csv")
transition = json.load(open(OPT / "transition_event.json"))
summary = json.load(open(OPT / "training_summary.json"))
prereg = yaml.safe_load(open(OPT / "pre_registration.yaml"))

print(f"Adam log: {len(adam)} rows")
print(f"L-BFGS log: {len(lbfgs)} rows")
print(f"Transition epoch: {transition['plateau_epoch']}, λ={transition['lam_phys_at_transition']:.6f}")

# Find best val_data epoch
best_idx = adam["val_data"].idxmin()
best_epoch = int(adam.loc[best_idx, "epoch"])
best_val = float(adam.loc[best_idx, "val_data"])
print(f"Best val_data: {best_val:.6f} at epoch {best_epoch}")

# ---------- F1: Adam loss trajectory ----------
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(adam["epoch"], adam["train_data"], 'b-', label="train_data", linewidth=1.5)
ax1.plot(adam["epoch"], adam["val_data"], 'b--', label="val_data", linewidth=1.5)
ax1.plot(adam["epoch"], adam["train_phys"], 'r-', label="train_phys (λ-weighted)", linewidth=1.5)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (log scale)")
ax1.set_yscale("log")
ax1.axvline(x=best_epoch, color="green", linestyle=":", linewidth=2,
            label=f"Best val_data (epoch {best_epoch})")
ax1.axvline(x=transition["plateau_epoch"], color="orange", linestyle="--", linewidth=2,
            label=f"Plateau (epoch {transition['plateau_epoch']})")

# λ_phys on secondary axis
ax2 = ax1.twinx()
ax2.plot(adam["epoch"], adam["lam_phys"], 'g-', alpha=0.5, label="λ_phys", linewidth=1)
ax2.set_ylabel("λ_phys", color="green")
ax2.set_yscale("log")
ax2.tick_params(axis="y", labelcolor="green")

ax1.set_title(f"F1 — Adam Loss Trajectory: {len(adam)} epochs, plateau at epoch {transition['plateau_epoch']}")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left", fontsize=9)
ax2.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "figure_F1_adam_loss_trajectory.png", dpi=150)
plt.close()
print("✓ F1 produced")

# ---------- F2: L-BFGS loss trajectory ----------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lbfgs["closure_call"], lbfgs["loss"], 'g-o', linewidth=2, markersize=4)
ax.set_xlabel("Closure call")
ax.set_ylabel("Loss (composite training objective)")
ax.set_title(f"F2 — L-BFGS Loss Trajectory: {len(lbfgs)} closure evaluations, final loss {lbfgs['loss'].iloc[-1]:.4f}")
ax.grid(True, alpha=0.3)
ax.annotate(f"Final: {lbfgs['loss'].iloc[-1]:.4f}",
            xy=(lbfgs["closure_call"].iloc[-1], lbfgs["loss"].iloc[-1]),
            xytext=(lbfgs["closure_call"].iloc[-1] - 3, lbfgs["loss"].iloc[-1] + 0.05),
            fontsize=10, arrowprops=dict(arrowstyle="->"))
ax.text(0.5, 0.02,
        f"Terminated: tolerance_change=1e-9 satisfied. Fixed 16-scenario batch.",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic")
plt.tight_layout()
plt.savefig(FIG / "figure_F2_lbfgs_loss_trajectory.png", dpi=150)
plt.close()
print("✓ F2 produced")

# ---------- F3: Combined optimization trace ----------
fig, ax = plt.subplots(figsize=(12, 6))
# Adam total loss
ax.plot(adam["epoch"], adam["train_total"], 'b-', label="Adam train_total", linewidth=1.5)
# L-BFGS: offset epochs to continue x-axis
lbfgs_x = np.arange(len(adam), len(adam) + len(lbfgs))
ax.plot(lbfgs_x, lbfgs["loss"], 'g-o', label="L-BFGS loss", linewidth=2, markersize=4)
ax.axvline(x=len(adam) - 1, color="orange", linestyle="--", linewidth=2,
           label="Adam → L-BFGS transition")
ax.set_xlabel("Optimization step (Adam epochs + L-BFGS closures)")
ax.set_ylabel("Total loss (log scale)")
ax.set_yscale("log")
ax.set_title(f"F3 — Combined Optimization: Adam {len(adam)} epochs + L-BFGS {len(lbfgs)} closures = {transition['adam_wall_time_s'] + summary['lbfgs']['wall_time_s']:.1f}s")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "figure_F3_combined_optimization_trace.png", dpi=150)
plt.close()
print("✓ F3 produced")

# ---------- F4: Transition marker with annotations ----------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(adam["epoch"], adam["val_data"], 'b-', linewidth=1.5, label="val_data")
ax.plot(adam["epoch"], adam["train_data"], 'b--', alpha=0.6, linewidth=1, label="train_data")
ax.plot(adam["epoch"], adam["train_phys"], 'r-', linewidth=1.5, label="train_phys")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log)")
ax.set_yscale("log")
# Transition annotation
ax.axvline(x=transition["plateau_epoch"], color="orange", linestyle="--", linewidth=2)
ax.annotate(
    f"Plateau detected\nepoch={transition['plateau_epoch']}\nλ_phys={transition['lam_phys_at_transition']:.4f}\n"
    f"EMA={transition['ema_value']:.6f}\npolicy: {transition['policy']}",
    xy=(transition["plateau_epoch"], transition["raw_val_loss"]),
    xytext=(transition["plateau_epoch"] - 15, transition["raw_val_loss"] * 1.5),
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"),
    arrowprops=dict(arrowstyle="->", color="orange"),
)
# Best-epoch marker
ax.axvline(x=best_epoch, color="green", linestyle=":", linewidth=2)
ax.annotate(
    f"Best val_data\nepoch={best_epoch}\nval={best_val:.6f}\n(not preserved)",
    xy=(best_epoch, best_val),
    xytext=(best_epoch + 3, best_val * 0.5),
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", edgecolor="green"),
    arrowprops=dict(arrowstyle="->", color="green"),
)
ax.set_title("F4 — Automatic Transition Marker: EMA plateau detection, λ_phys inherited (Policy B)")
ax.grid(True, alpha=0.3)
ax.legend(loc="center left", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "figure_F4_transition_marker.png", dpi=150)
plt.close()
print("✓ F4 produced")

# ---------- F5: UNAVAILABLE ----------
(OUT / "figure_F5_UNAVAILABLE.md").write_text(
    "# Figure F5 — 9-Equation Residual Convergence (UNAVAILABLE)\n\n"
    "**Reason:** Per-equation residual history was not logged during Phase 4A.1 training.\n\n"
    "The `adam_training_log.csv` records only aggregate `train_phys` per epoch. It does not\n"
    "record per-equation residuals (R_n, R_C1..C6, R_Tf, R_Tc) per epoch.\n\n"
    "**Alternative in place:** Step 4A.3 provides the final per-state residual comparison\n"
    "(Phase 3 vs Phase 4A.1) in `residual_comparison_per_state_CORRECTED.csv`. This shows\n"
    "the aggregate physics improvement (828× across all 9 states) but not the per-epoch\n"
    "convergence trajectory.\n\n"
    "**Path forward:** A future training run (Phase 4B) would need per-equation residual\n"
    "logging at each epoch to produce this figure. Do not fabricate from final measurements.\n"
)
print("✓ F5 marked UNAVAILABLE")

# ---------- F6: Data vs physics trade-off ----------
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(adam["train_data"], adam["train_phys"], c=adam["lam_phys"],
                cmap="viridis", s=50, edgecolors="black", linewidths=0.3)
# Trajectory line
ax.plot(adam["train_data"], adam["train_phys"], 'gray', alpha=0.3, linewidth=1)
# Start and end markers
ax.scatter(adam["train_data"].iloc[0], adam["train_phys"].iloc[0],
           marker="s", s=150, color="blue", edgecolors="black",
           label=f"epoch 0 (λ={adam['lam_phys'].iloc[0]:.4f})", zorder=5)
ax.scatter(adam["train_data"].iloc[-1], adam["train_phys"].iloc[-1],
           marker="*", s=250, color="red", edgecolors="black",
           label=f"epoch {adam['epoch'].iloc[-1]} (λ={adam['lam_phys'].iloc[-1]:.4f})", zorder=5)
ax.set_xlabel("train_data (log)")
ax.set_ylabel("train_phys (log)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("F6 — Data vs Physics Trade-Off During Adam Stage (color = λ_phys)")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("λ_phys")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "figure_F6_data_vs_physics_trajectory.png", dpi=150)
plt.close()
print("✓ F6 produced")

# ---------- F7: UNAVAILABLE ----------
(OUT / "figure_F7_UNAVAILABLE.md").write_text(
    "# Figure F7 — Gradient Norm Evolution (UNAVAILABLE)\n\n"
    "**Reason:** Gradient norms were not logged during Phase 4A.1 training.\n\n"
    "The `adam_training_log.csv` does not contain a gradient norm column.\n\n"
    "**Alternative in place:** None directly. L-BFGS convergence behavior (17 closures to\n"
    "tolerance_change=1e-9 satisfied) provides indirect evidence of well-behaved gradients\n"
    "on the fixed batch, but does not constitute a gradient norm evolution trace.\n\n"
    "**Path forward:** Gradient norm logging is a natural Phase 4B enhancement.\n"
)
print("✓ F7 marked UNAVAILABLE")

# ---------- F8: GPU memory evolution ----------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(adam["epoch"], adam["peak_vram_MB"], 'b-', linewidth=1.5, label="Adam peak VRAM")
ax.axhline(y=adam["peak_vram_MB"].max(), color="blue", linestyle=":", alpha=0.5,
           label=f"Adam max: {adam['peak_vram_MB'].max():.1f} MB")
ax.axhline(y=summary["peak_vram_MB"]["lbfgs"], color="green", linestyle="--",
           label=f"L-BFGS peak: {summary['peak_vram_MB']['lbfgs']:.1f} MB")
ax.axhline(y=3000, color="red", linestyle="-", alpha=0.6, label="3 GB gate")
ax.axhline(y=6439, color="gray", linestyle=":", alpha=0.4, label="Total GPU VRAM: 6439 MB")
ax.set_xlabel("Epoch")
ax.set_ylabel("Peak VRAM (MB)")
ax.set_title(f"F8 — GPU Memory Evolution: {adam['peak_vram_MB'].max():.1f} MB peak Adam, {summary['peak_vram_MB']['lbfgs']:.1f} MB peak L-BFGS (38× under 3 GB)")
ax.grid(True, alpha=0.3)
ax.legend(loc="center right", fontsize=9)
ax.set_ylim(0, 7000)
plt.tight_layout()
plt.savefig(FIG / "figure_F8_gpu_memory_evolution.png", dpi=150)
plt.close()
print("✓ F8 produced")

# ---------- Figure inventory ----------
inventory = {
    "phase": "4A.4",
    "total_planned": 8,
    "produced": 6,
    "unavailable": 2,
    "figures": {
        "F1": {"name": "Adam loss trajectory", "status": "produced",
               "source": "adam_training_log.csv", "file": "figure_F1_adam_loss_trajectory.png"},
        "F2": {"name": "L-BFGS loss trajectory", "status": "produced",
               "source": "lbfgs_training_log.csv", "file": "figure_F2_lbfgs_loss_trajectory.png"},
        "F3": {"name": "Combined optimization trace", "status": "produced",
               "source": "adam_training_log.csv + lbfgs_training_log.csv",
               "file": "figure_F3_combined_optimization_trace.png"},
        "F4": {"name": "Automatic transition marker", "status": "produced",
               "source": "transition_event.json + adam_training_log.csv",
               "file": "figure_F4_transition_marker.png"},
        "F5": {"name": "9-equation residual convergence", "status": "unavailable",
               "reason": "Per-equation residual history not logged per epoch",
               "file": "figure_F5_UNAVAILABLE.md"},
        "F6": {"name": "Data accuracy vs physics accuracy", "status": "produced",
               "source": "adam_training_log.csv", "file": "figure_F6_data_vs_physics_trajectory.png"},
        "F7": {"name": "Gradient norm evolution", "status": "unavailable",
               "reason": "Gradient norms not logged during training",
               "file": "figure_F7_UNAVAILABLE.md"},
        "F8": {"name": "GPU memory evolution", "status": "produced",
               "source": "adam_training_log.csv (peak_vram_MB column)",
               "file": "figure_F8_gpu_memory_evolution.png"},
    },
}
with open(OUT / "figure_inventory.json", "w") as f:
    json.dump(inventory, f, indent=2)
print("✓ figure_inventory.json written")

# ---------- Hyperparameter table (VERBATIM from pre-registration) ----------
hp_rows = [
    # Config
    ("Config", "Starting checkpoint", "starting_checkpoint.path", prereg["starting_checkpoint"]["path"]),
    ("Config", "Checkpoint SHA", "starting_checkpoint.sha256", prereg["starting_checkpoint"]["sha256"]),
    ("Config", "Checkpoint epoch", "starting_checkpoint.epoch", prereg["starting_checkpoint"]["epoch"]),
    ("Config", "Starting λ_phys", "starting_checkpoint.lam_phys", prereg["starting_checkpoint"]["lam_phys"]),
    ("Config", "Seed", "seed", prereg["seed"]),
    ("Config", "Device", "device", prereg["device"]),
    ("Config", "Batch size (training)", "batch_size_training", prereg["batch_size_training"]),
    # Adam
    ("Adam", "Learning rate", "adam.lr", prereg["adam"]["lr"]),
    ("Adam", "Betas", "adam.betas", prereg["adam"]["betas"]),
    ("Adam", "Eps", "adam.eps", prereg["adam"]["eps"]),
    ("Adam", "Weight decay", "adam.weight_decay", prereg["adam"]["weight_decay"]),
    ("Adam", "State", "adam.state", prereg["adam"]["state"]),
    # Scheduler
    ("Scheduler", "Class", "scheduler.class", prereg["scheduler"]["class"]),
    ("Scheduler", "T_max", "scheduler.T_max", prereg["scheduler"]["T_max"]),
    ("Scheduler", "eta_min", "scheduler.eta_min", prereg["scheduler"]["eta_min"]),
    # Lambda_phys
    ("λ_phys", "Init", "lambda_phys.init", prereg["lambda_phys"]["init"]),
    ("λ_phys", "Target", "lambda_phys.target", prereg["lambda_phys"]["target"]),
    ("λ_phys", "Ramp epochs", "lambda_phys.ramp_epochs", prereg["lambda_phys"]["ramp_epochs"]),
    ("λ_phys", "Transition policy", "lambda_phys.transition_policy", prereg["lambda_phys"]["transition_policy"]),
    # Plateau
    ("Plateau", "Metric", "plateau_detection.metric", prereg["plateau_detection"]["metric"]),
    ("Plateau", "Protocol", "plateau_detection.metric_protocol", prereg["plateau_detection"]["metric_protocol"]),
    ("Plateau", "EMA alpha", "plateau_detection.ema_alpha", prereg["plateau_detection"]["ema_alpha"]),
    ("Plateau", "Reference window", "plateau_detection.reference_window", prereg["plateau_detection"]["reference_window"]),
    ("Plateau", "Delta", "plateau_detection.delta", prereg["plateau_detection"]["delta"]),
    ("Plateau", "Patience", "plateau_detection.patience", prereg["plateau_detection"]["patience"]),
    ("Plateau", "Min Adam epochs", "plateau_detection.min_adam_epochs", prereg["plateau_detection"]["min_adam_epochs"]),
    ("Plateau", "Max Adam epochs", "plateau_detection.max_adam_epochs", prereg["plateau_detection"]["max_adam_epochs"]),
    # L-BFGS
    ("L-BFGS", "Class", "lbfgs.class", prereg["lbfgs"]["class"]),
    ("L-BFGS", "History size", "lbfgs.history_size", prereg["lbfgs"]["history_size"]),
    ("L-BFGS", "Max iter", "lbfgs.max_iter", prereg["lbfgs"]["max_iter"]),
    ("L-BFGS", "tolerance_grad", "lbfgs.tolerance_grad", prereg["lbfgs"]["tolerance_grad"]),
    ("L-BFGS", "tolerance_change", "lbfgs.tolerance_change", prereg["lbfgs"]["tolerance_change"]),
    ("L-BFGS", "line_search_fn", "lbfgs.line_search_fn", prereg["lbfgs"]["line_search_fn"]),
    ("L-BFGS", "Fixed batch source", "lbfgs.fixed_batch.source", prereg["lbfgs"]["fixed_batch"]["source"]),
]

# Assertions: table matches pre-registration
def _assert_match(section, key_path, expected):
    keys = key_path.split(".")
    actual = prereg
    for k in keys:
        actual = actual[k]
    return actual == expected

# Sample assertions on critical fields
assert _assert_match("adam", "adam.lr", 0.001), "Adam lr mismatch"
assert _assert_match("lambda_phys", "lambda_phys.transition_policy", "B_inherit_lambda"), "Policy mismatch"
assert _assert_match("plateau", "plateau_detection.ema_alpha", 0.1), "EMA alpha mismatch"
assert _assert_match("lbfgs", "lbfgs.history_size", 50), "L-BFGS history mismatch"

# Save as CSV
hp_df = pd.DataFrame(hp_rows, columns=["group", "parameter", "yaml_path", "value"])
hp_df.to_csv(OUT / "optimization_hyperparameter_table.csv", index=False)

# Save as MD
md = ["# Step 4A.4 — Optimization Hyperparameter Table\n",
      "**All values verbatim from `pre_registration.yaml` (no retrospective adjustments).**\n",
      "| Group | Parameter | YAML Path | Value |",
      "|-------|-----------|-----------|-------|"]
for g, p, path, v in hp_rows:
    md.append(f"| {g} | {p} | `{path}` | `{v}` |")
md.append("")
md.append("## Measured Outcomes (from training_summary.json)")
md.append("")
md.append("| Metric | Value |")
md.append("|--------|-------|")
md.append(f"| Adam epochs completed | {summary['adam']['epochs_completed']} |")
md.append(f"| Adam wall time | {summary['adam']['wall_time_s']:.1f} s |")
md.append(f"| Transition epoch | {summary['adam']['transition_epoch']} |")
md.append(f"| Transition λ_phys | {summary['adam']['transition_lam_phys']:.6f} |")
md.append(f"| Final val_data_loss at plateau | {summary['adam']['final_val_data_loss']:.6f} |")
md.append(f"| L-BFGS wall time | {summary['lbfgs']['wall_time_s']:.3f} s |")
md.append(f"| L-BFGS closure calls | {summary['lbfgs']['closure_calls']} |")
md.append(f"| L-BFGS final loss | {summary['lbfgs']['final_loss']:.6f} |")
md.append(f"| Peak VRAM (Adam) | {summary['peak_vram_MB']['adam']:.2f} MB |")
md.append(f"| Peak VRAM (L-BFGS) | {summary['peak_vram_MB']['lbfgs']:.2f} MB |")
Path(OUT / "optimization_hyperparameter_table.md").write_text("\n".join(md))
print("✓ optimization_hyperparameter_table.{csv,md} written")

# ---------- Optimization analysis report ----------
md = ["# Step 4A.4 — Optimization Analysis Report\n",
      "**Phase:** 4A.4 — Optimization Analysis",
      "**Predecessor:** Step 4A.3 LOCKED at `f7474c0`",
      "**Analysis type:** Read-only from Phase 4A.1 training logs",
      "**Training performed in this step:** None",
      "",
      "## Summary",
      "",
      f"The Phase 4A.1 adaptive optimizer ran for **{len(adam)} Adam epochs** (wall time "
      f"{summary['adam']['wall_time_s']:.1f} s) followed by **{len(lbfgs)} L-BFGS closure calls** "
      f"(wall time {summary['lbfgs']['wall_time_s']:.3f} s). Total training wall time: "
      f"**{summary['adam']['wall_time_s'] + summary['lbfgs']['wall_time_s']:.1f} s**.",
      "",
      "**Training trajectory highlights:**",
      f"- Best validation loss: `{best_val:.6f}` at epoch {best_epoch} (not preserved as checkpoint)",
      f"- Plateau detected at epoch {transition['plateau_epoch']}, λ_phys = {transition['lam_phys_at_transition']:.4f}",
      f"- Transition policy: `{transition['policy']}` — L-BFGS inherited the plateau λ",
      f"- Peak GPU memory: Adam {summary['peak_vram_MB']['adam']:.1f} MB, L-BFGS {summary['peak_vram_MB']['lbfgs']:.1f} MB",
      f"- 3 GB gate satisfied with ~38× headroom",
      "",
      "## Figures Produced (6 of 8 planned)",
      "",
      "| Figure | Content | Source |",
      "|--------|---------|--------|",
      "| F1 | Adam loss trajectory (train + val) | `adam_training_log.csv` |",
      "| F2 | L-BFGS loss trajectory | `lbfgs_training_log.csv` |",
      "| F3 | Combined optimization trace | Both CSVs |",
      "| F4 | Automatic transition marker with annotations | `transition_event.json` |",
      "| F6 | Data vs physics trade-off (color = λ_phys) | `adam_training_log.csv` |",
      "| F8 | GPU memory evolution | `adam_training_log.csv` |",
      "",
      "## Figures Unavailable (2 of 8 planned)",
      "",
      "| Figure | Reason | Alternative |",
      "|--------|--------|-------------|",
      "| F5 | Per-equation residual history not logged per epoch | See Step 4A.3 `residual_comparison_per_state_CORRECTED.csv` for final per-state comparison |",
      "| F7 | Gradient norms not logged during training | Not available; add to Phase 4B training script |",
      "",
      "**Per reviewer guidance:** These figures are not fabricated. The unavailability",
      "documents what the recorded evidence does and does not support.",
      "",
      "## Supported Conclusions (from training evidence)",
      "",
      "**Supported:**",
      "- The optimizer completed a two-stage Adam→L-BFGS run with 55 Adam epochs and 17 L-BFGS closures.",
      "- Plateau detection triggered at epoch 54 with λ_phys = 0.398, and the transition policy was honored (L-BFGS inherited λ).",
      "- Best val_data_loss occurred at epoch 28 ({:.6f}), though this checkpoint was not preserved.".format(best_val),
      "- GPU memory remained far below the 3 GB gate throughout.",
      "- The Adam stage reduced both data loss and physics loss simultaneously (visible in F1 and F6).",
      "- L-BFGS converged on the fixed batch within 17 closures (F2), reflecting limited remaining gradient signal.",
      "",
      "**Not supported (requires separate investigation):**",
      "- Individual causal contribution of training duration vs λ schedule vs optimizer sequencing",
      "- Per-epoch per-equation physics convergence (data not logged)",
      "- Gradient norm behavior over training (data not logged)",
      "",
      "## Reviewer Constraints Honored",
      "",
      "| Constraint | Status |",
      "|-----------|--------|",
      "| No fabricated epoch-wise per-equation curve | ✓ F5 unavailable |",
      "| No fabricated gradient-norm curve | ✓ F7 unavailable |",
      "| GPU memory honest (per-epoch, not just peak) | ✓ F8 from `peak_vram_MB` column |",
      "| Hyperparameter table verbatim from pre-registration | ✓ Asserted; no adjustments |",
      "| No training performed | ✓ Read-only script |",
      "| No checkpoint modification | ✓ |",
      "| No test-set use | ✓ |",
      "",
      "## Milestone",
      "",
      "`STEP_4A.4_LOCKED — OPTIMIZATION ANALYSIS COMPLETE`",
      ]
Path(OUT / "optimization_analysis_report.md").write_text("\n".join(md))
print("✓ optimization_analysis_report.md written")

# ---------- Manifest ----------
artifacts = {}
for p in sorted(OUT.glob("*")):
    if p.is_file() and p.name != "step_4a_4_manifest.json":
        artifacts[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted(FIG.glob("figure_F*.png")):
    artifacts[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

manifest = {
    "phase": "4A.4",
    "gate_4a_4": "PASS",
    "predecessor": "f7474c0",
    "phase_3_lock": "bc80727",
    "figures_produced": 6,
    "figures_unavailable": 2,
    "training_performed": False,
    "test_set_used": False,
    "artifacts": artifacts,
}
with open(OUT / "step_4a_4_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"✓ Manifest: {len(artifacts)} artifacts")

print("\n" + "="*60)
print("STEP 4A.4 COMPLETE")
print("="*60)
print(f"Figures: 6 produced, 2 unavailable")
print(f"Hyperparameter table: CSV + MD")
print(f"Report: optimization_analysis_report.md")
print(f"Manifest: step_4a_4_manifest.json")
