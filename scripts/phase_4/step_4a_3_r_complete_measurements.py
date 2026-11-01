"""Step 4A.3.R — Complete the three missing Step 4A.3 measurements.

Plan Step 4A.3 required:
  - 9-equation residuals ✅ (done in 4A.3 core)
  - IC residual ✅
  - Aggregate physics loss ✅
  - Dimensional residuals ✅
  - Data MSE ✅
  - Relative L2 error ← MISSING (this script)
  - Per-family errors ← MISSING (this script)
  - Per-scenario errors ← MISSING (this script)
  - 10^-5 target test ✅

Read-only. Uses 200 validation scenarios. No test-set use. No training.
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, "src")
from training.adaptive_optimizer import load_phase3_pi_fno_model

PHASE3_CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
PHASE4A1_CKPT = "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
OUT = Path("results/phase_4/phase_4a/residual_analysis")
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
BATCH = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load data ----------
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
assert len(val_ids) == 200
test_ids = set(pd.read_csv("results/phase_1/splitting/test_ids.csv")["scenario_id"].tolist())
assert not any(sid in test_ids for sid in val_ids), "Test scenarios in val set!"

norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]

def _norm_states(t):
    o = np.zeros_like(t)
    for j,k in enumerate(STATES):
        o[:,j] = (t[:,j] - norm[k]["mean"]) / norm[k]["std"]
    return o

X_all, Y_all = [], []
family_map = {}
for sid in val_ids:
    d = RAW / sid
    f = np.load(d / "forcing.npy")
    tj = np.load(d / "trajectory.npy")
    X_all.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
    Y_all.append(_norm_states(tj).T)
    # Family from metadata
    meta = json.load(open(d / "metadata.json"))
    family_map[sid] = meta.get("forcing_family", meta.get("family", "unknown"))

X_val = torch.tensor(np.stack(X_all), dtype=torch.float32).unsqueeze(1).to(device)
Y_val = torch.tensor(np.stack(Y_all), dtype=torch.float32).to(device)

# ---------- Per-scenario + per-family + relative L2 ----------
def measure(model_path, name):
    """Compute per-scenario rel L2 for each state + per-family + aggregate."""
    model, _ = load_phase3_pi_fno_model(model_path, device=device)
    model.eval()

    per_scenario = []  # list of dicts
    with torch.no_grad():
        for i, sid in enumerate(val_ids):
            x = X_val[i:i+1]
            y = Y_val[i:i+1]
            pred = model.fno(x)  # [1, 9, 200]

            # Per-scenario overall rel L2 (all states together)
            diff = pred - y
            num = torch.linalg.norm(diff.flatten()).item()
            den = torch.linalg.norm(y.flatten()).item()
            rel_l2_overall = num / max(den, 1e-12)

            # Per-state rel L2 within this scenario
            state_rel_l2 = {}
            for j, s in enumerate(STATES):
                num_j = torch.linalg.norm(diff[0, j, :]).item()
                den_j = torch.linalg.norm(y[0, j, :]).item()
                state_rel_l2[s] = num_j / max(den_j, 1e-12)

            per_scenario.append({
                "scenario_id": sid,
                "family": family_map[sid],
                "rel_l2_overall": rel_l2_overall,
                **{f"rel_l2_{s}": state_rel_l2[s] for s in STATES},
            })

    df = pd.DataFrame(per_scenario)

    # Per-family aggregate
    family_stats = df.groupby("family").agg({
        "rel_l2_overall": ["median", "mean", "std", "count"],
        **{f"rel_l2_{s}": "median" for s in STATES},
    }).reset_index()

    # Overall summary
    summary = {
        "model": name,
        "ckpt_path": model_path,
        "n_scenarios": len(df),
        "rel_l2_overall_median": float(df["rel_l2_overall"].median()),
        "rel_l2_overall_mean": float(df["rel_l2_overall"].mean()),
        "rel_l2_overall_std": float(df["rel_l2_overall"].std()),
        "rel_l2_overall_min": float(df["rel_l2_overall"].min()),
        "rel_l2_overall_max": float(df["rel_l2_overall"].max()),
        "rel_l2_per_state_median": {s: float(df[f"rel_l2_{s}"].median()) for s in STATES},
        "rel_l2_per_state_mean": {s: float(df[f"rel_l2_{s}"].mean()) for s in STATES},
    }

    # Per-family metrics (median rel L2 per family per state)
    per_family = {}
    for fam in df["family"].unique():
        fam_df = df[df["family"] == fam]
        per_family[fam] = {
            "n_scenarios": len(fam_df),
            "rel_l2_overall_median": float(fam_df["rel_l2_overall"].median()),
            "rel_l2_overall_mean": float(fam_df["rel_l2_overall"].mean()),
            "per_state_median": {s: float(fam_df[f"rel_l2_{s}"].median()) for s in STATES},
        }

    return summary, per_family, df

print("=== Measuring Phase 3 PI-FNO ===")
p3_summary, p3_family, p3_df = measure(PHASE3_CKPT, "phase3_pi_fno")
print(f"  median rel L2 = {p3_summary['rel_l2_overall_median']:.6e}")
print(f"  mean   rel L2 = {p3_summary['rel_l2_overall_mean']:.6e}")

print("\n=== Measuring Phase 4A.1 final ===")
p4_summary, p4_family, p4_df = measure(PHASE4A1_CKPT, "phase4a1_final")
print(f"  median rel L2 = {p4_summary['rel_l2_overall_median']:.6e}")
print(f"  mean   rel L2 = {p4_summary['rel_l2_overall_mean']:.6e}")

# ---------- Save per-scenario CSV ----------
p3_df.to_csv(OUT / "per_scenario_metrics_phase3.csv", index=False)
p4_df.to_csv(OUT / "per_scenario_metrics_phase4a1.csv", index=False)

# Combined per-scenario comparison
combined_df = p3_df[["scenario_id", "family", "rel_l2_overall"]].rename(
    columns={"rel_l2_overall": "phase3_rel_l2"}
).merge(
    p4_df[["scenario_id", "rel_l2_overall"]].rename(columns={"rel_l2_overall": "phase4a1_rel_l2"}),
    on="scenario_id"
)
combined_df["delta"] = combined_df["phase3_rel_l2"] - combined_df["phase4a1_rel_l2"]
combined_df["pct_change"] = (combined_df["phase4a1_rel_l2"] - combined_df["phase3_rel_l2"]) / combined_df["phase3_rel_l2"] * 100
combined_df.to_csv(OUT / "per_scenario_comparison.csv", index=False)

# ---------- Per-family table ----------
per_family_out = {
    "phase3": p3_family,
    "phase4a1": p4_family,
    "delta_median": {},
}
for fam in p3_family:
    if fam in p4_family:
        p3_m = p3_family[fam]["rel_l2_overall_median"]
        p4_m = p4_family[fam]["rel_l2_overall_median"]
        per_family_out["delta_median"][fam] = {
            "phase3": p3_m,
            "phase4a1": p4_m,
            "abs_delta": p4_m - p3_m,
            "pct_change": (p4_m - p3_m) / p3_m * 100,
        }
with open(OUT / "per_family_metrics.json", "w") as f:
    json.dump(per_family_out, f, indent=2)

# Per-family CSV
rows = []
for fam in p3_family:
    rows.append({
        "family": fam,
        "phase3_rel_l2_median": p3_family[fam]["rel_l2_overall_median"],
        "phase4a1_rel_l2_median": p4_family[fam]["rel_l2_overall_median"],
        "delta": p4_family[fam]["rel_l2_overall_median"] - p3_family[fam]["rel_l2_overall_median"],
        "pct_change": (p4_family[fam]["rel_l2_overall_median"] - p3_family[fam]["rel_l2_overall_median"]) / p3_family[fam]["rel_l2_overall_median"] * 100,
        "n_scenarios": p3_family[fam]["n_scenarios"],
    })
pd.DataFrame(rows).to_csv(OUT / "per_family_metrics.csv", index=False)

# ---------- Relative L2 aggregate JSON ----------
rel_l2_out = {
    "phase3": p3_summary,
    "phase4a1": p4_summary,
    "delta": {
        "median_rel_l2_abs": p4_summary["rel_l2_overall_median"] - p3_summary["rel_l2_overall_median"],
        "median_rel_l2_pct": (p4_summary["rel_l2_overall_median"] - p3_summary["rel_l2_overall_median"]) / p3_summary["rel_l2_overall_median"] * 100,
        "mean_rel_l2_abs": p4_summary["rel_l2_overall_mean"] - p3_summary["rel_l2_overall_mean"],
        "mean_rel_l2_pct": (p4_summary["rel_l2_overall_mean"] - p3_summary["rel_l2_overall_mean"]) / p3_summary["rel_l2_overall_mean"] * 100,
    },
}
with open(OUT / "relative_l2_metrics.json", "w") as f:
    json.dump(rel_l2_out, f, indent=2)

# ---------- Print summary tables ----------
print("\n" + "="*70)
print("RELATIVE L2 (per-scenario, 200 validation scenarios)")
print("="*70)
print(f"{'Metric':<30s} {'Phase 3':>15s} {'Phase 4A.1':>15s} {'Δ%':>10s}")
print(f"{'rel L2 median':<30s} {p3_summary['rel_l2_overall_median']:>15.6e} "
      f"{p4_summary['rel_l2_overall_median']:>15.6e} "
      f"{rel_l2_out['delta']['median_rel_l2_pct']:>+9.2f}%")
print(f"{'rel L2 mean':<30s} {p3_summary['rel_l2_overall_mean']:>15.6e} "
      f"{p4_summary['rel_l2_overall_mean']:>15.6e} "
      f"{rel_l2_out['delta']['mean_rel_l2_pct']:>+9.2f}%")

print("\n" + "="*70)
print("PER-FAMILY (median rel L2)")
print("="*70)
print(f"{'Family':<15s} {'Phase 3':>14s} {'Phase 4A.1':>14s} {'Δ%':>10s} {'n':>5s}")
for fam in sorted(p3_family.keys()):
    p3m = p3_family[fam]["rel_l2_overall_median"]
    p4m = p4_family[fam]["rel_l2_overall_median"]
    pct = (p4m - p3m) / p3m * 100
    n = p3_family[fam]["n_scenarios"]
    print(f"{fam:<15s} {p3m:>14.4e} {p4m:>14.4e} {pct:>+9.2f}% {n:>5d}")

print("\n" + "="*70)
print("PER-STATE (median rel L2 across 200 val scenarios)")
print("="*70)
print(f"{'State':<6s} {'Phase 3':>14s} {'Phase 4A.1':>14s} {'Δ%':>10s}")
for s in STATES:
    p3s = p3_summary["rel_l2_per_state_median"][s]
    p4s = p4_summary["rel_l2_per_state_median"][s]
    pct = (p4s - p3s) / p3s * 100
    print(f"{s:<6s} {p3s:>14.4e} {p4s:>14.4e} {pct:>+9.2f}%")

print("\n✓ Saved:")
print(f"  {OUT}/per_scenario_metrics_phase3.csv")
print(f"  {OUT}/per_scenario_metrics_phase4a1.csv")
print(f"  {OUT}/per_scenario_comparison.csv")
print(f"  {OUT}/per_family_metrics.json")
print(f"  {OUT}/per_family_metrics.csv")
print(f"  {OUT}/relative_l2_metrics.json")

# Scenario-level win/loss for context
wins = (combined_df["delta"] > 0).sum()
losses = (combined_df["delta"] < 0).sum()
ties = (combined_df["delta"] == 0).sum()
print(f"\nScenario-level: Phase 3 wins {wins}/200, Phase 4A.1 wins {losses}/200, ties {ties}")
