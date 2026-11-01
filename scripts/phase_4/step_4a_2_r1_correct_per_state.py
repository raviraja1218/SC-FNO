"""Step 4A.2.R1 — Recompute correct per-state residuals.

Fixes the 4A.2 diagnostic bug: the previous per-state path manually
denormalized du_hat_dt, which was wrong. compute_physics_loss passes
du_hat_dt from compute_du_dt directly (already physical units).

This script uses the exact same path as compute_physics_loss for
per-state diagnostics, guaranteeing aggregate = mean(9 states).
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path

sys.path.insert(0, "src")
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.derivatives import compute_du_dt
from physics_informed.residuals import compute_residuals
from physics_informed.losses import load_composite_loss
from training.adaptive_optimizer import load_phase3_pi_fno_model

device = "cuda" if torch.cuda.is_available() else "cpu"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
BATCH = 16
NQ = 50
T_LO, T_HI = 0.5, 9.5

PHASE3_CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
PHASE4A1_CKPT = "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt"
OUT = Path("results/phase_4/phase_4a/residual_analysis")
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"

# Load config
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
def _s(v): return float(v[0]) if isinstance(v, list) else float(v)
n_ = kin_cfg["neutronics"]
kin = KineticsParameters(beta_eff=_s(n_["beta_eff"]), Lambda=_s(n_["Lambda"]),
    beta_i=[_s(g["beta_i"]) for g in n_["precursor_groups"]],
    lambda_i=[_s(g["lambda_i"]) for g in n_["precursor_groups"]])
fb = th_cfg["reactivity_feedback"]
th = THParameters(M_f=_s(th_cfg["fuel"]["M_f"]), c_p_f=_s(th_cfg["fuel"]["c_p_f"]),
    f_f=_s(th_cfg["fuel"]["f_f"]), M_c=_s(th_cfg["coolant"]["M_c"]),
    c_p_c=_s(th_cfg["coolant"]["c_p_c"]), m_dot=_s(th_cfg["coolant"]["m_dot"]),
    T_in=_s(th_cfg["coolant"]["T_in"]), U=_s(th_cfg["heat_transfer"]["U"]),
    A=_s(th_cfg["heat_transfer"]["A"]), P0=_s(th_cfg["power"]["P0"]),
    alpha_f=_s(fb["alpha_f"]), alpha_c=_s(fb["alpha_c"]),
    T_f_ref=_s(fb["T_f_ref"]), T_c_ref=_s(fb["T_c_ref"]))
loss_fn = load_composite_loss()

val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]

X_all = []
for sid in val_ids:
    d = RAW / sid
    f = np.load(d / "forcing.npy")
    X_all.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
X = torch.tensor(np.stack(X_all), dtype=torch.float32).unsqueeze(1).to(device)

def measure(model_path, name):
    model, _ = load_phase3_pi_fno_model(model_path, device=device)
    model.eval()

    per_state_sum = {s: 0.0 for s in STATES}
    per_state_dim_sum = {s: 0.0 for s in STATES}
    agg_sum = 0.0
    n_batches = 0

    for i in range(0, len(val_ids), BATCH):
        x = X[i:i+BATCH]
        B = x.shape[0]
        t_q = torch.linspace(T_LO, T_HI, NQ, device=device).unsqueeze(0).expand(B, -1).contiguous()

        # Aggregate (authoritative)
        with torch.enable_grad():
            L_agg = loss_fn.compute_physics_loss(model, x, t_q,
                lambda tq: torch.zeros_like(tq), kin, th)
            agg_sum += L_agg.item()

            # Per-state — IDENTICAL path as compute_physics_loss
            z, u_hat, du_hat_dt = compute_du_dt(model, x, t_q)
            u_phys = loss_fn._denorm_batch(u_hat) if loss_fn.state_scales else u_hat
            R = compute_residuals(u_hat=u_phys, du_hat_dt=du_hat_dt,
                t_query=t_q, rho_ext_fn=lambda tq: torch.zeros_like(tq),
                kin=kin, th=th)
            stacked = R.stacked()
            for j, s in enumerate(STATES):
                s_j = loss_fn.residual_scales[s]["residual_p95"]
                per_state_sum[s] += (stacked[:, j, :] / s_j).pow(2).mean().item()
                per_state_dim_sum[s] += stacked[:, j, :].abs().mean().item()
        n_batches += 1

    return {
        "name": name,
        "ckpt_path": model_path,
        "L_physics_aggregate": agg_sum / n_batches,
        "per_state_normalized": {s: per_state_sum[s] / n_batches for s in STATES},
        "per_state_dimensional": {s: per_state_dim_sum[s] / n_batches for s in STATES},
        "n_batches": n_batches,
        "batch_size": BATCH,
        "nq": NQ,
        "t_query_range": [T_LO, T_HI],
    }

print("Measuring Phase 3 PI-FNO...")
p3 = measure(PHASE3_CKPT, "phase3_pi_fno")
print("Measuring Phase 4A.1 final...")
p4 = measure(PHASE4A1_CKPT, "phase4a1_final")

# Verify reconciliation
mean_p3 = sum(p3["per_state_normalized"].values()) / 9
mean_p4 = sum(p4["per_state_normalized"].values()) / 9
assert abs(mean_p3 - p3["L_physics_aggregate"]) < 1e-10, "Phase 3 reconciliation failed"
assert abs(mean_p4 - p4["L_physics_aggregate"]) < 1e-10, "Phase 4A.1 reconciliation failed"
print(f"\n✓ Phase 3 reconciliation: manual={mean_p3:.6e}, agg={p3['L_physics_aggregate']:.6e}")
print(f"✓ Phase 4A.1 reconciliation: manual={mean_p4:.6e}, agg={p4['L_physics_aggregate']:.6e}")

# Build correct CSV
import csv
corrected_csv = OUT / "residual_comparison_per_state_CORRECTED.csv"
with open(corrected_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["state", "unit",
                "phase3_normalized", "phase4a1_normalized",
                "delta_normalized", "pct_change", "ratio_x",
                "phase3_dimensional", "phase4a1_dimensional"])
    units = {"n":"1/s", "C1":"1/s","C2":"1/s","C3":"1/s","C4":"1/s","C5":"1/s","C6":"1/s",
             "Tf":"K/s", "Tc":"K/s"}
    for s in STATES:
        p3n = p3["per_state_normalized"][s]
        p4n = p4["per_state_normalized"][s]
        p3d = p3["per_state_dimensional"][s]
        p4d = p4["per_state_dimensional"][s]
        ratio = p3n / p4n if p4n > 0 else 0
        pct = (p4n - p3n) / p3n * 100 if p3n > 0 else 0
        w.writerow([s, units[s], f"{p3n:.6e}", f"{p4n:.6e}",
                    f"{p4n-p3n:+.6e}", f"{pct:+.2f}%", f"{ratio:.1f}x",
                    f"{p3d:.6e}", f"{p4d:.6e}"])

# Summary JSON
summary = {
    "correction": "Phase 4A.2.R1 — correct per-state residuals",
    "reason": "Previous diagnostic manually denormalized du_hat_dt; compute_physics_loss passes raw du_hat_dt from compute_du_dt (already physical units)",
    "phase3": p3,
    "phase4a1": p4,
    "aggregate": {
        "phase3": p3["L_physics_aggregate"],
        "phase4a1": p4["L_physics_aggregate"],
        "ratio": p3["L_physics_aggregate"] / p4["L_physics_aggregate"],
        "pct_reduction": (1 - p4["L_physics_aggregate"] / p3["L_physics_aggregate"]) * 100,
    },
    "reconciliation": {
        "phase3_manual_vs_agg_match": abs(mean_p3 - p3["L_physics_aggregate"]) < 1e-10,
        "phase4a1_manual_vs_agg_match": abs(mean_p4 - p4["L_physics_aggregate"]) < 1e-10,
    },
    "target_1e-5": {
        "target": 1e-5,
        "phase3_achieved": p3["L_physics_aggregate"] < 1e-5,
        "phase4a1_achieved": p4["L_physics_aggregate"] < 1e-5,
        "phase3_ratio": p3["L_physics_aggregate"] / 1e-5,
        "phase4a1_ratio": p4["L_physics_aggregate"] / 1e-5,
    },
}
with open(OUT / "residual_comparison_CORRECTED.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== CORRECTED TABLE ===")
print(f"{'state':6s} {'Phase3':>12s} {'Phase4A.1':>14s} {'Ratio':>10s} {'Δ%':>10s}")
for s in STATES:
    a = p3["per_state_normalized"][s]
    b = p4["per_state_normalized"][s]
    r = a / b if b > 0 else 0
    pct = (b - a) / a * 100
    print(f"{s:6s} {a:12.4e} {b:14.4e} {r:9.1f}x {pct:+9.2f}%")

print(f"\nAggregate Phase 3:   {p3['L_physics_aggregate']:.6e}")
print(f"Aggregate Phase 4A.1: {p4['L_physics_aggregate']:.6e}")
print(f"Improvement:          {p3['L_physics_aggregate'] / p4['L_physics_aggregate']:.1f}x")
print(f"\n✓ Saved corrected CSV + JSON")
