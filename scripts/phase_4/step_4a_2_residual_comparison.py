"""Step 4A.2 — Residual convergence assessment.

Compares Phase 3 PI-FNO vs Phase 4A.1 final on:
  - 9 per-state normalized residuals
  - Aggregate L_physics (via loss_fn.compute_physics_loss)
  - L_IC
  - L_data (Protocol A, BATCH=16)
  - Dimensional residuals
  - 10^-5 target check

Measurement-only. No training. No test-set use.
"""
import json
import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import yaml

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.derivatives import (
    compute_du_dt,
)
from physics_informed.residuals import compute_residuals
from physics_informed.losses import load_composite_loss
from training.adaptive_optimizer import load_phase3_pi_fno_model

# --- Safe JSON serialization (handles bool, np.bool_, np.float, etc.) ---
def _safe_json(o):
    """Recursively convert to JSON-serializable types."""
    if isinstance(o, bool):
        return bool(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _safe_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe_json(v) for v in o]
    return o


# ---------- Paths ----------
PHASE3_CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
PHASE4A1_CKPT = "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
OUT = Path("results/phase_4/phase_4a/residual_analysis")
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
BATCH = 16
NQ = 50
T_LO, T_HI = 0.5, 9.5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ---------- Verify Phase 3 SHA ----------
phase3_sha = hashlib.sha256(Path(PHASE3_CKPT).read_bytes()).hexdigest()
phase4a1_sha = hashlib.sha256(Path(PHASE4A1_CKPT).read_bytes()).hexdigest()
expected_phase3 = "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50"
assert phase3_sha == expected_phase3, f"Phase 3 SHA mismatch: {phase3_sha}"
print(f"Phase 3 PI-FNO SHA: {phase3_sha[:16]}... ✓")
print(f"Phase 4A.1 SHA: {phase4a1_sha[:16]}...")

# ---------- Load kin/th/loss/norm ----------
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg  = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
def _s(v): return float(v[0]) if isinstance(v, list) else float(v)
n_ = kin_cfg["neutronics"]
kin = KineticsParameters(
    beta_eff=_s(n_["beta_eff"]), Lambda=_s(n_["Lambda"]),
    beta_i=[_s(g["beta_i"]) for g in n_["precursor_groups"]],
    lambda_i=[_s(g["lambda_i"]) for g in n_["precursor_groups"]],
)
fb = th_cfg["reactivity_feedback"]
th = THParameters(
    M_f=_s(th_cfg["fuel"]["M_f"]), c_p_f=_s(th_cfg["fuel"]["c_p_f"]),
    f_f=_s(th_cfg["fuel"]["f_f"]), M_c=_s(th_cfg["coolant"]["M_c"]),
    c_p_c=_s(th_cfg["coolant"]["c_p_c"]), m_dot=_s(th_cfg["coolant"]["m_dot"]),
    T_in=_s(th_cfg["coolant"]["T_in"]), U=_s(th_cfg["heat_transfer"]["U"]),
    A=_s(th_cfg["heat_transfer"]["A"]), P0=_s(th_cfg["power"]["P0"]),
    alpha_f=_s(fb["alpha_f"]), alpha_c=_s(fb["alpha_c"]),
    T_f_ref=_s(fb["T_f_ref"]), T_c_ref=_s(fb["T_c_ref"]),
)
loss_fn = load_composite_loss()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
scaling = json.load(open("results/phase_3/loss_integration/per_state_scaling.json"))
per_state_scales = {s: scaling["per_state"][s]["residual_p95"] for s in STATES}

# ---------- Load validation data ----------
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
assert len(val_ids) == 200
# Verify no test-set contamination
test_ids = set(pd.read_csv("results/phase_1/splitting/test_ids.csv")["scenario_id"].tolist())
assert not any(sid in test_ids for sid in val_ids), "Test scenarios mixed into val set!"

def _norm_states(t):
    o = np.zeros_like(t)
    for j,k in enumerate(STATES):
        o[:,j] = (t[:,j] - norm[k]["mean"]) / norm[k]["std"]
    return o

X_all, Y_all = [], []
for sid in val_ids:
    d = RAW / sid
    f = np.load(d / "forcing.npy"); tj = np.load(d / "trajectory.npy")
    X_all.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
    Y_all.append(_norm_states(tj).T)
X_val = torch.tensor(np.stack(X_all), dtype=torch.float32).unsqueeze(1).to(device)
Y_val = torch.tensor(np.stack(Y_all), dtype=torch.float32).to(device)
print(f"Validation data: X={X_val.shape}, Y={Y_val.shape}")

# ---------- Measurement function ----------
def measure_model(model, name):
    """Measure all residuals + losses on 200 val scenarios."""
    model.eval()
    print(f"\n=== Measuring {name} ===")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Aggregate losses via loss_fn.compute_physics_loss + compute_data_loss + compute_ic_loss
    data_losses, phys_losses, ic_losses = [], [], []
    per_state_residuals_sum = {s: 0.0 for s in STATES}
    per_state_dimensional_sum = {s: 0.0 for s in STATES}
    n_batches = 0

    for i in range(0, len(val_ids), BATCH):
        x = X_val[i:i+BATCH]; y = Y_val[i:i+BATCH]; B = x.shape[0]

        # Data loss (Protocol A)
        with torch.no_grad():
            pred = model.fno(x)
            data_losses.append(((pred - y) ** 2).mean().item())

        # Physics loss — uses loss_fn.compute_physics_loss (authoritative)
        t_q = torch.linspace(T_LO, T_HI, NQ, device=device).unsqueeze(0).expand(B, -1).contiguous()
        L_phys = loss_fn.compute_physics_loss(
            model, x, t_q, lambda tq: torch.zeros_like(tq), kin, th
        )
        phys_losses.append(float(L_phys.item() if isinstance(L_phys, torch.Tensor) else L_phys))

        # IC loss
        with torch.no_grad():
            z_ic = model._fno_latent(x)
            t_ic = torch.zeros(B, 1, device=device)
            u_ic = model.decoder(z_ic, t_ic)
            ic_gt = y[:, :, 0].unsqueeze(-1)
            ic_scale = torch.tensor([norm[k]["std"] for k in STATES],
                                    device=device).view(1,9,1)
            ic_losses.append(torch.mean(((u_ic - ic_gt) / ic_scale) ** 2).item())

        # Per-state residuals — manual denorm, matching Phase 4A.0.R4 pattern
        with torch.enable_grad():
            z2, u_hat_norm, du_hat_dt_norm = compute_du_dt(model, x, t_q)
            u_phys = loss_fn._denorm_batch(u_hat_norm)
            du_phys = du_hat_dt_norm.clone()
            for j, s in enumerate(STATES):
                du_phys[:, j, :] = du_hat_dt_norm[:, j, :] * norm[s]["std"]

            def rho_ext_fn(tq):
                if tq.dim() == 1:
                    tq = tq.unsqueeze(0).expand(B, -1)
                return torch.stack([
                    torch.nn.functional.interpolate(
                        x[b, 0].unsqueeze(0).unsqueeze(0),
                        size=tq.shape[1], mode="linear", align_corners=True
                    ).squeeze() * norm["rho_ext"]["std"] + norm["rho_ext"]["mean"]
                    for b in range(B)
                ], dim=0)

            R = compute_residuals(
                u_hat=u_phys, du_hat_dt=du_phys, t_query=t_q,
                rho_ext_fn=rho_ext_fn, kin=kin, th=th,
            )
            stacked = R.stacked()  # [B, 9, NQ] dimensional residuals
            for j, s in enumerate(STATES):
                r_dim = stacked[:, j, :]
                per_state_dimensional_sum[s] += float(r_dim.abs().mean().item())
                s_j = per_state_scales[s]
                per_state_residuals_sum[s] += float((r_dim / s_j).pow(2).mean().item())

        n_batches += 1

    wall = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

    result = {
        "name": name,
        "ckpt_sha": phase3_sha if name == "phase3_pi_fno" else phase4a1_sha,
        "n_val_scenarios": len(val_ids),
        "n_batches": n_batches,
        "batch_size": BATCH,
        "nq": NQ,
        "t_query_range": [T_LO, T_HI],
        "L_data_protocol_A": float(np.mean(data_losses)),
        "L_physics": float(np.mean(phys_losses)),
        "L_IC": float(np.mean(ic_losses)),
        "per_state_normalized_residual": {s: per_state_residuals_sum[s] / n_batches for s in STATES},
        "per_state_dimensional_residual": {s: per_state_dimensional_sum[s] / n_batches for s in STATES},
        "wall_time_s": wall,
        "peak_vram_MB": peak_vram,
    }
    print(f"  L_data = {result['L_data_protocol_A']:.6f}")
    print(f"  L_physics = {result['L_physics']:.6e}")
    print(f"  L_IC = {result['L_IC']:.6e}")
    print(f"  Wall time: {wall:.1f}s, Peak VRAM: {peak_vram:.1f} MB")
    return result

# ---------- Run both measurements ----------
print("\nLoading Phase 3 PI-FNO...")
model_p3, ckpt_p3 = load_phase3_pi_fno_model(PHASE3_CKPT, device=device)
r_p3 = measure_model(model_p3, "phase3_pi_fno")

print("\nLoading Phase 4A.1 final...")
# Load 4A.1 final via same loader (it's also PaddedFNO-wrapped)
model_4a1, ckpt_4a1 = load_phase3_pi_fno_model(PHASE4A1_CKPT, device=device)
r_4a1 = measure_model(model_4a1, "phase4a1_final")

# ---------- Compute comparisons ----------
comparison = {
    "phase3": r_p3,
    "phase4a1": r_4a1,
    "delta": {
        "L_data_abs": r_4a1["L_data_protocol_A"] - r_p3["L_data_protocol_A"],
        "L_data_pct": (r_4a1["L_data_protocol_A"] - r_p3["L_data_protocol_A"]) / r_p3["L_data_protocol_A"] * 100,
        "L_physics_abs": r_4a1["L_physics"] - r_p3["L_physics"],
        "L_physics_pct": (r_4a1["L_physics"] - r_p3["L_physics"]) / r_p3["L_physics"] * 100,
        "L_IC_abs": r_4a1["L_IC"] - r_p3["L_IC"],
        "L_IC_pct": (r_4a1["L_IC"] - r_p3["L_IC"]) / max(r_p3["L_IC"], 1e-12) * 100,
    },
}
for s in STATES:
    p3 = r_p3["per_state_normalized_residual"][s]
    p4 = r_4a1["per_state_normalized_residual"][s]
    comparison["delta"][f"R_{s}_abs"] = p4 - p3
    comparison["delta"][f"R_{s}_pct"] = (p4 - p3) / max(p3, 1e-12) * 100

# 10^-5 target check
target = 1e-5
comparison["target_1e-5"] = {
    "target": target,
    "phase3_achieved": r_p3["L_physics"] < target,
    "phase4a1_achieved": r_4a1["L_physics"] < target,
    "phase3_ratio": r_p3["L_physics"] / target,
    "phase4a1_ratio": r_4a1["L_physics"] / target,
}

# Stability
comparison["stability"] = {
    "phase3_all_finite": all(np.isfinite(v) for v in r_p3["per_state_normalized_residual"].values()),
    "phase4a1_all_finite": all(np.isfinite(v) for v in r_4a1["per_state_normalized_residual"].values()),
    "phase3_L_physics_finite": np.isfinite(r_p3["L_physics"]),
    "phase4a1_L_physics_finite": np.isfinite(r_4a1["L_physics"]),
    "phase3_under_3GB": r_p3["peak_vram_MB"] < 3000,
    "phase4a1_under_3GB": r_4a1["peak_vram_MB"] < 3000,
}

with open(OUT / "residual_comparison_aggregate.json", "w") as f:
    json.dump(_safe_json(comparison), f, indent=2)

# CSV for per-state
import csv
with open(OUT / "residual_comparison_per_state.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["state", "unit",
                "phase3_normalized_residual", "phase4a1_normalized_residual",
                "delta_normalized", "pct_change",
                "phase3_dimensional_residual", "phase4a1_dimensional_residual"])
    units = {"n":"1/s", "C1":"1/s","C2":"1/s","C3":"1/s","C4":"1/s","C5":"1/s","C6":"1/s",
             "Tf":"K/s", "Tc":"K/s"}
    for s in STATES:
        p3n = r_p3["per_state_normalized_residual"][s]
        p4n = r_4a1["per_state_normalized_residual"][s]
        p3d = r_p3["per_state_dimensional_residual"][s]
        p4d = r_4a1["per_state_dimensional_residual"][s]
        w.writerow([s, units[s], f"{p3n:.6e}", f"{p4n:.6e}",
                    f"{p4n-p3n:+.6e}", f"{(p4n-p3n)/max(p3n,1e-12)*100:+.2f}%",
                    f"{p3d:.6e}", f"{p4d:.6e}"])

print("\n" + "="*60)
print("MEASUREMENT COMPLETE")
print("="*60)
print(json.dumps(comparison["delta"], indent=2))
print("\n10^-5 target check:")
print(json.dumps(comparison["target_1e-5"], indent=2))
print("\nStability:")
print(json.dumps(_safe_json(comparison["stability"]), indent=2))
