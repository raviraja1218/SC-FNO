"""Step 4A.2.R1 — Audit aggregate physics loss vs per-state residuals.

Reconciles L_physics = 2.03e-6 with per-state R_n = 1897.

Key question: how does compute_physics_loss aggregate the 9 states?
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

CKPT = "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
BATCH = 16
NQ = 50

device = "cuda"

# Load model
model, ckpt = load_phase3_pi_fno_model(CKPT, device=device)
model.eval()

# Load config
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
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
scaling = json.load(open("results/phase_3/loss_integration/per_state_scaling.json"))
per_state_scales = {s: scaling["per_state"][s]["residual_p95"] for s in STATES}
print(f"Per-state scales (residual_p95):")
for s in STATES:
    print(f"  {s}: {per_state_scales[s]:.6e}")

# Load one val batch
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
def _norm_states(t):
    o = np.zeros_like(t)
    for j,k in enumerate(STATES):
        o[:,j] = (t[:,j] - norm[k]["mean"]) / norm[k]["std"]
    return o
X_all, Y_all = [], []
for sid in val_ids[:BATCH]:
    d = RAW / sid
    f = np.load(d / "forcing.npy"); tj = np.load(d / "trajectory.npy")
    X_all.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
    Y_all.append(_norm_states(tj).T)
X = torch.tensor(np.stack(X_all), dtype=torch.float32).unsqueeze(1).to(device)
Y = torch.tensor(np.stack(Y_all), dtype=torch.float32).to(device)
B = X.shape[0]
t_q = torch.linspace(0.5, 9.5, NQ, device=device).unsqueeze(0).expand(B, -1).contiguous()

print(f"\n=== aggregate loss_fn.compute_physics_loss ===")
L_agg = loss_fn.compute_physics_loss(model, X, t_q, lambda tq: torch.zeros_like(tq), kin, th)
print(f"  L_physics = {L_agg.item():.6e}")

print(f"\n=== Internal path via compute_du_dt + compute_residuals ===")
# This mirrors what compute_physics_loss does internally
z, u_hat, du_hat_dt = compute_du_dt(model, X, t_q)
print(f"  z shape: {z.shape}")
print(f"  u_hat shape: {u_hat.shape}")

# What does _denorm_batch do?
u_phys = loss_fn._denorm_batch(u_hat) if loss_fn.state_scales else u_hat
print(f"  u_phys: shape={u_phys.shape}, mean={u_phys.mean().item():.4f}, "
      f"first5_n: {u_phys[0,0,:5].detach().cpu().numpy()}")

# Compute residuals
def rho_ext_fn(tq):
    if tq.dim() == 1:
        tq = tq.unsqueeze(0).expand(B, -1)
    return torch.stack([
        torch.nn.functional.interpolate(
            X[b, 0].unsqueeze(0).unsqueeze(0),
            size=tq.shape[1], mode="linear", align_corners=True
        ).squeeze() * norm["rho_ext"]["std"] + norm["rho_ext"]["mean"]
        for b in range(B)
    ], dim=0)

R = compute_residuals(
    u_hat=u_phys, du_hat_dt=du_hat_dt, t_query=t_q,
    rho_ext_fn=rho_ext_fn, kin=kin, th=th,
)
stacked = R.stacked()  # [B, 9, NQ]
print(f"  stacked residual shape: {stacked.shape}")
print(f"  raw R_n mean |value|: {stacked[:,0,:].abs().mean().item():.4e}")
print(f"  raw R_C1 mean |value|: {stacked[:,1,:].abs().mean().item():.4e}")

print(f"\n=== Per-state contributions using loss_fn.residual_scales ===")
per_state_contributions = {}
for j, s in enumerate(STATES):
    s_j = loss_fn.residual_scales[s]["residual_p95"]
    R_j = stacked[:, j, :] / s_j
    contribution = (R_j ** 2).mean().item()
    per_state_contributions[s] = {
        "residual_p95": s_j,
        "raw_mean_abs": stacked[:, j, :].abs().mean().item(),
        "squared_normalized_mean": contribution,
    }
    print(f"  {s:4s}: scale={s_j:.4e}, raw_|R|={stacked[:,j,:].abs().mean().item():.4e}, "
          f"(R/s)²_mean={contribution:.4e}")

print(f"\n=== Aggregate decomposition ===")
# loss_fn.compute_physics_loss source (from Phase 4A.1 report):
# for j, state in enumerate(self.STATES):
#     s_j = self.residual_scales[state]["residual_p95"]
#     R_j = stacked[:, j, :] / s_j
#     total = total + (R_j ** 2).mean()
# return total / len(self.STATES)
manual_agg = sum(per_state_contributions[s]["squared_normalized_mean"] for s in STATES) / 9
print(f"  Manual mean of 9 contributions: {manual_agg:.6e}")
print(f"  loss_fn.compute_physics_loss:   {L_agg.item():.6e}")
print(f"  Match? {abs(manual_agg - L_agg.item()) < 1e-6}")

# NOW — what if R_n is excluded?
manual_no_rn = sum(per_state_contributions[s]["squared_normalized_mean"] 
                   for s in STATES if s != "n") / 8
print(f"\n  Manual mean EXCLUDING R_n (÷8): {manual_no_rn:.6e}")
print(f"  (This would match 2.03e-6 if R_n is masked)")

# Or what if the R_n residual is tiny in this batch?
print(f"\n  If R_n in this batch is small but large in OTHER batches, that's a clue")
print(f"  R_n contribution this batch: {per_state_contributions['n']['squared_normalized_mean']:.4e}")
