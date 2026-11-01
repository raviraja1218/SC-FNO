"""Step 4A.0.R4 — Trace R_n through compute_physics_loss to explain 1900 vs 1.69e-3."""
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
from physics_informed.derivatives import load_phase2_fno, SCFNOModel, compute_du_dt
from physics_informed.decoder import build_decoder
from physics_informed.residuals import compute_residuals
from physics_informed.losses import load_composite_loss

CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
fno = load_phase2_fno(device=device)
decoder = build_decoder(seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd = {k:v for k,v in ckpt["model_state_dict"].items() if k != "_metadata"}
model.load_state_dict(sd, strict=False); model.eval()

# Kin/th
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg  = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
def s(v): return float(v[0]) if isinstance(v, list) else float(v)
n = kin_cfg["neutronics"]
kin = KineticsParameters(
    beta_eff=s(n["beta_eff"]), Lambda=s(n["Lambda"]),
    beta_i=[s(g["beta_i"]) for g in n["precursor_groups"]],
    lambda_i=[s(g["lambda_i"]) for g in n["precursor_groups"]],
)
fb = th_cfg["reactivity_feedback"]
th = THParameters(
    M_f=s(th_cfg["fuel"]["M_f"]), c_p_f=s(th_cfg["fuel"]["c_p_f"]),
    f_f=s(th_cfg["fuel"]["f_f"]), M_c=s(th_cfg["coolant"]["M_c"]),
    c_p_c=s(th_cfg["coolant"]["c_p_c"]), m_dot=s(th_cfg["coolant"]["m_dot"]),
    T_in=s(th_cfg["coolant"]["T_in"]), U=s(th_cfg["heat_transfer"]["U"]),
    A=s(th_cfg["heat_transfer"]["A"]), P0=s(th_cfg["power"]["P0"]),
    alpha_f=s(fb["alpha_f"]), alpha_c=s(fb["alpha_c"]),
    T_f_ref=s(fb["T_f_ref"]), T_c_ref=s(fb["T_c_ref"]),
)
loss_fn = load_composite_loss()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]

# Load ONE val batch
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()[:16]
forcing_list, traj_list = [], []
for sid in val_ids:
    d = RAW / sid
    f = np.load(d / "forcing.npy"); tj = np.load(d / "trajectory.npy")
    forcing_list.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
    t_n = np.zeros_like(tj)
    for j,k in enumerate(STATES): t_n[:,j] = (tj[:,j] - norm[k]["mean"]) / norm[k]["std"]
    traj_list.append(t_n.T)

x = torch.tensor(np.stack(forcing_list), dtype=torch.float32).unsqueeze(1).to(device)
B = x.shape[0]
t_q = torch.linspace(0.5, 9.5, 50, device=device).unsqueeze(0).expand(B, -1).contiguous()

# ---- Diagnostic 1: Raw compute_du_dt output magnitudes ----
z, u_hat_norm, du_hat_dt_norm = compute_du_dt(model, x, t_q)
print("compute_du_dt output magnitudes (first batch element, mean |value|):")
for j, st in enumerate(STATES):
    print(f"  {st}: u_hat_norm={u_hat_norm[0,j].abs().mean().item():.4e}, "
          f"du_hat_dt_norm={du_hat_dt_norm[0,j].abs().mean().item():.4e}")

# ---- Diagnostic 2: compare du/dt magnitude to expected physical du/dt ----
print("\nExpected physical du/dt magnitude (from GT via finite difference):")
# GT finite difference on normalized trajectory, times norm_std
for j, st in enumerate(STATES):
    t_grid = torch.linspace(0, 10, 200, device=device)
    y_phys = torch.tensor(np.load(RAW / val_ids[0] / "trajectory.npy"), dtype=torch.float32, device=device)[:, j]
    dydt_phys = torch.gradient(y_phys, spacing=(t_grid,), dim=0)[0]
    print(f"  {st}: |dy/dt_phys| mean = {dydt_phys.abs().mean().item():.4e} (physical units/s)")

# ---- Diagnostic 3: Two R_n computations ----
# Path A (Phase 3 authoritative — compute_physics_loss internal)
L_phys_agg = loss_fn.compute_physics_loss(
    model, x, t_q, lambda tq: torch.zeros_like(tq), kin, th
)
print(f"\nLoss_fn.compute_physics_loss aggregate: {float(L_phys_agg):.6e}")

# Path B (my manual — with denormalized du/dt)
with torch.enable_grad():
    z2, u_hat_n2, du_hat_n2 = compute_du_dt(model, x, t_q)
    u_phys = loss_fn._denorm_batch(u_hat_n2)
    du_phys = du_hat_n2.clone()
    for j, st in enumerate(STATES):
        du_phys[:, j, :] = du_hat_n2[:, j, :] * norm[st]["std"]
    R = compute_residuals(
        u_hat=u_phys, du_hat_dt=du_phys, t_query=t_q,
        rho_ext_fn=lambda tq: torch.zeros_like(tq), kin=kin, th=th,
    )
    stacked = R.stacked()
    rn_diag = stacked[:, 0, :]  # R_n dimensional
    print(f"\nManual diagnostic (denorm du by norm_std):")
    print(f"  R_n dimensional: mean |value| = {rn_diag.abs().mean().item():.4e}")
    s_n = loss_fn.residual_scales["n"]["residual_p95"]
    print(f"  s_n (residual_p95 for n) = {s_n}")
    print(f"  R_n / s_n mean |value| = {(rn_diag / s_n).abs().mean().item():.4e}")
    print(f"  (R_n / s_n)^2 mean = {((rn_diag / s_n) ** 2).mean().item():.4e}")

# ---- Diagnostic 4: Same computation but WITHOUT denorm du ----
with torch.enable_grad():
    z3, u_hat_n3, du_hat_n3 = compute_du_dt(model, x, t_q)
    u_phys3 = loss_fn._denorm_batch(u_hat_n3)
    # PASS du_hat_dt as-is (matching what compute_physics_loss does)
    R3 = compute_residuals(
        u_hat=u_phys3, du_hat_dt=du_hat_n3, t_query=t_q,
        rho_ext_fn=lambda tq: torch.zeros_like(tq), kin=kin, th=th,
    )
    stacked3 = R3.stacked()
    rn_diag3 = stacked3[:, 0, :]
    print(f"\nManual diagnostic (NO du denorm — matching compute_physics_loss):")
    print(f"  R_n dimensional: mean |value| = {rn_diag3.abs().mean().item():.4e}")
    print(f"  (R_n / s_n)^2 mean = {((rn_diag3 / s_n) ** 2).mean().item():.4e}")

out = {
    "aggregate_loss_fn": float(L_phys_agg),
    "manual_with_denorm_dn": float(((rn_diag / s_n) ** 2).mean().item()),
    "manual_no_denorm_dn": float(((rn_diag3 / s_n) ** 2).mean().item()),
    "s_n": float(s_n),
    "diagnosis": "If manual_no_denorm_dn matches aggregate/9 closely, then compute_physics_loss passes raw du_hat_dt and compute_du_dt returns physical derivative. If manual_with_denorm matches, then compute_du_dt returns normalized derivative."
}

with open("results/phase_4/phase_4a/preflight/rn_trace.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n" + json.dumps(out, indent=2))
print("\n=== 4A.0.R4 R_n trace: DONE ===")
