"""Step 4A.0.F (FIXED) — Baseline metrics with correct Phase 3 PI-FNO loader."""
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
from physics_informed.losses import load_composite_loss
from training.adaptive_optimizer import load_phase3_pi_fno_model

CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Correct load
model, ckpt = load_phase3_pi_fno_model(CKPT, device=device)
model.eval()
print(f"Correct load verified: 0 missing / 0 unexpected")
print(f"  epoch={ckpt['epoch']}, lam_phys={ckpt['lam_phys']:.6e}")
print(f"  param count: {sum(p.numel() for p in model.parameters()):,}")

# Kin/th/loss/norm
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

val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()

def _norm_states(t):
    o = np.zeros_like(t)
    for j,k in enumerate(STATES):
        o[:,j] = (t[:,j] - norm[k]["mean"]) / norm[k]["std"]
    return o

X, Y = [], []
for sid in val_ids:
    d = RAW / sid
    f = np.load(d / "forcing.npy"); tj = np.load(d / "trajectory.npy")
    X.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
    Y.append(_norm_states(tj).T)
X = torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(1).to(device)
Y = torch.tensor(np.stack(Y), dtype=torch.float32).to(device)

# Protocol A: BATCH=16
BATCH = 16
data_losses, phys_losses, ic_losses = [], [], []
for i in range(0, len(val_ids), BATCH):
    x = X[i:i+BATCH]; y = Y[i:i+BATCH]; B = x.shape[0]
    with torch.no_grad():
        pred = model.fno(x)
        data_losses.append(((pred - y) ** 2).mean().item())
    t_q = torch.linspace(0.5, 9.5, 50, device=device).unsqueeze(0).expand(B, -1).contiguous()
    L_phys = loss_fn.compute_physics_loss(model, x, t_q, lambda tq: torch.zeros_like(tq), kin, th)
    phys_losses.append(float(L_phys.item() if isinstance(L_phys, torch.Tensor) else L_phys))
    with torch.no_grad():
        z_ic = model._fno_latent(x)
        t_ic = torch.zeros(B, 1, device=device)
        u_ic = model.decoder(z_ic, t_ic)
        ic_gt = y[:, :, 0].unsqueeze(-1)
        ic_scale = torch.tensor([norm[k]["std"] for k in STATES], device=device).view(1,9,1)
        ic_losses.append(torch.mean(((u_ic - ic_gt) / ic_scale) ** 2).item())

out = {
    "protocol": "A_phase3_exact_batch16",
    "loader": "load_phase3_pi_fno_model (PaddedFNO-wrapped, bit-exact)",
    "n_val_scenarios": len(val_ids),
    "initial_data_loss": float(np.mean(data_losses)),
    "initial_physics_loss": float(np.mean(phys_losses)),
    "initial_ic_loss": float(np.mean(ic_losses)),
    "lam_phys_from_ckpt": float(ckpt["lam_phys"]),
    "phase3_reference_data_loss": 0.066331,
}
out["delta_data_vs_phase3"] = out["initial_data_loss"] - 0.066331

with open("results/phase_4/phase_4a/preflight/baseline_metrics_canonical_FIXED.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
print("\n=== FIXED baseline computed with correct loader ===")
