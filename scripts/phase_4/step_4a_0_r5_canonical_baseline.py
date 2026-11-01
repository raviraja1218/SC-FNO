"""Step 4A.0.R5 — Recompute canonical baseline with Protocol A (BATCH=16).

Protocol A (Phase 3 exact):
  - BATCH = 16
  - Per-batch data loss = mean over B x 9 x 200 elements of (pred - y)^2
  - Aggregate data loss = np.mean(per_batch_losses)  [equal weight per batch]
This reproduces Phase 3's recorded val_data_loss = 0.066331 bit-exact.
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
from physics_informed.derivatives import load_phase2_fno, SCFNOModel
from physics_informed.decoder import build_decoder
from physics_informed.losses import load_composite_loss

CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# Load model
fno = load_phase2_fno(device=device)
decoder = build_decoder(seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd = {k:v for k,v in ckpt["model_state_dict"].items() if k != "_metadata"}
model.load_state_dict(sd, strict=False); model.eval()

# Kin / TH
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg  = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
def s(v): return float(v[0]) if isinstance(v, list) else float(v)
n_ = kin_cfg["neutronics"]
kin = KineticsParameters(
    beta_eff=s(n_["beta_eff"]), Lambda=s(n_["Lambda"]),
    beta_i=[s(g["beta_i"]) for g in n_["precursor_groups"]],
    lambda_i=[s(g["lambda_i"]) for g in n_["precursor_groups"]],
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

val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()

def norm_states(t):
    o = np.zeros_like(t)
    for j,k in enumerate(STATES):
        o[:,j] = (t[:,j] - norm[k]["mean"]) / norm[k]["std"]
    return o

X_all, Y_all = [], []
for sid in val_ids:
    d = RAW / sid
    f = np.load(d / "forcing.npy"); tj = np.load(d / "trajectory.npy")
    X_all.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
    Y_all.append(norm_states(tj).T)

X_all = torch.tensor(np.stack(X_all), dtype=torch.float32).unsqueeze(1).to(device)
Y_all = torch.tensor(np.stack(Y_all), dtype=torch.float32).to(device)

BATCH = 16
data_losses, phys_losses, ic_losses = [], [], []

for i in range(0, len(val_ids), BATCH):
    x = X_all[i:i+BATCH]; y = Y_all[i:i+BATCH]; B = x.shape[0]

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

mean_data = float(np.mean(data_losses))
mean_phys = float(np.mean(phys_losses))
mean_ic = float(np.mean(ic_losses))
lam = float(ckpt.get("lam_phys", 1.0))
mean_total = mean_data + lam * mean_phys + mean_ic
peak_alloc = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

out = {
    "protocol": "A_phase3_exact_batch16",
    "n_val_scenarios": len(val_ids),
    "initial_data_loss": mean_data,
    "initial_physics_loss": mean_phys,
    "initial_ic_loss": mean_ic,
    "initial_total_loss": mean_total,
    "lam_phys_from_ckpt": lam,
    "peak_vram_allocated_MB": peak_alloc,
    "phase3_reference_data_loss": 0.066331,
    "phase3_reference_physics_loss": 8.56e-4,
    "delta_data_vs_phase3": mean_data - 0.066331,
    "delta_phys_vs_phase3": mean_phys - 8.56e-4,
}

with open("results/phase_4/phase_4a/preflight/baseline_metrics_canonical.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
print("\n=== 4A.0.R5 Canonical baseline (Protocol A, BATCH=16): DONE ===")
