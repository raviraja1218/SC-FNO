"""Verify P-PI-FNO best checkpoint produces meaningful physics residuals on held-out validation."""
import sys, json
from pathlib import Path
import torch, numpy as np
sys.path.insert(0, "src")

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.decoder import build_decoder
from physics_informed.derivatives import load_phase2_fno, SCFNOModel
from physics_informed.losses import load_composite_loss
from training.train_sc_fno import PaddedFNO

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build P-PI-FNO model (padding=20)
fno_raw = load_phase2_fno(device=device)
fno = PaddedFNO(fno_raw, pad_size=20).to(device)
decoder = build_decoder(seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device)

# Load best checkpoint
ckpt = torch.load("models/sc_fno/phase_3/p_pi_fno/best_val.pt",
                  map_location=device, weights_only=False)
sd = ckpt["model_state_dict"]
if "_metadata" in sd:
    sd = {k: v for k, v in sd.items() if k != "_metadata"}
model.load_state_dict(sd)
model.eval()

# Load val scenarios
import pandas as pd
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()  # full 200
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

kin = KineticsParameters(beta_eff=0.006502, Lambda=1e-4,
    beta_i=[0.000215,0.001424,0.001274,0.002568,0.000748,0.000273],
    lambda_i=[0.0124,0.0305,0.1110,0.3010,1.1400,3.0100])
th = THParameters(M_f=27000., c_p_f=300., f_f=0.97, M_c=4000., c_p_c=5500.,
    m_dot=13636.36, T_in=563., U=3060., A=3000., P0=3e9,
    alpha_f=-3e-5, alpha_c=-5e-4, T_f_ref=900., T_c_ref=583.)

loss_fn = load_composite_loss()

res_data = []; res_phys = []; res_ic = []
for sid in val_ids:
    forcing = np.load(RAW / sid / "forcing.npy")
    traj = np.load(RAW / sid / "trajectory.npy")
    rho_n = (forcing - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
    u_n = np.zeros_like(traj)
    for j, s in enumerate(STATES):
        u_n[:, j] = (traj[:, j] - norm[s]["mean"]) / norm[s]["std"]
    rho_t = torch.tensor(rho_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    u_t = torch.tensor(u_n.T, dtype=torch.float32).unsqueeze(0).to(device)
    t_q = torch.linspace(0.5, 9.5, 50, device=device).unsqueeze(0)

    # Data and IC losses do not need autograd
    with torch.no_grad():
        L_d = loss_fn.compute_data_loss(model, rho_t, u_t)
        L_ic = loss_fn.compute_ic_loss(model, rho_t, u_t[:,:,0])

    # Physics loss REQUIRES autograd (compute_du_dt uses torch.autograd.grad)
    L_p = loss_fn.compute_physics_loss(
        model, rho_t, t_q, lambda tq: torch.zeros_like(tq), kin, th
    )
    L_p = L_p.detach()

    res_data.append(float(L_d.item()))
    res_phys.append(float(L_p.item()))
    res_ic.append(float(L_ic.item()))

out = {
    "checkpoint": "models/sc_fno/phase_3/p_pi_fno/best_val.pt",
    "n_val_scenarios": len(val_ids),
    "data_loss_mean": float(np.mean(res_data)),
    "data_loss_std": float(np.std(res_data)),
    "physics_loss_mean": float(np.mean(res_phys)),
    "physics_loss_std": float(np.std(res_phys)),
    "ic_loss_mean": float(np.mean(res_ic)),
    "ic_loss_std": float(np.std(res_ic)),
    "interpretation": (
        "If physics_loss_mean is near zero on held-out val AND data_loss_mean is also "
        "reasonable, then physics residuals are genuinely satisfied — not a degenerate "
        "minimisation. If physics_loss_mean is near zero but data_loss_mean is high, "
        "the model may have converged to a trivial equilibrium that satisfies physics "
        "but doesn't track the actual trajectory."
    ),
}
Path("results/phase_3/training/physics_sanity_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
