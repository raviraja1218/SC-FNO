"""Step 4A.0.R3 — Reconcile the three val loss numbers.

Compute val data loss three ways on the same checkpoint + 200 val scenarios:
  A) Phase 3's exact protocol: np.mean of per-batch mean losses (BATCH=16)
  B) 4A.0.D protocol: mean of per-scenario MSEs (each scenario equal weight)
  C) 4A.0.F protocol: global mean over all B*9*200 elements

Then compare to Phase 3's recorded 0.066331.
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, "src")
from physics_informed.derivatives import load_phase2_fno, SCFNOModel
from physics_informed.decoder import build_decoder

CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

device = "cuda" if torch.cuda.is_available() else "cpu"

fno = load_phase2_fno(device=device)
decoder = build_decoder(seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd = {k:v for k,v in ckpt["model_state_dict"].items() if k != "_metadata"}
model.load_state_dict(sd, strict=False)
model.eval()

val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]

def norm_states(t):
    o = np.zeros_like(t)
    for j,k in enumerate(STATES):
        o[:,j] = (t[:,j] - norm[k]["mean"]) / norm[k]["std"]
    return o

# Load all 200 into memory once
X_all, Y_all = [], []
for sid in val_ids:
    d = RAW / sid
    f = np.load(d / "forcing.npy"); tj = np.load(d / "trajectory.npy")
    X_all.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
    Y_all.append(norm_states(tj).T)

X_all = torch.tensor(np.stack(X_all), dtype=torch.float32).unsqueeze(1).to(device)
Y_all = torch.tensor(np.stack(Y_all), dtype=torch.float32).to(device)

results = {}

with torch.no_grad():
    # A) Phase 3 exact: np.mean of per-batch mean losses with BATCH=16
    per_batch_losses = []
    for i in range(0, 200, 16):
        x = X_all[i:i+16]; y = Y_all[i:i+16]
        pred = model.fno(x)
        per_batch_losses.append(((pred - y) ** 2).mean().item())
    results["A_phase3_protocol_batch16"] = float(np.mean(per_batch_losses))

    # B) 4A.0.D protocol: mean of per-scenario MSE, BATCH=32
    per_scenario_mse = []
    for i in range(0, 200, 32):
        x = X_all[i:i+32]; y = Y_all[i:i+32]
        pred = model.fno(x)
        se = (pred - y) ** 2
        mse_scn = se.mean(dim=[1,2]).cpu().numpy()
        per_scenario_mse.extend(mse_scn.tolist())
    results["B_4a_0d_mean_of_scenario_mse"] = float(np.mean(per_scenario_mse))

    # C) 4A.0.F protocol: global mean over all elements
    total_se = 0.0; total_n = 0
    for i in range(0, 200, 32):
        x = X_all[i:i+32]; y = Y_all[i:i+32]
        pred = model.fno(x)
        total_se += ((pred - y) ** 2).sum().item()
        total_n += pred.numel()
    results["C_4a_0f_global_mean"] = float(total_se / total_n)

results["phase3_recorded"] = 0.066331

print(json.dumps(results, indent=2))
print()
print(f"A (Phase 3 protocol) vs Phase 3 recorded: Δ = {results['A_phase3_protocol_batch16'] - 0.066331:+.6f}")
print(f"B (4A.0.D) vs Phase 3 recorded:            Δ = {results['B_4a_0d_mean_of_scenario_mse'] - 0.066331:+.6f}")
print(f"C (4A.0.F) vs Phase 3 recorded:            Δ = {results['C_4a_0f_global_mean'] - 0.066331:+.6f}")
print()
print("Canonical recommendation: A (Phase 3 protocol) — this is what Phase 3 recorded.")

with open("results/phase_4/phase_4a/preflight/val_loss_unified.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== 4A.0.R3 Val loss reconciliation: DONE ===")
