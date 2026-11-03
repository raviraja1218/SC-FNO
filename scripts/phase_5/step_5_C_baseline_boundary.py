"""Compute Phase 5.B (non-causal) seed 42 boundary ratios for direct comparison."""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

# Load Phase 5.B seed 42 checkpoint
from training.adaptive_optimizer import load_phase3_pi_fno_model
from phase_5_corrected.model_v2 import SCFNOModelV2

model_p3, _ = load_phase3_pi_fno_model("models/sc_fno/phase_3/pi_fno/best_val.pt", device=device)
raw_fno = model_p3.fno.fno if hasattr(model_p3.fno, "fno") else model_p3.fno
model = SCFNOModelV2(raw_fno, model_p3.decoder, pad_size=0).to(device)

ckpt = torch.load("models/sc_fno/phase_5/step_5_B/seed_42_best.pt",
                  map_location=device, weights_only=False)
sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"Loaded Phase 5.B seed 42: missing={len(missing)}, unexpected={len(unexpected)}")
model.eval()

# Load val data
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
va = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
X_va = np.zeros((len(va), 200), dtype=np.float32)
Y_va = np.zeros((len(va), 9, 200), dtype=np.float32)
for i, sid in enumerate(va):
    d = RAW / sid
    f = np.load(d/"forcing.npy")
    t = np.load(d/"trajectory.npy")
    X_va[i] = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
    for j, k in enumerate(STATES):
        Y_va[i, j] = (t[:, j] - norm[k]["mean"]) / norm[k]["std"]

X_va_t = torch.tensor(X_va)
Y_va_t = torch.tensor(Y_va)

START = slice(0, 20)
INTERIOR = slice(20, 180)
END = slice(180, 200)

mae_start = {k: 0.0 for k in STATES}
mae_interior = {k: 0.0 for k in STATES}
mae_end = {k: 0.0 for k in STATES}
n = 0

with torch.no_grad():
    for vi in range(0, X_va_t.shape[0], 16):
        v_end = min(vi + 16, X_va_t.shape[0])
        xb = X_va_t[vi:v_end].unsqueeze(1).to(device)
        yb = Y_va_t[vi:v_end].to(device)
        pred = model.forward_data(xb)
        diff = (pred - yb).abs()
        for sj, k in enumerate(STATES):
            mae_start[k] += diff[:, sj, START].mean().item() * (v_end - vi)
            mae_interior[k] += diff[:, sj, INTERIOR].mean().item() * (v_end - vi)
            mae_end[k] += diff[:, sj, END].mean().item() * (v_end - vi)
        n += v_end - vi

print("\n" + "=" * 70)
print("Phase 5.B (non-causal) seed 42 — boundary ratios")
print("=" * 70)
boundary = {}
for k in STATES:
    s = mae_start[k] / max(n, 1)
    i = mae_interior[k] / max(n, 1)
    e = mae_end[k] / max(n, 1)
    boundary[k] = {
        "mae_start": s, "mae_interior": i, "mae_end": e,
        "start_interior": s / i if i > 1e-30 else None,
        "end_interior": e / i if i > 1e-30 else None,
    }
    print(f"  {k:3s}: start/interior = {boundary[k]['start_interior']:.2f}x")

out = {
    "model": "Phase 5.B non-causal seed 42",
    "boundary_ratios": boundary,
    "source_checkpoint": "models/sc_fno/phase_5/step_5_B/seed_42_best.pt",
}
json.dump(out, open("results/phase_5/step_5_C/boundary_analysis/phase5b_noncausal_baseline.json", "w"), indent=2)
print(f"\nSaved: phase5b_noncausal_baseline.json")
