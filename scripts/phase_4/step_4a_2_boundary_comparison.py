"""Step 4A.2 — Boundary behavior comparison."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import torch
from training.adaptive_optimizer import load_phase3_pi_fno_model

PHASE3_CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
PHASE4A1_CKPT = "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
OUT = Path("results/phase_4/phase_4a/residual_analysis")
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
BATCH = 16
START_W = 20
INT_START = 20
INT_END = 180
END_W = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]

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

def measure_boundary(model, name):
    model.eval()
    acc = {s: {"start": [], "interior": [], "end": []} for s in STATES}
    with torch.no_grad():
        for i in range(0, len(val_ids), BATCH):
            x = X_val[i:i+BATCH]; y = Y_val[i:i+BATCH]
            pred = model.fno(x)
            abs_err = (pred - y).abs()
            for j, s in enumerate(STATES):
                acc[s]["start"].append(abs_err[:, j, :START_W].mean().item())
                acc[s]["interior"].append(abs_err[:, j, INT_START:INT_END].mean().item())
                acc[s]["end"].append(abs_err[:, j, -END_W:].mean().item())
    result = {"name": name, "states": {}}
    for s in STATES:
        st = float(np.mean(acc[s]["start"]))
        it = float(np.mean(acc[s]["interior"]))
        en = float(np.mean(acc[s]["end"]))
        result["states"][s] = {
            "start_mae": st, "interior_mae": it, "end_mae": en,
            "start_interior_ratio": st / max(it, 1e-12),
            "end_interior_ratio": en / max(it, 1e-12),
        }
    return result

def _safe_json(o):
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (int, np.integer)): return int(o)
    if isinstance(o, (float, np.floating)): return float(o)
    if isinstance(o, dict): return {str(k): _safe_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_safe_json(v) for v in o]
    return o

print("Loading Phase 3 PI-FNO...")
m3, _ = load_phase3_pi_fno_model(PHASE3_CKPT, device=device)
p3 = measure_boundary(m3, "phase3_pi_fno")

print("Loading Phase 4A.1 final...")
m4, _ = load_phase3_pi_fno_model(PHASE4A1_CKPT, device=device)
p4 = measure_boundary(m4, "phase4a1_final")

comparison = {"phase3": p3, "phase4a1": p4, "delta_ratio": {}}
for s in STATES:
    r3 = p3["states"][s]["start_interior_ratio"]
    r4 = p4["states"][s]["start_interior_ratio"]
    comparison["delta_ratio"][s] = {
        "phase3": r3, "phase4a1": r4,
        "delta": r4 - r3,
        "pct": (r4 - r3) / max(r3, 1e-12) * 100,
    }

with open(OUT / "boundary_comparison.json", "w") as f:
    json.dump(_safe_json(comparison), f, indent=2)

print("\n=== Boundary ratios (start/interior) ===")
for s in STATES:
    print(f"  {s:4s}: P3={p3['states'][s]['start_interior_ratio']:.3f}, "
          f"P4A.1={p4['states'][s]['start_interior_ratio']:.3f}, "
          f"Δ={p4['states'][s]['start_interior_ratio']-p3['states'][s]['start_interior_ratio']:+.3f}")
