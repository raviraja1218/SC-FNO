"""Step 5.D.C — Resolution transfer (output at 200, interpolate to N)."""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
OUT = Path("results/phase_5/step_5_D/resolution")
OUT.mkdir(parents=True, exist_ok=True)
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
test_ids = pd.read_csv("results/phase_1/splitting/test_ids.csv")["scenario_id"].tolist()[:20]

from training.adaptive_optimizer import load_phase3_pi_fno_model
from phase_5_corrected.model_v2 import SCFNOModelV2

def load_5b():
    m3, _ = load_phase3_pi_fno_model("models/sc_fno/phase_3/pi_fno/best_val.pt", device=device)
    raw = m3.fno.fno if hasattr(m3.fno, "fno") else m3.fno
    m = SCFNOModelV2(raw, m3.decoder, pad_size=0).to(device)
    ck = torch.load("models/sc_fno/phase_5/step_5_B/seed_42_best.pt",
                    map_location=device, weights_only=False)
    sd = {k:v for k,v in ck["model_state_dict"].items() if k != "_metadata"}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def get_interp(traj, N_out):
    t_in = np.linspace(0, 10, traj.shape[0])
    t_out = np.linspace(0, 10, N_out)
    return np.stack([np.interp(t_out, t_in, traj[:, j]) for j in range(9)], axis=1)

def evaluate_resolution(model, N_out):
    rel_l2 = []
    for sid in test_ids:
        d = RAW / sid
        fn = np.load(d/"forcing.npy")
        tp = np.load(d/"trajectory.npy")
        fn_n = (fn - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        xb = torch.tensor(fn_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_n = model.forward_data(xb).cpu().numpy()[0]
        pred_phys = np.stack([pred_n[j] * norm[k]["std"] + norm[k]["mean"]
                              for j, k in enumerate(STATES)], axis=1)
        pred_interp = get_interp(pred_phys, N_out)
        gt_interp = get_interp(tp, N_out)
        per_state = []
        for j in range(9):
            d_gt = np.linalg.norm(gt_interp[:, j])
            if d_gt > 1e-30:
                per_state.append(np.linalg.norm(gt_interp[:, j] - pred_interp[:, j]) / d_gt)
            else:
                per_state.append(np.linalg.norm(gt_interp[:, j] - pred_interp[:, j]))
        rel_l2.append(np.mean(per_state))
    return float(np.median(rel_l2)), float(np.mean(rel_l2))

def main():
    model = load_5b()
    results = {}
    for N in [200, 500, 1000, 5000, 10000]:
        med, mean = evaluate_resolution(model, N)
        results[N] = {"median": med, "mean": mean}
        print(f"  N={N:6d}: median = {med:.4e}, mean = {mean:.4e}")
    json.dump(results, open(OUT / "resolution_transfer.json", "w"), indent=2)
    print(f"Saved: {OUT}/resolution_transfer.json")

if __name__ == "__main__":
    main()
