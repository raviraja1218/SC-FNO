"""Step 4A.0.D — Reproduce Phase 3 PI-FNO validation loss on 200 val scenarios.

Phase 3 recorded: PI-FNO best val_data_loss = 0.066331.
If the checkpoint is intact and our eval matches Phase 3's protocol,
we should reproduce it within ~0.001.

Evaluation protocol (Phase 3, from train_sc_fno.py):
  - Data path: model.fno(rho_norm) -> [B, 9, 200]
  - Loss: MSE over all 9 states x 200 timesteps (mean over batch)
  - Both pred and gt normalized with Phase 1 training-only stats
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

EXPECTED_VAL_LOSS = 0.066331
TOLERANCE = 0.001
CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"

out = {"checks": {}, "all_pass": False, "measured": {}}
def check(name, ok, detail=""):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
fno = load_phase2_fno(device=device)
decoder = build_decoder(seed=42).to(device)
model = SCFNOModel(fno, decoder).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd_clean = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
model.load_state_dict(sd_clean, strict=False)
model.eval()
print(f"  Loaded PI-FNO (epoch={ckpt.get('epoch')}, "
      f"recorded val_data_loss={ckpt.get('val_data_loss')})")

# Load val scenarios
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
STATE_KEYS = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

def normalize_states(traj_phys):
    out = np.zeros_like(traj_phys)
    for j, k in enumerate(STATE_KEYS):
        out[:, j] = (traj_phys[:, j] - norm[k]["mean"]) / norm[k]["std"]
    return out

BATCH = 32
per_scenario_mse = []

with torch.no_grad():
    for i in range(0, len(val_ids), BATCH):
        batch_ids = val_ids[i:i+BATCH]
        forcing_batch = []
        traj_batch = []
        for sid in batch_ids:
            d = RAW / sid
            forcing = np.load(d / "forcing.npy")
            traj = np.load(d / "trajectory.npy")
            forcing_norm = (forcing - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
            traj_norm = normalize_states(traj)     # [200, 9]
            forcing_batch.append(forcing_norm)
            traj_batch.append(traj_norm.T)         # [9, 200]

        x = torch.tensor(np.stack(forcing_batch), dtype=torch.float32).unsqueeze(1).to(device)
        y = torch.tensor(np.stack(traj_batch), dtype=torch.float32).to(device)

        pred = model.fno(x)                        # [B, 9, 200]
        se = (pred - y) ** 2
        mse_per_scenario = se.mean(dim=[1, 2]).cpu().numpy()
        per_scenario_mse.extend(mse_per_scenario.tolist())

mean_val_mse = float(np.mean(per_scenario_mse))
out["measured"]["mean_val_data_mse"] = mean_val_mse
out["measured"]["std_val_data_mse"] = float(np.std(per_scenario_mse))
out["measured"]["min_val_data_mse"] = float(np.min(per_scenario_mse))
out["measured"]["max_val_data_mse"] = float(np.max(per_scenario_mse))
out["measured"]["expected_val_data_mse"] = EXPECTED_VAL_LOSS
out["measured"]["delta"] = mean_val_mse - EXPECTED_VAL_LOSS

check("val_mse_reproduces_phase3",
      abs(mean_val_mse - EXPECTED_VAL_LOSS) < TOLERANCE,
      f"{mean_val_mse:.6f} vs expected {EXPECTED_VAL_LOSS:.6f} "
      f"(Δ={mean_val_mse - EXPECTED_VAL_LOSS:+.6f}, tol={TOLERANCE})")

out["all_pass"] = all(c["pass"] for c in out["checks"].values())

with open("results/phase_4/phase_4a/preflight/val_metrics_reproduced.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out["measured"], indent=2))
print(f"\n=== 4A.0.D Reproduce val metrics: {'PASS' if out['all_pass'] else 'FAIL'} ===")
