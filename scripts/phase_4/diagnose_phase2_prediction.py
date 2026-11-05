import sys
sys.path.insert(0, "src")

import json
import numpy as np
import torch
from pathlib import Path

from physics_informed.derivatives import load_phase2_fno

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SID = "scenario_0458"

RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"

NPZ_PATH = Path(
    "results/phase_2/evaluation/predictions_summary.npz"
)

NORM_PATH = Path(
    "results/phase_1/splitting/normalization_statistics.json"
)

TEST_IDS_PATH = Path(
    "results/phase_1/splitting/test_ids.csv"
)

STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

def report_array(name, arr):
    print(
        f"{name}: shape={arr.shape}, "
        f"dtype={arr.dtype}, std={arr.std():.8f}"
    )

# --------------------------------------------------
# Load Phase 2 saved predictions
# --------------------------------------------------
p2 = np.load(NPZ_PATH, allow_pickle=True)

print("\n========== PHASE 2 NPZ ==========")
print("Keys:", list(p2.keys()))

saved_sids = p2["sids"].astype(str).tolist()

if SID not in saved_sids:
    raise ValueError(f"{SID} not found in saved NPZ IDs")

idx_saved = saved_sids.index(SID)

print("Scenario:", SID)
print("Saved NPZ index:", idx_saved)

saved_states = p2["state_order"].astype(str).tolist()

print("Saved state order:", saved_states)

if set(saved_states) != set(STATES):
    raise ValueError("State names do not match expected states")

state_indices = [saved_states.index(s) for s in STATES]

# --------------------------------------------------
# Compare test_ids index
# --------------------------------------------------
import pandas as pd

test_ids = pd.read_csv(TEST_IDS_PATH)["scenario_id"].astype(str).tolist()

if SID not in test_ids:
    raise ValueError(f"{SID} not found in test_ids.csv")

idx_test = test_ids.index(SID)

print("test_ids.csv index:", idx_test)
print("Index agreement:", idx_saved == idx_test)

# --------------------------------------------------
# Load raw scenario and normalize
# --------------------------------------------------
scenario_dir = RAW / SID

forcing = np.load(scenario_dir / "forcing.npy")
trajectory = np.load(scenario_dir / "trajectory.npy")

norm = json.load(open(NORM_PATH))["global"]

forcing_mean = norm["rho_ext"]["mean"]
forcing_std = norm["rho_ext"]["std"]

forcing_norm = (forcing - forcing_mean) / forcing_std

trajectory_norm = np.zeros_like(trajectory, dtype=np.float64)

for j, state in enumerate(STATES):
    trajectory_norm[:, j] = (
        trajectory[:, j] - norm[state]["mean"]
    ) / norm[state]["std"]

# --------------------------------------------------
# Fresh Phase 2 FNO prediction
# --------------------------------------------------
x = torch.tensor(
    forcing_norm,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(DEVICE)

y = torch.tensor(
    trajectory_norm.T,
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)

fno = load_phase2_fno(device=DEVICE)
fno.eval()

with torch.no_grad():
    pred_fresh = fno(x)

fresh_pred = pred_fresh[0].detach().cpu().numpy()
computed_true = y[0].detach().cpu().numpy()

# --------------------------------------------------
# Extract saved predictions and targets
# --------------------------------------------------
saved_pred = p2["pred_norm"][idx_saved][state_indices]
saved_true = p2["true_norm"][idx_saved][state_indices]

saved_pred_phys = p2["pred_phys"][idx_saved][state_indices]
saved_true_phys = p2["true_phys"][idx_saved][state_indices]

# --------------------------------------------------
# Report
# --------------------------------------------------
print("\n========== SHAPES ==========")
report_array("Fresh prediction", fresh_pred)
report_array("Saved prediction", saved_pred)
report_array("Computed target", computed_true)
report_array("Saved target", saved_true)

print("\n========== PREDICTION COMPARISON ==========")
print("Max absolute difference:",
      np.max(np.abs(fresh_pred - saved_pred)))
print("Relative difference:",
      rel_l2(fresh_pred, saved_pred))

print("\n========== TARGET COMPARISON ==========")
print("Max absolute difference:",
      np.max(np.abs(computed_true - saved_true)))
print("Relative difference:",
      rel_l2(computed_true, saved_true))

print("\n========== RELATIVE L2 ==========")
print("Fresh prediction vs computed target:",
      rel_l2(fresh_pred, computed_true))

print("Saved prediction vs saved target:",
      rel_l2(saved_pred, saved_true))

print("Fresh prediction vs saved target:",
      rel_l2(fresh_pred, saved_true))

print("Saved prediction vs computed target:",
      rel_l2(saved_pred, computed_true))

print("\n========== PER-STATE TARGET COMPARISON ==========")
for j, state in enumerate(STATES):
    print(
        f"{state:>3}: "
        f"computed_std={computed_true[j].std():.8f}, "
        f"saved_std={saved_true[j].std():.8f}, "
        f"max_diff={np.max(np.abs(computed_true[j] - saved_true[j])):.8e}"
    )

print("\n========== PHYSICAL SPACE ==========")
print("Saved physical aggregate RelL2:",
      rel_l2(saved_pred_phys, saved_true_phys))

for j, state in enumerate(STATES):
    print(
        f"{state:>3}: "
        f"{rel_l2(saved_pred_phys[j], saved_true_phys[j]):.8e}"
    )

print("\n========== INPUT ==========")
print("Forcing shape:", forcing.shape)
print("Forcing normalized std:", forcing_norm.std())
print("Forcing unique values:", np.unique(forcing_norm).size)
print("Forcing first value:", forcing_norm.flat[0])

print("\nDIAGNOSTIC COMPLETE")
