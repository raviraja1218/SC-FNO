"""Step 3.1.D — Measure residual characteristic scales from training data.

Method: Load ground-truth trajectories, compute finite-difference derivatives,
then compute residuals with u_hat = ground truth, du_hat_dt = finite-diff.
This gives realistic magnitudes for residual normalization in Step 3.3.

NOTE: This uses ground-truth as u_hat for scale measurement ONLY. It is NOT
the training residual — training residual uses predicted u_hat. This script
measures SCALE, not training loss.
"""
import sys
sys.path.insert(0, "src")

import dataclasses
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.residuals import compute_residuals

# --- Load parameters ---
def _build_kin():
    fields = {f.name for f in dataclasses.fields(KineticsParameters)}
    beta_vec = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
    lam_vec = np.array([0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100])
    kw = {"beta_eff": 0.006502, "Lambda": 1.0e-4,
          "beta_i": beta_vec, "lambda_i": lam_vec}
    return KineticsParameters(**{k: v for k, v in kw.items() if k in fields})

def _build_th():
    fields = {f.name for f in dataclasses.fields(THParameters)}
    canonical = {
        "M_f": 27000.0, "c_p_f": 300.0, "f_f": 0.97,
        "M_c": 4000.0, "c_p_c": 5500.0, "m_dot": 13636.36, "T_in": 563.0,
        "U": 3060.0, "A": 3000.0, "P0": 3.0e9,
        "alpha_f": -3.0e-5, "alpha_c": -5.0e-4,
        "T_f_ref": 900.0, "T_c_ref": 583.0,
    }
    return THParameters(**{k: v for k, v in canonical.items() if k in fields})

kin = _build_kin()
th = _build_th()

# --- Sample 100 training scenarios ---
train_ids = pd.read_csv("results/phase_1/splitting/train_ids.csv")["scenario_id"].tolist()
import random
random.seed(42)
sample = random.sample(train_ids, 100)

RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"

# Accumulators
per_state_abs = {name: [] for name in
                 ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]}

for sid in sample:
    traj = np.load(RAW / sid / "trajectory.npy")   # [200, 9]
    t = np.load(RAW / sid / "time.npy")             # [200]
    forcing = np.load(RAW / sid / "forcing.npy")    # [200]

    u_np = traj.T[None, :, :]                        # [1, 9, 200]
    u = torch.tensor(u_np, dtype=torch.float32)

    # Finite-difference derivative (central diff)
    du_np = np.gradient(u_np, t, axis=2)             # [1, 9, 200]
    du = torch.tensor(du_np, dtype=torch.float32)

    t_tensor = torch.tensor(t[None, :], dtype=torch.float32)
    # rho_ext interpolated on t
    def rho_fn(tq, _force=forcing, _tgrid=t):
        # tq [1, 200]; force [200]
        return torch.tensor(np.interp(tq[0].numpy(), _tgrid, _force)[None, :],
                            dtype=torch.float32)

    out = compute_residuals(u, du, t_tensor, rho_fn, kin, th)

    R_n  = out.R_n.abs().numpy().flatten()
    R_C  = out.R_C.abs().numpy().transpose(0, 2, 1)   # [B, N, 6]
    R_Tf = out.R_Tf.abs().numpy().flatten()
    R_Tc = out.R_Tc.abs().numpy().flatten()

    per_state_abs["n"].append(R_n)
    for i, name in enumerate(["C1", "C2", "C3", "C4", "C5", "C6"]):
        per_state_abs[name].append(R_C[:, :, i].flatten())
    per_state_abs["Tf"].append(R_Tf)
    per_state_abs["Tc"].append(R_Tc)

# --- Aggregate statistics ---
summary = {}
for name, chunks in per_state_abs.items():
    if not chunks:
        continue
    allv = np.concatenate(chunks)
    summary[name] = {
        "max": float(allv.max()),
        "mean": float(allv.mean()),
        "p50": float(np.percentile(allv, 50)),
        "p95": float(np.percentile(allv, 95)),
        "p99": float(np.percentile(allv, 99)),
        "n_samples": int(allv.size),
        "characteristic_scale": float(np.percentile(allv, 95)),
    }

out = {
    "method": "finite-difference derivatives on ground-truth trajectories",
    "purpose": "characteristic residual scales for Step 3.3 loss normalization",
    "n_scenarios_sampled": 100,
    "n_samples_per_state": int(100 * 200),
    "per_state": summary,
    "note": "Uses ground truth for SCALE measurement only, not for training. "
            "Step 3.3 will use these scales to normalize physics loss contributions.",
}
with open("results/phase_3/residuals/residual_scales.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
print("\n=== Residual scale measurement: DONE ===")
