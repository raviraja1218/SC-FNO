"""Step 5.1 — Validation-only comparison. NO test set use."""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]

OUT = Path("results/phase_5/step_5_1/evaluation")
OUT.mkdir(parents=True, exist_ok=True)

# Load val scenarios
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()

X_va = np.zeros((len(val_ids), 200), dtype=np.float32)
Y_va = np.zeros((len(val_ids), 9, 200), dtype=np.float32)
Y_va_phys = np.zeros((len(val_ids), 9, 200), dtype=np.float64)
for i, sid in enumerate(val_ids):
    d = RAW / sid
    f = np.load(d / "forcing.npy"); t = np.load(d / "trajectory.npy")
    X_va[i] = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
    for j, k in enumerate(STATES):
        Y_va[i, j] = (t[:, j] - norm[k]["mean"]) / norm[k]["std"]
        Y_va_phys[i, j] = t[:, j]

# Load 4 variants' best checkpoints
from physics_informed.derivatives import load_phase2_fno, SCFNOModel
from physics_informed.decoder import build_decoder
from physics_informed.losses import load_composite_loss
from models.boundary_aware_fno import UnifiedLatentFNO

VARIANTS = {
    "P5-A": {"pcgrad": True, "unified_padding": False, "adaptive_lambda": False},
    "P5-B": {"pcgrad": False, "unified_padding": True, "adaptive_lambda": False},
    "P5-C": {"pcgrad": False, "unified_padding": False, "adaptive_lambda": True},
    "P5-D": {"pcgrad": True, "unified_padding": True, "adaptive_lambda": True},
}

loss_fn = load_composite_loss()

def load_variant(vname, cfg):
    ckpt_path = f"models/sc_fno/phase_5/step_5_1/{vname.lower()}_best.pt"
    if not Path(ckpt_path).exists():
        print(f"[SKIP] {vname}: checkpoint not found")
        return None
    fno = load_phase2_fno(device=device, checkpoint_path="models/baseline/fno_best.pt")
    if cfg["unified_padding"]:
        fno = UnifiedLatentFNO(fno, pad_size=20)
    decoder = build_decoder(seed=42).to(device)
    model = SCFNOModel(fno, decoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def eval_val(model, cfg):
    """Per-scenario rel L2 on validation."""
    B = 16
    n = len(val_ids)
    per_scn = np.zeros((n, 9), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, B):
            j = min(i+B, n)
            xb = torch.tensor(X_va[i:j], dtype=torch.float32).unsqueeze(1).to(device)
            if cfg["unified_padding"]:
                pred = model.fno(xb)
            else:
                pred = model.fno(xb)
            pred_np = pred.cpu().numpy()
            for bi in range(j - i):
                for sj, k in enumerate(STATES):
                    yp = Y_va_phys[i+bi, sj]
                    pp = pred_np[bi, sj] * norm[k]["std"] + norm[k]["mean"]
                    d = np.linalg.norm(yp)
                    per_scn[i+bi, sj] = np.linalg.norm(yp - pp) / d if d > 1e-30 else np.linalg.norm(yp - pp)
    return per_scn

results = {}
for vname, cfg in VARIANTS.items():
    print(f"Evaluating {vname}...")
    model = load_variant(vname, cfg)
    if model is None:
        continue
    per_scn = eval_val(model, cfg)
    per_scn_mean = per_scn.mean(axis=1)
    results[vname] = {
        "variant": vname,
        "val_rel_l2_median": float(np.median(per_scn_mean)),
        "val_rel_l2_mean": float(np.mean(per_scn_mean)),
        "per_state_median": {k: float(np.median(per_scn[:, sj])) for sj, k in enumerate(STATES)},
        "per_state_mean": {k: float(np.mean(per_scn[:, sj])) for sj, k in enumerate(STATES)},
    }

# Summary
df = pd.DataFrame([
    {"variant": r["variant"],
     "val_rel_l2_median": r["val_rel_l2_median"],
     "val_rel_l2_mean": r["val_rel_l2_mean"]}
    for r in results.values()
])
df.to_csv(OUT / "validation_comparison.csv", index=False)
json.dump(results, open(OUT / "validation_comparison_full.json", "w"), indent=2)

print("\n=== Validation Comparison (200 val scenarios, NO test set) ===")
print(df.to_string(index=False))
