"""Step 4B.5 — Test-Set Evaluation of 7 Models.

Evaluates P2-FNO, P3-PI-FNO, Phase 4A.1, P4-A, P4-B, P4-C, P4-D on 200
untouched test scenarios. Computes per-state per-scenario rel L2, physics loss
(9-equation), IC loss, boundary ratios, endpoint errors, per-family breakdown,
paired bootstrap vs P2-FNO and P3-PI-FNO.

Read-only. No training. No model checkpoint modification.
"""
import json
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, "src")

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.derivatives import load_phase2_fno
from physics_informed.decoder import build_decoder
from physics_informed.losses import load_composite_loss
from training.adaptive_optimizer import load_phase3_pi_fno_model
from models.boundary_aware_fno import UnifiedLatentFNO

# ---- Config ----
OUT = Path("results/phase_4/phase_4b/evaluation")
OUT.mkdir(parents=True, exist_ok=True)
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
BATCH = 32
NQ = 50
T_LO, T_HI = 0.5, 9.5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- Load configs ----
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
def _s(v): return float(v[0]) if isinstance(v, list) else float(v)
n_ = kin_cfg["neutronics"]
kin = KineticsParameters(
    beta_eff=_s(n_["beta_eff"]), Lambda=_s(n_["Lambda"]),
    beta_i=[_s(g["beta_i"]) for g in n_["precursor_groups"]],
    lambda_i=[_s(g["lambda_i"]) for g in n_["precursor_groups"]],
)
fb = th_cfg["reactivity_feedback"]
th = THParameters(
    M_f=_s(th_cfg["fuel"]["M_f"]), c_p_f=_s(th_cfg["fuel"]["c_p_f"]),
    f_f=_s(th_cfg["fuel"]["f_f"]), M_c=_s(th_cfg["coolant"]["M_c"]),
    c_p_c=_s(th_cfg["coolant"]["c_p_c"]), m_dot=_s(th_cfg["coolant"]["m_dot"]),
    T_in=_s(th_cfg["coolant"]["T_in"]), U=_s(th_cfg["heat_transfer"]["U"]),
    A=_s(th_cfg["heat_transfer"]["A"]), P0=_s(th_cfg["power"]["P0"]),
    alpha_f=_s(fb["alpha_f"]), alpha_c=_s(fb["alpha_c"]),
    T_f_ref=_s(fb["T_f_ref"]), T_c_ref=_s(fb["T_c_ref"]),
)
loss_fn = load_composite_loss()
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]

# ---- Load test data ----
test_ids = pd.read_csv("results/phase_1/splitting/test_ids.csv")["scenario_id"].tolist()
assert len(test_ids) == 200
print("Test scenarios:", len(test_ids))

def _norm_states(t):
    o = np.zeros_like(t)
    for j,k in enumerate(STATES):
        o[:,j] = (t[:,j] - norm[k]["mean"]) / norm[k]["std"]
    return o

family_map = {}
X_all, Y_all = [], []
for sid in test_ids:
    d = RAW / sid
    f = np.load(d / "forcing.npy")
    tj = np.load(d / "trajectory.npy")
    X_all.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
    Y_all.append(_norm_states(tj).T)
    meta = json.load(open(d / "metadata.json"))
    family_map[sid] = meta.get("forcing_family", meta.get("family", "unknown"))

X_test = torch.tensor(np.stack(X_all), dtype=torch.float32).unsqueeze(1).to(device)
Y_test = torch.tensor(np.stack(Y_all), dtype=torch.float32).to(device)
print("X_test:", X_test.shape, "Y_test:", Y_test.shape)
print("Families:", dict(pd.Series(list(family_map.values())).value_counts()))

# ---- Model loading ----
MODELS_CONFIG = {
    "P2-FNO": {"path": "models/baseline/fno_best.pt", "wrapper": "raw", "is_p2": True},
    "P3-PI-FNO": {"path": "models/sc_fno/phase_3/pi_fno/best_val.pt", "wrapper": "raw", "is_p2": False},
    "Phase4A.1": {"path": "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt", "wrapper": "raw", "is_p2": False},
    "P4-A": {"path": "models/sc_fno/phase_4/phase_4b/p4_a_best.pt", "wrapper": "raw", "is_p2": False},
    "P4-B": {"path": "models/sc_fno/phase_4/phase_4b/p4_b_best.pt", "wrapper": "unified", "is_p2": False},
    "P4-C": {"path": "models/sc_fno/phase_4/phase_4b/p4_c_best.pt", "wrapper": "raw", "is_p2": False},
    "P4-D": {"path": "models/sc_fno/phase_4/phase_4b/p4_d_best.pt", "wrapper": "unified", "is_p2": False},
}

def load_p4_unified_model(ckpt_path, device):
    """Load P4-B or P4-D checkpoint (triple-wrapped FNO)."""
    from training.train_sc_fno import PaddedFNO
    from physics_informed.derivatives import SCFNOModel, load_phase2_fno
    from physics_informed.decoder import build_decoder

    fno_raw = load_phase2_fno(device=device)
    fno_padded = PaddedFNO(fno_raw, pad_size=0)
    fno_unified = UnifiedLatentFNO(fno_padded, pad_size=20, pad_mode="reflect")
    decoder = build_decoder(seed=42).to(device)
    model = SCFNOModel(fno_unified, decoder).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            f"P4 unified load mismatch for {ckpt_path}: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    model.eval()
    return model, ckpt


def load_model(name, cfg):
    if cfg["is_p2"]:
        fno = load_phase2_fno(device=device)
        return {"type": "p2_fno", "fno": fno, "decoder": None, "model": None}
    if cfg["wrapper"] == "unified":
        model_raw, ckpt = load_p4_unified_model(cfg["path"], device=device)
        return {"type": "sc_fno", "fno": model_raw.fno, "decoder": model_raw.decoder, "model": model_raw}
    else:
        model, ckpt = load_phase3_pi_fno_model(cfg["path"], device=device)
        model.eval()
        return {"type": "sc_fno", "fno": model.fno, "decoder": model.decoder, "model": model}

print()
print("=== Loading 7 models ===")
models = {}
for name, cfg in MODELS_CONFIG.items():
    t0 = time.perf_counter()
    models[name] = load_model(name, cfg)
    print("  [%s] loaded in %.2fs" % (name, time.perf_counter() - t0))

# ---- Smoke test ----
print()
print("=== Smoke test: forward pass on 2 test samples ===")
x_smoke = X_test[:2]
for name, m in models.items():
    with torch.no_grad():
        if m["type"] == "p2_fno":
            out = m["fno"](x_smoke)
            assert out.shape == (2, 9, 200)
            assert torch.isfinite(out).all()
        else:
            if hasattr(m["fno"], "get_latent"):
                latent = m["fno"].get_latent(x_smoke)
                assert latent.shape == (2, 32, 200)
                t_eval = torch.linspace(0, 10, 200, device=device).unsqueeze(0).expand(2, -1)
                out = m["decoder"](latent, t_eval)
                assert out.shape == (2, 9, 200)
                assert torch.isfinite(out).all()
            else:
                out = m["fno"](x_smoke)
                assert out.shape == (2, 9, 200)
                assert torch.isfinite(out).all()
    print("  [%s] forward OK" % name)

print()
print("=== ALL MODELS LOADED AND SMOKE-TESTED ===")
