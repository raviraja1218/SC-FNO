"""Step 3.5 — Shared evaluation utilities."""
import sys, json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from training.train_sc_fno import PaddedFNO
from physics_informed.decoder import build_decoder
from physics_informed.derivatives import load_phase2_fno, SCFNOModel

STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"


def load_norm():
    return json.load(open(
        "results/phase_1/splitting/normalization_statistics.json"
    ))["global"]


def load_test_ids():
    return pd.read_csv("results/phase_1/splitting/test_ids.csv")["scenario_id"].tolist()


def load_test_data():
    """Return dict: sid -> {'forcing', 'trajectory', 'time'} (all raw physical units)."""
    test_ids = load_test_ids()
    data = {}
    for sid in test_ids:
        d = RAW / sid
        data[sid] = {
            "forcing":    np.load(d / "forcing.npy"),       # [200]
            "trajectory": np.load(d / "trajectory.npy"),    # [200, 9]
            "time":       np.load(d / "time.npy"),          # [200]
        }
    return data


def denorm_states(u_norm, norm):
    """u_norm: [B, 9, T] normalized -> denormalized physical."""
    out = u_norm.clone()
    for j, s in enumerate(STATES):
        out[:, j] = out[:, j] * norm[s]["std"] + norm[s]["mean"]
    return out


def denorm_forcing(rho_norm, norm):
    return rho_norm * norm["rho_ext"]["std"] + norm["rho_ext"]["mean"]


def build_model_for_variant(variant, device="cpu"):
    """Load one of: p2, p_fno, pi_fno, p_pi_fno."""
    fno_raw = load_phase2_fno(device=device)
    if variant == "p2":
        # P2-FNO uses standard FNO forward (no padding, no decoder training)
        decoder = build_decoder(seed=42).to(device)
        model = SCFNOModel(fno_raw, decoder).to(device).eval()
        return model, 0, "p2"
    padding = 20 if variant in ("p_fno", "p_pi_fno") else 0
    fno = PaddedFNO(fno_raw, pad_size=padding).to(device)
    decoder = build_decoder(seed=42).to(device)
    model = SCFNOModel(fno, decoder).to(device).eval()
    ckpt_path = f"models/sc_fno/phase_3/{variant}/best_val.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
    model.load_state_dict(sd)
    model.eval()
    return model, padding, variant


def predict_data_path(model, padding, rho_norm, device):
    """Run through data path: fno(rho) -> projection -> 9 states at 200 grid."""
    B = rho_norm.shape[0]
    rho_t = torch.tensor(rho_norm, dtype=torch.float32).unsqueeze(1).to(device)  # [B,1,200]
    with torch.no_grad():
        u_pred = model.fno(rho_t) if padding == 0 else model.fno(rho_t)  # PaddedFNO crops internally
    return u_pred.cpu().numpy()  # [B, 9, 200] normalized


def predict_decoder_path(model, rho_norm, t_query, device):
    """Run through physics path: decoder(z, t_query)."""
    from physics_informed.derivatives import compute_du_dt
    B = rho_norm.shape[0]
    rho_t = torch.tensor(rho_norm, dtype=torch.float32).unsqueeze(1).to(device)
    t_q = torch.tensor(t_query, dtype=torch.float32).to(device)
    if t_q.dim() == 1:
        t_q = t_q.unsqueeze(0).expand(B, -1).contiguous()
    with torch.no_grad():
        z = model._fno_latent(rho_t)  # [B, 32, 200]
        u = model.decoder(z, t_q)     # [B, 9, Nq]
    return u.cpu().numpy(), z.cpu().numpy()


# Metric functions
def mae(a, b, axis=0):
    return np.mean(np.abs(a - b), axis=axis)

def rmse(a, b, axis=0):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))

def rel_l2(a, b):
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + 1e-12
    return num / den

def max_abs(a, b):
    return np.max(np.abs(a - b))


def compute_scenario_metrics(pred, gt):
    """pred, gt: [9, 200]. Returns dict per state + aggregate."""
    out = {"per_state": {}, "aggregate_rel_l2": rel_l2(pred.flatten(), gt.flatten())}
    for j, s in enumerate(STATES):
        out["per_state"][s] = {
            "mae":    float(mae(pred[j], gt[j])),
            "rmse":   float(rmse(pred[j], gt[j])),
            "rel_l2": float(rel_l2(pred[j], gt[j])),
            "max_abs": float(max_abs(pred[j], gt[j])),
        }
    return out


def compute_boundary_ratio(pred, gt, window=10):
    """pred, gt: [9, 200]. Start MAE / interior MAE per state."""
    out = {}
    for j, s in enumerate(STATES):
        start_mae = float(mae(pred[j, :window], gt[j, :window]))
        interior_mae = float(mae(pred[j, window:-window], gt[j, window:-window]))
        ratio = start_mae / (interior_mae + 1e-12)
        out[s] = {"start_mae": start_mae, "interior_mae": interior_mae, "ratio": ratio}
    return out


def get_family(sid):
    """Map scenario ID to forcing family from manifest."""
    # From Phase 1: first 500 step, next 500 ramp, next 500 sinusoidal, last 500 GRF
    n = int(sid.split("_")[1])
    if 1 <= n <= 500: return "step"
    if 501 <= n <= 1000: return "ramp"
    if 1001 <= n <= 1500: return "sinusoidal"
    return "grf"
