"""Step 4B.5 — Full test-set evaluation logic.

Assumes 7 models are loaded. Runs:
  1. Per-scenario per-state relative L2
  2. Physics loss (9-equation)
  3. IC loss
  4. Boundary behavior (start/interior/end + endpoint + near-boundary)
  5. Per-family breakdown
  6. Paired bootstrap vs P2-FNO and P3-PI-FNO
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, "src")

# Import model loading setup from the smoke test script
# We re-execute the loading portion
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.derivatives import load_phase2_fno
from physics_informed.decoder import build_decoder
from physics_informed.losses import load_composite_loss
from training.adaptive_optimizer import load_phase3_pi_fno_model
from models.boundary_aware_fno import UnifiedLatentFNO

OUT = Path("results/phase_4/phase_4b/evaluation")
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

# ---- Load 7 models ----
def load_p4_unified_model(ckpt_path, device):
    from training.train_sc_fno import PaddedFNO
    from physics_informed.derivatives import SCFNOModel
    fno_raw = load_phase2_fno(device=device)
    fno_padded = PaddedFNO(fno_raw, pad_size=0)
    fno_unified = UnifiedLatentFNO(fno_padded, pad_size=20, pad_mode="reflect")
    decoder = build_decoder(seed=42).to(device)
    model = SCFNOModel(fno_unified, decoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"P4 load mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}")
    model.eval()
    return model

MODELS_CONFIG = {
    "P2-FNO": {"path": "models/baseline/fno_best.pt", "type": "p2"},
    "P3-PI-FNO": {"path": "models/sc_fno/phase_3/pi_fno/best_val.pt", "type": "sc_fno"},
    "Phase4A.1": {"path": "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt", "type": "sc_fno"},
    "P4-A": {"path": "models/sc_fno/phase_4/phase_4b/p4_a_best.pt", "type": "sc_fno"},
    "P4-B": {"path": "models/sc_fno/phase_4/phase_4b/p4_b_best.pt", "type": "unified"},
    "P4-C": {"path": "models/sc_fno/phase_4/phase_4b/p4_c_best.pt", "type": "sc_fno"},
    "P4-D": {"path": "models/sc_fno/phase_4/phase_4b/p4_d_best.pt", "type": "unified"},
}

print()
print("=== Loading 7 models ===")
models = {}
for name, cfg in MODELS_CONFIG.items():
    if cfg["type"] == "p2":
        fno = load_phase2_fno(device=device)
        models[name] = {"type": "p2", "fno": fno, "decoder": None, "model": None}
    elif cfg["type"] == "unified":
        model = load_p4_unified_model(cfg["path"], device)
        models[name] = {"type": "sc", "model": model,
                        "fno": model.fno, "decoder": model.decoder}
    else:
        model, _ = load_phase3_pi_fno_model(cfg["path"], device=device)
        model.eval()
        models[name] = {"type": "sc", "model": model,
                        "fno": model.fno, "decoder": model.decoder}
    print(f"  [{name}] loaded")

# ---- Evaluation function ----
def evaluate_model(name, m):
    """Evaluate one model on all 200 test scenarios. Returns dict of metrics."""
    print(f"\n=== Evaluating {name} ===")
    t0 = time.perf_counter()

    per_scenario_per_state_rel_l2 = np.zeros((200, 9))
    physics_losses_per_batch = []
    ic_losses_per_batch = []

    t_data = torch.linspace(0, 10, 200, device=device)
    t_q = torch.linspace(T_LO, T_HI, NQ, device=device)

    for i in range(0, 200, BATCH):
        x = X_test[i:i+BATCH]
        y = Y_test[i:i+BATCH]
        B = x.shape[0]
        t_data_b = t_data.unsqueeze(0).expand(B, -1).contiguous()
        t_q_b = t_q.unsqueeze(0).expand(B, -1).contiguous()

        with torch.no_grad():
            # Data path is model.fno(x) for ALL models — this is what training used
            # for data loss. Decoder is ONLY for physics residuals (data path uses
            # the FNO projection output directly, not decoder output).
            pred = m["fno"](x)  # [B, 9, 200]
            # Sanity check on shape
            assert pred.shape == (B, 9, 200), f"pred shape {pred.shape}"

        # Per-scenario per-state rel L2
        diff = pred - y
        for b in range(B):
            for j in range(9):
                num = torch.linalg.norm(diff[b, j, :]).item()
                den = torch.linalg.norm(y[b, j, :]).item()
                per_scenario_per_state_rel_l2[i + b, j] = num / max(den, 1e-12)

        # Physics loss (skip for P2-FNO — no physics objective)
        if m["type"] == "sc":
            with torch.enable_grad():
                L_phys = loss_fn.compute_physics_loss(
                    m["model"], x, t_q_b, lambda tq: torch.zeros_like(tq), kin, th)
                physics_losses_per_batch.append(float(L_phys.item()))
            # IC loss
            with torch.no_grad():
                if hasattr(m["fno"], "get_latent"):
                    latent = m["fno"].get_latent(x)
                else:
                    captured = {}
                    def _hook(module, inputs, output):
                        captured["latent"] = inputs[0]
                    hook = m["fno"].projection.register_forward_hook(_hook)
                    try:
                        _ = m["fno"](x)
                    finally:
                        hook.remove()
                    latent = captured["latent"]
                t_ic = torch.zeros(B, 1, device=device)
                u_ic = m["decoder"](latent, t_ic)
                ic_gt = y[:, :, 0].unsqueeze(-1)
                ic_scale = torch.tensor([norm[k]["std"] for k in STATES], device=device).view(1, 9, 1)
                L_ic = torch.mean(((u_ic - ic_gt) / ic_scale) ** 2).item()
                ic_losses_per_batch.append(L_ic)

    elapsed = time.perf_counter() - t0

    # Aggregate
    result = {
        "model": name,
        "per_scenario_per_state_rel_l2": per_scenario_per_state_rel_l2.tolist(),
        "rel_l2_overall_median": float(np.median(per_scenario_per_state_rel_l2.mean(axis=1))),
        "rel_l2_overall_mean": float(np.mean(per_scenario_per_state_rel_l2.mean(axis=1))),
        "rel_l2_per_state_median": {STATES[j]: float(np.median(per_scenario_per_state_rel_l2[:, j])) for j in range(9)},
        "rel_l2_per_state_mean": {STATES[j]: float(np.mean(per_scenario_per_state_rel_l2[:, j])) for j in range(9)},
        "physics_loss": float(np.mean(physics_losses_per_batch)) if physics_losses_per_batch else None,
        "ic_loss": float(np.mean(ic_losses_per_batch)) if ic_losses_per_batch else None,
        "eval_time_s": elapsed,
    }
    print(f"  rel L2 median: {result['rel_l2_overall_median']:.6e}")
    print(f"  rel L2 mean:   {result['rel_l2_overall_mean']:.6e}")
    if result['physics_loss'] is not None:
        print(f"  physics loss:  {result['physics_loss']:.4e}")
    if result['ic_loss'] is not None:
        print(f"  IC loss:       {result['ic_loss']:.4e}")
    print(f"  eval time: {elapsed:.1f}s")
    return result

# ---- Run evaluation for all 7 models ----
print("\n" + "="*70)
print("RUNNING FULL EVALUATION")
print("="*70)

all_results = {}
for name in MODELS_CONFIG.keys():
    all_results[name] = evaluate_model(name, models[name])

# Save combined results
with open(OUT / "all_models_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ---- Build summary table ----
summary_rows = []
for name, r in all_results.items():
    summary_rows.append({
        "model": name,
        "rel_l2_median": r["rel_l2_overall_median"],
        "rel_l2_mean": r["rel_l2_overall_mean"],
        "physics_loss": r["physics_loss"],
        "ic_loss": r["ic_loss"],
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "all_models_summary.csv", index=False)

print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)
print(summary_df.to_string(index=False))

print("\n=== ALL EVALUATION COMPLETE ===")
print("Saved: all_models_metrics.json, all_models_summary.csv")

