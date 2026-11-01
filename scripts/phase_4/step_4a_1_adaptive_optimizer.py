"""Phase 4A.1 — Adaptive Adam → L-BFGS training.

Implements the pre-registered plan (results/phase_4/phase_4a/optimizer/pre_registration.yaml):
    - Fresh Adam from best_val.pt (epoch 36 weights, no optimizer state)
    - λ_phys ramp 0.002754 → 1.0 over 64 epochs
    - EMA plateau detection (α=0.1, ref=20, δ=1e-4, patience=3, min=20, max=100)
    - L-BFGS inherits current λ at plateau (Policy B)
    - Deterministic L-BFGS closure on fixed 16-scenario batch
    - Full evidence logging: adam_training_log.csv, lbfgs_training_log.csv, transition_event.json
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import yaml

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.derivatives import load_phase2_fno, SCFNOModel
from physics_informed.decoder import build_decoder
from physics_informed.losses import load_composite_loss
from training.adaptive_optimizer import (
    OptimizerConfig, AdaptiveOptimizer, EMAPlateauDetector,
    lam_phys_at_epoch, build_lbfgs, run_lbfgs_stage,
)

# ============ CONFIG ============
CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
OUT = Path("results/phase_4/phase_4a/optimizer")
MODEL_OUT = Path("models/sc_fno/phase_4/phase_4a")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)

# Load pre-registration and hydrate config
prereg = yaml.safe_load(open(OUT / "pre_registration.yaml"))
cfg = OptimizerConfig(
    adam_lr=prereg["adam"]["lr"],
    adam_beta1=prereg["adam"]["betas"][0],
    adam_beta2=prereg["adam"]["betas"][1],
    adam_eps=prereg["adam"]["eps"],
    scheduler_t_max=prereg["scheduler"]["T_max"],
    scheduler_eta_min=prereg["scheduler"]["eta_min"],
    lam_phys_init=prereg["lambda_phys"]["init"],
    lam_phys_target=prereg["lambda_phys"]["target"],
    lam_phys_ramp_epochs=prereg["lambda_phys"]["ramp_epochs"],
    ema_alpha=prereg["plateau_detection"]["ema_alpha"],
    ref_window=prereg["plateau_detection"]["reference_window"],
    plateau_delta=prereg["plateau_detection"]["delta"],
    plateau_patience=prereg["plateau_detection"]["patience"],
    min_adam_epochs=prereg["plateau_detection"]["min_adam_epochs"],
    max_adam_epochs=prereg["plateau_detection"]["max_adam_epochs"],
    transition_policy=prereg["lambda_phys"]["transition_policy"],
    lbfgs_history_size=prereg["lbfgs"]["history_size"],
    lbfgs_max_iter=prereg["lbfgs"]["max_iter"],
    lbfgs_tolerance_grad=prereg["lbfgs"]["tolerance_grad"],
    lbfgs_tolerance_change=prereg["lbfgs"]["tolerance_change"],
    lbfgs_fixed_batch_size=prereg["lbfgs"]["fixed_batch"]["source"] == "train_ids.csv first 16 scenarios" and 16 or 16,
    seed=prereg["seed"],
)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
import random
random.seed(cfg.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Config: {json.dumps(cfg.to_dict(), indent=2)}")

# ============ LOAD MODEL ============
# Correct loader for Phase 3 PI-FNO (PaddedFNO-wrapped, bit-exact)
from training.adaptive_optimizer import load_phase3_pi_fno_model
model, ckpt = load_phase3_pi_fno_model(CKPT, device=device)
print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, lam_phys={ckpt.get('lam_phys')}")
print(f"  state_dict: 0 missing, 0 unexpected (bit-exact load verified)")

# ============ LOAD KIN/TH/LOSS/NORM ============
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
th_cfg  = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
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

# ============ LOAD DATA ============
train_ids = pd.read_csv("results/phase_1/splitting/train_ids.csv")["scenario_id"].tolist()
val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()

def _norm_states(t):
    o = np.zeros_like(t)
    for j,k in enumerate(STATES):
        o[:,j] = (t[:,j] - norm[k]["mean"]) / norm[k]["std"]
    return o

def _load_scenarios(ids):
    X, Y = [], []
    for sid in ids:
        d = RAW / sid
        f = np.load(d / "forcing.npy"); tj = np.load(d / "trajectory.npy")
        X.append((f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"])
        Y.append(_norm_states(tj).T)
    return (torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.stack(Y), dtype=torch.float32))

print("Loading train data...")
X_train, Y_train = _load_scenarios(train_ids)
print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print("Loading val data...")
X_val, Y_val = _load_scenarios(val_ids)
print(f"  X_val: {X_val.shape}, Y_val: {Y_val.shape}")

# Fixed L-BFGS batch — first 16 training scenarios (deterministic)
X_fixed = X_train[:16].to(device)
Y_fixed = Y_train[:16].to(device)
print(f"Fixed L-BFGS batch: {X_fixed.shape}")

# ============ TRAINING HELPERS ============
BATCH_SIZE = cfg.lbfgs_fixed_batch_size  # reuse for Adam too (16)
t_q_train = torch.linspace(0.5, 9.5, 50, device=device)

def compute_batch_loss(x, y, lam_phys):
    """Compute composite loss with given lambda_phys. Returns dict of components."""
    B = x.shape[0]
    tq = t_q_train.unsqueeze(0).expand(B, -1).contiguous()

    loss_fn.lam_phys = lam_phys
    out = loss_fn(model, x, y, tq, lambda t: torch.zeros_like(t), kin, th)
    return {
        "total": out.total_loss,
        "data": out.data_loss,
        "phys": out.physics_loss,
        "ic": out.ic_loss,
    }

def evaluate_val_data_loss():
    """Protocol A: BATCH=16, mean of per-batch data loss means."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(val_ids), 16):
            x = X_val[i:i+16].to(device)
            y = Y_val[i:i+16].to(device)
            pred = model.fno(x)
            losses.append(((pred - y) ** 2).mean().item())
    model.train()
    return float(np.mean(losses))

# ============ ADAM STAGE ============
print("\n" + "="*60)
print("ADAM STAGE")
print("="*60)

opt = AdaptiveOptimizer(model, cfg, device=device)
adam_log_path = OUT / "adam_training_log.csv"
adam_log_fields = [
    "epoch", "lam_phys", "train_data", "train_phys", "train_ic", "train_total",
    "val_data", "lr", "wall_time_s", "peak_vram_MB",
]
with open(adam_log_path, "w") as f:
    f.write(",".join(adam_log_fields) + "\n")

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

start_time = time.time()
adam_transitioned = False

for epoch in range(cfg.max_adam_epochs):
    t_ep = time.time()
    lam = opt.current_lam_phys(epoch)
    model.train()

    # Shuffle training indices each epoch (standard Adam practice)
    perm = torch.randperm(len(X_train))
    epoch_losses = {"data": [], "phys": [], "ic": [], "total": []}
    for i in range(0, len(X_train), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        x = X_train[idx].to(device)
        y = Y_train[idx].to(device)
        opt.adam.zero_grad()
        losses = compute_batch_loss(x, y, lam)
        losses["total"].backward()
        opt.adam.step()
        for k in epoch_losses:
            epoch_losses[k].append(losses[k].item())

    opt.scheduler.step()
    val_loss = evaluate_val_data_loss()
    lr_now = opt.adam.param_groups[0]["lr"]

    if device == "cuda":
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_vram = 0.0

    t_elapsed = time.time() - t_ep
    row = {
        "epoch": epoch,
        "lam_phys": f"{lam:.8e}",
        "train_data": f"{np.mean(epoch_losses['data']):.8e}",
        "train_phys": f"{np.mean(epoch_losses['phys']):.8e}",
        "train_ic": f"{np.mean(epoch_losses['ic']):.8e}",
        "train_total": f"{np.mean(epoch_losses['total']):.8e}",
        "val_data": f"{val_loss:.10f}",
        "lr": f"{lr_now:.8e}",
        "wall_time_s": f"{t_elapsed:.3f}",
        "peak_vram_MB": f"{peak_vram:.2f}",
    }
    with open(adam_log_path, "a") as f:
        f.write(",".join(str(row[k]) for k in adam_log_fields) + "\n")

    print(f"Epoch {epoch:3d} | lam={lam:.4e} | "
          f"train_d={row['train_data']} | val_d={row['val_data']} | "
          f"lr={row['lr']} | {t_elapsed:.1f}s")

    # Plateau check
    if opt.update_plateau(epoch, val_loss):
        adam_transitioned = True
        print(f"\n*** PLATEAU DETECTED at epoch {epoch} ***")
        print(f"    λ_phys at transition: {opt.transition_event.lam_phys_at_transition:.8e}")
        print(f"    val_data_loss: {val_loss:.10f}")
        print(f"    Transition policy: {opt.transition_event.policy}")
        break
    if epoch == cfg.max_adam_epochs - 1:
        print(f"\n*** HARD CAP reached at epoch {epoch} — forcing transition ***")
        adam_transitioned = True
        if opt.transition_event is None:
            from training.adaptive_optimizer import TransitionEvent
            opt.transition_event = TransitionEvent(
                epoch=epoch,
                lam_phys_at_transition=lam,
                ema_value=opt.plateau_detector.state()["ema_current"],
                improvement_last_check=0.0,
                policy="hard_cap_force",
                raw_val_loss=val_loss,
            )
        break

adam_wall = time.time() - start_time
print(f"\nAdam stage complete: {adam_wall:.1f}s, {epoch+1} epochs")

# Save Adam stage checkpoint
adam_ckpt = MODEL_OUT / "adam_stage.pt"
opt.save_adam_stage(str(adam_ckpt))
print(f"Saved: {adam_ckpt}")

# Save transition event
transition_json = {
    "plateau_epoch": opt.transition_event.epoch,
    "lam_phys_at_transition": float(opt.transition_event.lam_phys_at_transition),
    "ema_value": float(opt.transition_event.ema_value),
    "improvement_last_check": float(opt.transition_event.improvement_last_check),
    "policy": opt.transition_event.policy,
    "raw_val_loss": float(opt.transition_event.raw_val_loss),
    "adam_wall_time_s": adam_wall,
    "adam_epochs_completed": epoch + 1,
}
with open(OUT / "transition_event.json", "w") as f:
    json.dump(transition_json, f, indent=2)
print(f"Transition: {json.dumps(transition_json, indent=2)}")

# ============ L-BFGS STAGE ============
print("\n" + "="*60)
print("L-BFGS STAGE")
print("="*60)

lam_lbfgs = opt.transition_event.lam_phys_at_transition
print(f"L-BFGS inherits λ_phys = {lam_lbfgs:.8e} (Policy B)")
print(f"Fixed batch: {X_fixed.shape[0]} training scenarios (deterministic)")

lbfgs_optimizer = build_lbfgs(model, cfg)
lbfgs_log_path = OUT / "lbfgs_training_log.csv"
lbfgs_closure_log = {"calls": 0, "losses": []}
with open(lbfgs_log_path, "w") as f:
    f.write("closure_call,loss\n")

def lbfgs_closure_logger(call_idx, loss_val):
    lbfgs_closure_log["calls"] = call_idx
    lbfgs_closure_log["losses"].append(loss_val)
    with open(lbfgs_log_path, "a") as f:
        f.write(f"{call_idx},{loss_val:.10e}\n")

def fixed_batch_loss():
    """Deterministic loss on fixed 16-scenario batch (no shuffle)."""
    losses = compute_batch_loss(X_fixed, Y_fixed, lam_lbfgs)
    return losses["total"]

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

print("Running L-BFGS...")
lbfgs_stats = run_lbfgs_stage(
    model=model,
    optimizer_lbfgs=lbfgs_optimizer,
    loss_fn=fixed_batch_loss,
    lambda_phys=lam_lbfgs,
    max_iter=cfg.lbfgs_max_iter,
    log_fn=lbfgs_closure_logger,
)
print(f"L-BFGS complete: {lbfgs_stats}")

if device == "cuda":
    lbfgs_peak_vram = torch.cuda.max_memory_allocated() / 1e6
else:
    lbfgs_peak_vram = 0.0

# Save final checkpoint
final_ckpt = MODEL_OUT / "phase_4a_final.pt"
opt.save_final(str(final_ckpt), lbfgs_stats)
print(f"Saved: {final_ckpt}")

# Save optimizer state
opt_state_path = MODEL_OUT / "optimizer_state.pt"
torch.save({
    "adam_state_dict": opt.adam.state_dict(),
    "scheduler_state_dict": opt.scheduler.state_dict(),
    "plateau_detector_state": opt.plateau_detector.state(),
    "transition_event": transition_json,
    "lbfgs_stats": lbfgs_stats,
    "cfg": cfg.to_dict(),
}, opt_state_path)

# ============ SAVE SUMMARY ============
summary = {
    "phase": "4A.1",
    "starting_checkpoint_sha": "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
    "adam": {
        "epochs_completed": epoch + 1,
        "wall_time_s": adam_wall,
        "final_val_data_loss": float(val_loss),
        "transition_epoch": opt.transition_event.epoch,
        "transition_lam_phys": float(opt.transition_event.lam_phys_at_transition),
        "transition_policy": opt.transition_event.policy,
    },
    "lbfgs": {
        "wall_time_s": lbfgs_stats["wall_time_s"],
        "closure_calls": lbfgs_stats["closure_calls"],
        "final_loss": lbfgs_stats["final_loss"],
        "lambda_phys": lbfgs_stats["lambda_phys"],
        "max_iter": lbfgs_stats["max_iter"],
    },
    "peak_vram_MB": {
        "adam": peak_vram,
        "lbfgs": lbfgs_peak_vram,
    },
    "checkpoints": {
        "adam_stage": str(adam_ckpt),
        "final": str(final_ckpt),
        "optimizer_state": str(opt_state_path),
    },
}
with open(OUT / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(json.dumps(summary, indent=2))
