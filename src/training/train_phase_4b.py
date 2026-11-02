"""Phase 4B variant trainer.

Trains P4-A, P4-B, P4-C, or P4-D from the locked Phase 3 PI-FNO checkpoint
using the frozen Phase 3 Adam protocol. Each variant differs only in the
intervention set (frozen at Step 4B.0):

    P4-A: PCGrad only
    P4-B: Unified latent padding only
    P4-C: Adaptive lambda only
    P4-D: PCGrad + Adaptive lambda + Unified latent padding

No test-set data is loaded. Only train_ids.csv and validation_ids.csv.
"""
from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters
from physics_informed.derivatives import load_phase2_fno, SCFNOModel, compute_du_dt
from physics_informed.decoder import build_decoder
from physics_informed.losses import load_composite_loss
from training.adaptive_optimizer import load_phase3_pi_fno_model
from training.pcgrad import PCGradOptimizer
from training.adaptive_lambda import AdaptiveLambda
from models.boundary_aware_fno import UnifiedLatentFNO

RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
NQ = 50
T_LO, T_HI = 0.5, 9.5
BATCH = 16
MAX_EPOCHS = 200
PATIENCE = 30

# ---------- Variant configs ----------
VARIANTS = {
    "P4-A": {"pcgrad": True, "adaptive_lambda": False, "unified_padding": False},
    "P4-B": {"pcgrad": False, "adaptive_lambda": False, "unified_padding": True},
    "P4-C": {"pcgrad": False, "adaptive_lambda": True, "unified_padding": False},
    "P4-D": {"pcgrad": True, "adaptive_lambda": True, "unified_padding": True},
}


def phase3_default_ramp(epoch, init=1e-4, target=1.0, ramp_epochs=100):
    if epoch >= ramp_epochs:
        return target
    frac = epoch / ramp_epochs
    ratio = target / init
    return init * (10.0 ** (frac * math.log10(ratio)))


def build_query_times(B, device="cuda"):
    near_start = torch.linspace(0.5, 1.5, 4, device=device)
    middle = torch.linspace(1.5, 8.5, 42, device=device)
    near_end = torch.linspace(8.5, 9.5, 4, device=device)
    t_q = torch.cat([near_start, middle, near_end])
    return t_q.unsqueeze(0).expand(B, -1).contiguous()


def load_kin_th():
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
    return kin, th


def load_data(split):
    ids = pd.read_csv(f"results/phase_1/splitting/{split}_ids.csv")["scenario_id"].tolist()
    norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
    X, Y, T = [], [], []
    for sid in ids:
        d = RAW / sid
        f = np.load(d / "forcing.npy")
        tj = np.load(d / "trajectory.npy")
        t = np.load(d / "time.npy")
        x_norm = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        y_norm = np.zeros_like(tj)
        for j, k in enumerate(STATES):
            y_norm[:, j] = (tj[:, j] - norm[k]["mean"]) / norm[k]["std"]
        X.append(x_norm)
        Y.append(y_norm.T)  # [9, 200]
        T.append(t)
    return {
        "ids": ids,
        "X": torch.tensor(np.stack(X), dtype=torch.float32),  # [N, 200]
        "Y": torch.tensor(np.stack(Y), dtype=torch.float32),  # [N, 9, 200]
        "time": torch.tensor(np.stack(T), dtype=torch.float32),  # [N, 200]
        "norm": norm,
    }


def compute_gradient_cosine(grads_a, grads_b, eps=1e-8):
    """Compute cosine between two grad lists (both may have Nones)."""
    flat_a, flat_b = [], []
    for ga, gb in zip(grads_a, grads_b):
        if ga is not None and gb is not None:
            flat_a.append(ga.flatten())
            flat_b.append(gb.flatten())
    if not flat_a:
        return 0.0
    fa = torch.cat(flat_a)
    fb = torch.cat(flat_b)
    na = torch.linalg.norm(fa)
    nb = torch.linalg.norm(fb)
    return float((fa * fb).sum() / (na * nb + eps))


def evaluate_val_data_loss(model, val_data, device, batch_size=BATCH):
    """Protocol A: BATCH=16, mean of per-batch data loss means."""
    model.eval()
    losses = []
    N = val_data["X"].shape[0]
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = val_data["X"][i:i+batch_size].unsqueeze(1).to(device)
            y = val_data["Y"][i:i+batch_size].to(device)
            pred = model.fno(x)
            losses.append(((pred - y) ** 2).mean().item())
    model.train()
    return float(np.mean(losses))


def main(variant_name):
    assert variant_name in VARIANTS, f"Unknown variant: {variant_name}"
    cfg = VARIANTS[variant_name]
    print(f"\n{'='*70}")
    print(f"TRAINING {variant_name} — pcgrad={cfg['pcgrad']}, "
          f"adaptive_lambda={cfg['adaptive_lambda']}, padding={cfg['unified_padding']}")
    print(f"{'='*70}")

    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    OUT = Path("results/phase_4/phase_4b/training")
    MODEL_OUT = Path("models/sc_fno/phase_4/phase_4b")

    # Load Phase 3 checkpoint
    model, ckpt = load_phase3_pi_fno_model(
        "models/sc_fno/phase_3/pi_fno/best_val.pt", device=device
    )
    print(f"Loaded Phase 3 PI-FNO (epoch={ckpt.get('epoch')}, lam_phys={ckpt.get('lam_phys')})")

    # Wrap FNO with UnifiedLatentFNO if variant uses padding
    if cfg["unified_padding"]:
        model.fno = UnifiedLatentFNO(model.fno, pad_size=20, pad_mode="reflect")
        print("  Wrapped with UnifiedLatentFNO(pad_size=20, reflect)")
        # Move wrapped model to device
        model = model.to(device)

    # Load data
    print("Loading train + val data...")
    train_data = load_data("train")
    val_data = load_data("validation")
    print(f"  Train: {train_data['X'].shape[0]} scenarios")
    print(f"  Val:   {val_data['X'].shape[0]} scenarios")

    kin, th = load_kin_th()
    loss_fn = load_composite_loss()

    # Build optimizer / intervention stack
    params = [p for p in model.parameters() if p.requires_grad]
    base_opt = Adam(params, lr=1.0e-3, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingLR(base_opt, T_max=200, eta_min=0.0)

    if cfg["pcgrad"]:
        pcgrad_opt = PCGradOptimizer(base_opt, eps=1e-8)
        print("  PCGrad enabled")

    if cfg["adaptive_lambda"]:
        adaptive_lambda = AdaptiveLambda()
        print("  AdaptiveLambda enabled")
    else:
        adaptive_lambda = None
        print("  Using Phase 3 default lambda ramp")

    # Log CSV
    log_path = OUT / f"{variant_name}_training_log.csv"
    log_fields = [
        "epoch", "wall_time_s", "peak_vram_MB",
        "train_data_loss", "train_physics_loss", "train_ic_loss", "train_total_loss",
        "val_data_loss",
        "lambda_phys",
        "epoch_raw_cosine",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # Training loop
    N_train = train_data["X"].shape[0]
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    t_start = time.time()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(MAX_EPOCHS):
        t_ep = time.time()

        # Determine lambda for THIS epoch
        if adaptive_lambda is not None:
            lam_phys = adaptive_lambda.current()
        else:
            lam_phys = phase3_default_ramp(epoch)

        model.train()
        perm = torch.randperm(N_train)
        epoch_data, epoch_physics, epoch_ic = [], [], []
        epoch_cosines = []

        for bi in range(0, N_train, BATCH):
            idx = perm[bi:bi+BATCH]
            x = train_data["X"][idx].unsqueeze(1).to(device)   # [B, 1, 200]
            y = train_data["Y"][idx].to(device)                 # [B, 9, 200]
            B = x.shape[0]
            t_q = build_query_times(B, device)

            if cfg["unified_padding"]:
                # Single latent for both paths
                latent = model.fno.get_latent(x)
                # Data path: decoder on 200-point grid
                t_data_grid = train_data["time"][idx].to(device)  # [B, 200]
                u_data = model.decoder(latent, t_data_grid)       # [B, 9, 200]
                L_data = ((u_data - y) ** 2).mean()
                # Physics path: decoder on t_q
                u_phys = model.decoder(latent, t_q)               # [B, 9, Nq]
                L_physics = loss_fn.compute_physics_loss_from_u(
                    u_phys, t_q, x, kin, th
                )
                # IC path: decoder at t=0
                t_ic = torch.zeros(B, 1, device=device)
                u_ic = model.decoder(latent, t_ic)                # [B, 9, 1]
                ic_gt = y[:, :, 0].unsqueeze(-1)
                ic_scale = torch.tensor(
                    [train_data["norm"][k]["std"] for k in STATES],
                    device=device
                ).view(1, 9, 1)
                L_ic = (((u_ic - ic_gt) / ic_scale) ** 2).mean()
            else:
                # Fallback: raw FNO path (Phase 3 style)
                pred_data = model.fno(x)
                L_data = ((pred_data - y) ** 2).mean()
                # Physics path via existing composite loss internals
                z, u_hat, du_hat_dt = compute_du_dt(model, x, t_q)
                u_phys = u_hat
                L_physics = loss_fn.compute_physics_loss_from_u(
                    u_phys, t_q, x, kin, th
                )
                z_ic = model._fno_latent(x)
                t_ic = torch.zeros(B, 1, device=device)
                u_ic = model.decoder(z_ic, t_ic)
                ic_gt = y[:, :, 0].unsqueeze(-1)
                ic_scale = torch.tensor(
                    [train_data["norm"][k]["std"] for k in STATES],
                    device=device
                ).view(1, 9, 1)
                L_ic = (((u_ic - ic_gt) / ic_scale) ** 2).mean()

            epoch_data.append(L_data.item())
            epoch_physics.append(L_physics.item() if isinstance(L_physics, torch.Tensor) else L_physics)
            epoch_ic.append(L_ic.item())

            if cfg["pcgrad"]:
                # Compute separate gradients
                grad_data = torch.autograd.grad(L_data, params, retain_graph=True, allow_unused=True)
                grad_physics = torch.autograd.grad(L_physics, params, retain_graph=True, allow_unused=True)
                grad_ic = torch.autograd.grad(L_ic, params, retain_graph=False, allow_unused=True)

                # Raw cosine for AdaptiveLambda
                cos = compute_gradient_cosine(grad_data, grad_physics)
                epoch_cosines.append(cos)

                pcgrad_opt.set_loss_values(
                    float(L_data), float(L_physics), float(L_ic),
                    lam_phys, 1.0
                )
                pcgrad_opt.set_gradients(grad_data, grad_physics, grad_ic)
                pcgrad_opt.set_epoch(epoch)
                pcgrad_opt.step()
            else:
                total_loss = L_data + lam_phys * L_physics + 1.0 * L_ic
                base_opt.zero_grad()
                total_loss.backward()
                base_opt.step()

        scheduler.step()

        # End-of-epoch lambda update
        epoch_cosine_mean = float(np.mean(epoch_cosines)) if epoch_cosines else 0.0
        if adaptive_lambda is not None:
            adaptive_lambda.update(epoch, epoch_cosine_mean)
            lam_phys = adaptive_lambda.current()

        # Evaluate val data loss
        val_loss = evaluate_val_data_loss(model, val_data, device)

        # VRAM
        if device == "cuda":
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / 1e6
        else:
            peak_vram = 0.0

        wall = time.time() - t_ep
        row = {
            "epoch": epoch,
            "wall_time_s": round(wall, 2),
            "peak_vram_MB": round(peak_vram, 2),
            "train_data_loss": round(float(np.mean(epoch_data)), 6),
            "train_physics_loss": round(float(np.mean(epoch_physics)), 6),
            "train_ic_loss": round(float(np.mean(epoch_ic)), 6),
            "train_total_loss": round(float(np.mean(epoch_data)) + lam_phys * float(np.mean(epoch_physics)) + float(np.mean(epoch_ic)), 6),
            "val_data_loss": round(val_loss, 6),
            "lambda_phys": f"{lam_phys:.6e}",
            "epoch_raw_cosine": round(epoch_cosine_mean, 6),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        print(f"Epoch {epoch:3d} | data={row['train_data_loss']:.4f} "
              f"phys={row['train_physics_loss']:.4e} val={row['val_data_loss']:.6f} "
              f"lam={lam_phys:.2e} | {wall:.1f}s")

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "val_data_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "config": cfg,
            }, MODEL_OUT / f"{variant_name.lower().replace('-','_')}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch}, best val {best_val:.6f})")
                break

    total_wall = time.time() - t_start
    print(f"\n{variant_name} DONE: {epoch+1} epochs in {total_wall:.1f}s, best val={best_val:.6f} @ epoch {best_epoch}")

    # Save summary
    summary = {
        "variant": variant_name,
        "config": cfg,
        "starting_checkpoint_sha": "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_data_loss": best_val,
        "total_wall_time_s": total_wall,
        "peak_vram_MB": peak_vram,
        "lambda_phys_final": float(adaptive_lambda.current()) if adaptive_lambda else phase3_default_ramp(epoch+1),
    }
    with open(OUT / f"{variant_name.lower().replace('-','_')}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save training config
    with open(OUT / f"{variant_name.lower().replace('-','_')}_training_config.yaml", "w") as f:
        yaml.dump({
            "variant": variant_name,
            "config": cfg,
            "optimizer": "Adam",
            "lr": 1.0e-3,
            "betas": [0.9, 0.999],
            "scheduler": "CosineAnnealingLR",
            "T_max": 200,
            "batch_size": BATCH,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "seed": 42,
            "starting_checkpoint": "models/sc_fno/phase_3/pi_fno/best_val.pt",
            "starting_checkpoint_sha": "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
        }, f, default_flow_style=False)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    main(args.variant)
