"""Phase 2 Step 2.3 — Standard FNO supervised training.

Supervised MSE only. No physics constraints.
- Adam lr=1e-3, cosine annealing per epoch (T_max=500)
- Max epochs 500, early stopping patience 30 on val MSE
- Gradient clipping max_norm=1.0 (usage logged per epoch)
- GPU timing: 3 warm-up + 10 repeats + CUDA sync + median/mean
- Test set NEVER loaded during training
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from data.dataset import make_dataloader  # noqa: E402


# ---------- Config ----------
@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 500
    early_stopping_patience: int = 30
    scheduler: str = "cosine"
    scheduler_T_max: int = 500
    scheduler_eta_min: float = 0.0
    gradient_clip: float = 1.0
    device: str = "cuda"
    n_modes: int = 16
    hidden_channels: int = 32
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 9
    temporal_points: int = 200


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_fno(cfg: TrainingConfig) -> nn.Module:
    return FNO(
        n_modes=(cfg.n_modes,),
        hidden_channels=cfg.hidden_channels,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        n_layers=cfg.n_layers,
        positional_embedding=None,
        domain_padding=None,
    )


def normalized_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Equal-weight MSE across all 9 channels, B, N_t."""
    return ((y_hat - y) ** 2).mean()


def measure_step_timing(model, x, y, optimizer, device, n_warmup=3, n_repeats=10):
    """GPU timing with warm-up, sync, and repeats — Reviewer Item C."""
    model.train()
    # Warm-up
    for _ in range(n_warmup):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = normalized_mse(y_hat, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = normalized_mse(y_hat, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        "warmup_iterations": n_warmup,
        "repeats": n_repeats,
        "sync_placement": "before_and_after_each_repeat",
        "timer": "time.perf_counter",
        "mean_s": float(times.mean()),
        "median_s": float(np.median(times)),
        "min_s": float(times.min()),
        "max_s": float(times.max()),
    }


def main():
    cfg = TrainingConfig()
    set_seeds(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = PROJECT / "results/phase_2/training"
    model_dir = PROJECT / "models/baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save frozen config
    import yaml
    with open(out_dir / "training_config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f)

    # Dataloaders — TRAIN + VAL ONLY (no test)
    train_dl = make_dataloader("train", batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_dl   = make_dataloader("validation", batch_size=cfg.batch_size)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # Model / optimizer / scheduler
    model = build_fno(cfg).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.scheduler_T_max, eta_min=cfg.scheduler_eta_min)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # GPU timing measurement (one batch)
    x_b, y_b, _ = next(iter(train_dl))
    x_b = x_b.to(device)
    y_b = y_b.to(device)
    timing = measure_step_timing(model, x_b, y_b, optimizer, device)
    with open(out_dir / "timing_report.json", "w") as f:
        json.dump(timing, f, indent=2)
    print(f"Timing (median/mean): {timing['median_s']*1000:.2f} ms / {timing['mean_s']*1000:.2f} ms")

    # CSV loggers
    epoch_csv = open(out_dir / "epoch_metrics.csv", "w", newline="")
    epoch_writer = csv.writer(epoch_csv)
    epoch_writer.writerow([
        "epoch", "train_mse", "val_mse", "learning_rate",
        "grad_norm_pre_clip", "grad_norm_post_clip", "grad_clip_active",
        "epoch_time_s", "peak_alloc_MB", "peak_reserved_MB", "is_best",
    ])

    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    stability_event = "none"
    training_start = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        t_epoch_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        # ---- Train ----
        model.train()
        train_losses = []
        grad_pre_list = []
        grad_post_list = []
        clip_active_count = 0
        batch_count = 0

        for x, y, _ in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = normalized_mse(y_hat, y)

            if not torch.isfinite(loss):
                stability_event = f"non_finite_loss_epoch_{epoch}"
                break

            loss.backward()

            # Pre-clip norm (returned by clip_grad_norm_)
            pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            if not torch.isfinite(pre_norm):
                stability_event = f"non_finite_grad_norm_epoch_{epoch}"
                break
            pre_val = float(pre_norm)
            grad_pre_list.append(pre_val)

            # Post-clip norm — safe flatten + vector_norm (no manual pow overflow)
            flat = torch.cat([p.grad.detach().flatten() for p in model.parameters()
                              if p.grad is not None])
            post_val = float(torch.linalg.vector_norm(flat))
            grad_post_list.append(post_val)

            if pre_val > cfg.gradient_clip:
                clip_active_count += 1

            optimizer.step()
            train_losses.append(float(loss.detach()))
            batch_count += 1

        if stability_event != "none":
            break

        # Scheduler step per epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _ in val_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_hat = model(x)
                val_losses.append(float(normalized_mse(y_hat, y).detach()))

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mse = float(np.mean(val_losses)) if val_losses else float("nan")
        grad_pre_mean = float(np.mean(grad_pre_list)) if grad_pre_list else 0.0
        grad_post_mean = float(np.mean(grad_post_list)) if grad_post_list else 0.0
        clip_active = 1 if clip_active_count > 0 else 0

        epoch_time = time.perf_counter() - t_epoch_start
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6

        # ---- Best checkpoint + early stopping ----
        is_best = 0
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            is_best = 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_mse": val_mse,
                "train_mse": train_mse,
                "config": asdict(cfg),
            }, model_dir / "fno_best.pt")
        else:
            patience_counter += 1

        epoch_writer.writerow([
            epoch, train_mse, val_mse, current_lr,
            grad_pre_mean, grad_post_mean, clip_active,
            epoch_time, peak_alloc, peak_reserved, is_best,
        ])
        epoch_csv.flush()

        # Exploding gradient guard
        if grad_pre_mean > 1e3:
            stability_event = f"exploding_gradients_epoch_{epoch}"
            break

        if epoch % 10 == 0 or is_best:
            print(f"Epoch {epoch:4d} | train {train_mse:.6f} | val {val_mse:.6f} "
                  f"| lr {current_lr:.2e} | clip {clip_active} | "
                  f"peak {peak_alloc:.1f}MB | best {best_epoch} | patience {patience_counter}")

        if patience_counter >= cfg.early_stopping_patience:
            stability_event = f"early_stop_epoch_{epoch}"
            break

    total_time = time.perf_counter() - training_start
    epoch_csv.close()

    # Save final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
    }, model_dir / "fno_baseline.pt")

    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "config": asdict(cfg),
    }, model_dir / "training_state.pt")

    # Training summary
    summary = {
        "config": asdict(cfg),
        "total_params": total_params,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "stability_event": stability_event,
        "total_time_s": total_time,
        "total_time_min": total_time / 60,
        "peak_alloc_MB_any_epoch": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_MB_any_epoch": torch.cuda.max_memory_reserved() / 1e6,
        "three_GB_target_MB": 3000,
        "under_3GB": bool((torch.cuda.max_memory_allocated() / 1e6) < 3000),
        "under_60min": bool(total_time < 3600),
        "timing": timing,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TRAINING SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
