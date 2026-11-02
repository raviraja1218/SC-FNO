"""Step 5.1 — Corrected Phase 4B retraining.

Fixes Phase 4B Bug A: patience measured from ramp-completion epoch (100),
NOT from best-val epoch. Guarantees λ_phys reaches target before early stop.

Trains one variant at a time. CLI: python step_5_1_train.py <variant>

Uses the proven Phase 4B integration patterns:
  - load_phase3_pi_fno_model() for the Phase 3 checkpoint
  - PCGradOptimizer(base_opt) wrapping with set_gradients/step
  - AdaptiveLambda() with defaults (matches frozen config)
"""
import sys, json, time
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"

STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]


# ============================================================================
# Config
# ============================================================================
def load_config(variant):
    cfgs = {
        "P5-A": {"name": "P5-A", "pcgrad": True,  "unified_padding": False, "adaptive_lambda": False},
        "P5-B": {"name": "P5-B", "pcgrad": False, "unified_padding": True,  "adaptive_lambda": False},
        "P5-C": {"name": "P5-C", "pcgrad": False, "unified_padding": False, "adaptive_lambda": True},
        "P5-D": {"name": "P5-D", "pcgrad": True,  "unified_padding": True,  "adaptive_lambda": True},
    }
    cfg = cfgs[variant].copy()
    cfg.update({
        "starting_checkpoint": "models/sc_fno/phase_3/pi_fno/best_val.pt",
        "starting_checkpoint_sha": "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
        "seed": 42, "batch_size": 16, "lr": 1e-3,
        "T_max": 300, "max_epochs": 300,
        "patience_after_ramp": 100,
        "min_epochs_before_stop": 100,
        "pad_size": 20,
        "pcgrad_eps": 1e-8,
        "lambda_alpha": 0.05, "lambda_min": 1e-5, "lambda_max": 10.0,
        "lambda_init": 1e-4, "ema_beta": 0.1, "lookback_epochs": 20,
        "min_lambda_for_stop": 0.01,
    })
    return cfg


def fixed_lambda(epoch, init=1e-4, target=1.0, ramp=100):
    if epoch >= ramp:
        return target
    return init * (10.0 ** (epoch / ramp * np.log10(target / init)))


# ============================================================================
# Data loading
# ============================================================================
def load_data():
    norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
    train_ids = pd.read_csv("results/phase_1/splitting/train_ids.csv")["scenario_id"].tolist()
    val_ids = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()

    def load_split(ids):
        X = np.zeros((len(ids), 200), dtype=np.float32)
        Y = np.zeros((len(ids), 9, 200), dtype=np.float32)
        for i, sid in enumerate(ids):
            d = RAW / sid
            f = np.load(d / "forcing.npy")
            t = np.load(d / "trajectory.npy")
            X[i] = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
            for j, k in enumerate(STATES):
                Y[i, j] = (t[:, j] - norm[k]["mean"]) / norm[k]["std"]
        return torch.tensor(X), torch.tensor(Y)

    X_tr, Y_tr = load_split(train_ids)
    X_va, Y_va = load_split(val_ids)
    return X_tr, Y_tr, X_va, Y_va, norm


# ============================================================================
# Model + intervention loader (returns 8-tuple)
# ============================================================================
def load_model_with_interventions(cfg):
    from training.adaptive_optimizer import load_phase3_pi_fno_model
    from physics_informed.losses import load_composite_loss
    from models.boundary_aware_fno import UnifiedLatentFNO
    from training.pcgrad import PCGradOptimizer
    from training.adaptive_lambda import AdaptiveLambda

    # Load Phase 3 PI-FNO
    model, ckpt = load_phase3_pi_fno_model(
        checkpoint_path=cfg["starting_checkpoint"], device=device)
    model.train()

    # Wrap FNO for unified padding variants
    if cfg["unified_padding"]:
        model.fno = UnifiedLatentFNO(model.fno, pad_size=cfg["pad_size"], pad_mode="reflect")
        model = model.to(device)

    # Loss + physics params
    loss_fn = load_composite_loss()
    from training.train_sc_fno import get_kin, get_th
    kin = get_kin()
    th = get_th()

    # Base optimizer (always built)
    params = [p for p in model.parameters() if p.requires_grad]
    base_opt = torch.optim.Adam(params, lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_opt, T_max=cfg["T_max"], eta_min=0.0)

    # PCGrad wrapper
    pcgrad_opt = None
    if cfg["pcgrad"]:
        pcgrad_opt = PCGradOptimizer(base_opt, eps=cfg["pcgrad_eps"])

    # AdaptiveLambda (defaults match frozen config)
    adaptive_lambda = None
    if cfg["adaptive_lambda"]:
        adaptive_lambda = AdaptiveLambda()

    return model, loss_fn, kin, th, base_opt, scheduler, pcgrad_opt, adaptive_lambda


# ============================================================================
# Training
# ============================================================================
def train_variant(variant):
    cfg = load_config(variant)
    print(f"\n{'='*70}")
    print(f"Training {cfg['name']}: PCGrad={cfg['pcgrad']}, "
          f"Padding={cfg['unified_padding']}, AdaptiveLambda={cfg['adaptive_lambda']}")
    print(f"{'='*70}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    X_tr, Y_tr, X_va, Y_va, norm = load_data()
    N_tr = X_tr.shape[0]
    print(f"Train: {N_tr} scenarios; Val: {X_va.shape[0]} scenarios")

    (model, loss_fn, kin, th, base_opt, scheduler,
     pcgrad_opt, adaptive_lambda) = load_model_with_interventions(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"  PCGrad: {pcgrad_opt is not None}, AdaptiveLambda: {adaptive_lambda is not None}")

    best_val = float("inf")
    best_epoch = 0
    epoch_since_best = 0
    lambda_history = []
    cosine_history = []
    history = []
    peak_vram = 0.0
    stop_epoch = None
    stop_reason = None
    start_time = time.time()

    T_QUERY = torch.linspace(0.5, 9.5, 50, device=device)

    from physics_informed.derivatives import compute_du_dt
    from physics_informed.residuals import compute_residuals

    for epoch in range(cfg["max_epochs"]):
        model.train()
        epoch_start = time.perf_counter()

        # Current lambda
        if adaptive_lambda is not None:
            current_lambda = adaptive_lambda.current()
        else:
            current_lambda = fixed_lambda(epoch, init=cfg["lambda_init"], target=1.0, ramp=100)

        perm = torch.randperm(N_tr)
        X_tr_p = X_tr[perm]
        Y_tr_p = Y_tr[perm]

        epoch_data = 0.0
        epoch_phys = 0.0
        epoch_ic = 0.0
        epoch_total = 0.0
        n_batches = 0
        epoch_grad_norm = 0.0
        epoch_cos = None

        for b_start in range(0, N_tr, cfg["batch_size"]):
            b_end = min(b_start + cfg["batch_size"], N_tr)
            xb = X_tr_p[b_start:b_end].unsqueeze(1).to(device)
            yb = Y_tr_p[b_start:b_end].to(device)
            bsz = xb.shape[0]
            tq = T_QUERY.unsqueeze(0).expand(bsz, -1)

            # Data loss
            pred = model.fno(xb)
            L_data = ((pred - yb) ** 2).mean()

            # Physics path — always uses _fno_latent, routes correctly
            z2, u_hat, du_dt = compute_du_dt(model, xb, tq)
            u_phys = loss_fn._denorm_batch(u_hat)
            R = compute_residuals(u_hat=u_phys, du_hat_dt=du_dt, t_query=tq,
                                  rho_ext_fn=lambda t: torch.zeros_like(t),
                                  kin=kin, th=th)
            stacked = R.stacked()
            L_phys_terms = []
            for sj, k in enumerate(STATES):
                s_j = loss_fn.residual_scales[k]["residual_p95"]
                L_phys_terms.append(((stacked[:, sj, :] / s_j) ** 2).mean())
            L_phys = torch.stack(L_phys_terms).mean()

            # IC loss
            t0 = torch.zeros(bsz, 1, device=device)
            u0 = loss_fn._denorm_batch(model.decoder(z2, t0))
            y0_phys = torch.zeros(bsz, 9, device=device)
            for sj, k in enumerate(STATES):
                y0_phys[:, sj] = yb[:, sj, 0] * norm[k]["std"] + norm[k]["mean"]
            L_ic = ((u0[:, :, 0] - y0_phys) ** 2).mean()

            total_loss = L_data + current_lambda * L_phys + L_ic

            if pcgrad_opt is not None:
                # Compute per-loss gradients
                trainable = [p for p in model.parameters() if p.requires_grad]
                g_data = torch.autograd.grad(L_data, trainable, retain_graph=True, allow_unused=True)
                g_phys = torch.autograd.grad(current_lambda * L_phys, trainable,
                                             retain_graph=True, allow_unused=True)
                g_ic = torch.autograd.grad(L_ic, trainable, retain_graph=False, allow_unused=True)

                pcgrad_opt.zero_grad()
                pcgrad_opt.set_gradients(g_data, g_phys, g_ic)
                pcgrad_opt.set_epoch(epoch)
                pcgrad_opt.step()
                scheduler.step()

                try:
                    m = pcgrad_opt.get_logged_metrics()
                    if m and "raw_cosine" in m:
                        epoch_cos = m["raw_cosine"]
                except Exception:
                    pass
                epoch_grad_norm += 0.0
            else:
                base_opt.zero_grad()
                total_loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                base_opt.step()
                scheduler.step()
                epoch_grad_norm += float(gn)

            epoch_data += float(L_data)
            epoch_phys += float(L_phys)
            epoch_ic += float(L_ic)
            epoch_total += float(total_loss)
            n_batches += 1

        # Validation
        model.eval()
        val_data_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for v_start in range(0, X_va.shape[0], cfg["batch_size"]):
                v_end = min(v_start + cfg["batch_size"], X_va.shape[0])
                xb = X_va[v_start:v_end].unsqueeze(1).to(device)
                yb = Y_va[v_start:v_end].to(device)
                pred = model.fno(xb)
                val_data_loss += float(((pred - yb) ** 2).mean())
                n_val += 1
        val_data_loss /= max(n_val, 1)

        avg_gn = epoch_grad_norm / max(n_batches, 1)
        avg_data = epoch_data / max(n_batches, 1)
        avg_phys = epoch_phys / max(n_batches, 1)
        avg_ic = epoch_ic / max(n_batches, 1)
        avg_total = epoch_total / max(n_batches, 1)

        vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        peak_vram = max(peak_vram, vram)

        lambda_history.append({"epoch": epoch, "lambda_phys": current_lambda})
        if epoch_cos is not None:
            cosine_history.append({"epoch": epoch, "raw_cosine": epoch_cos})

        row = {
            "epoch": epoch,
            "train_data_loss": avg_data,
            "train_physics_loss": avg_phys,
            "train_ic_loss": avg_ic,
            "train_total_loss": avg_total,
            "lambda_phys": current_lambda,
            "val_data_loss": val_data_loss,
            "gradient_norm": avg_gn,
            "peak_vram_MB": vram,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        elapsed = time.perf_counter() - epoch_start
        if epoch % 5 == 0 or epoch < 5:
            print(f"  Epoch {epoch:3d}: data={avg_data:.5f} phys={avg_phys:.4f} "
                  f"lam={current_lambda:.4f} val={val_data_loss:.5f} "
                  f"gn={avg_gn:.3f} vram={vram:.1f}MB time={elapsed:.1f}s")

        # Best checkpoint
        if val_data_loss < best_val:
            best_val = val_data_loss
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_data_loss, "epoch": epoch,
                "variant": cfg["name"], "config": cfg,
            }, f"models/sc_fno/phase_5/step_5_1/{variant.lower()}_best.pt")

        # AdaptiveLambda update
        if adaptive_lambda is not None:
            try:
                adaptive_lambda.update(epoch_cos if epoch_cos is not None else 0.0)
            except Exception as e:
                print(f"  [WARN] adaptive_lambda.update failed: {e}")

        # Early stopping — only after min_epochs_before_stop
        if epoch >= cfg["min_epochs_before_stop"]:
            if adaptive_lambda is not None:
                lambda_ok = current_lambda >= cfg["min_lambda_for_stop"]
            else:
                lambda_ok = True

            if lambda_ok:
                if val_data_loss >= best_val - 1e-6:
                    epoch_since_best += 1
                else:
                    epoch_since_best = 0

                if epoch_since_best >= cfg["patience_after_ramp"]:
                    stop_epoch = epoch
                    stop_reason = f"patience={cfg['patience_after_ramp']} after epoch {cfg['min_epochs_before_stop']}"
                    print(f"  EARLY STOP at epoch {epoch}: {stop_reason}")
                    break

    wall_time = time.time() - start_time

    torch.save({
        "model_state_dict": model.state_dict(),
        "val_loss": best_val, "epoch": stop_epoch or cfg["max_epochs"],
        "variant": cfg["name"], "config": cfg,
    }, f"models/sc_fno/phase_5/step_5_1/{variant.lower()}_final.pt")

    torch.save({
        "base_optimizer": base_opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": stop_epoch or cfg["max_epochs"],
    }, f"models/sc_fno/phase_5/step_5_1/{variant.lower()}_state.pt")

    pd.DataFrame(history).to_csv(
        f"results/phase_5/step_5_1/training_logs/{variant.lower()}_training_log.csv", index=False)
    pd.DataFrame(lambda_history).to_csv(
        f"results/phase_5/step_5_1/lambda_history/{variant.lower()}_lambda.csv", index=False)
    if cosine_history:
        pd.DataFrame(cosine_history).to_csv(
            f"results/phase_5/step_5_1/gradient_conflict/{variant.lower()}_cosine.csv", index=False)

    summary = {
        "variant": cfg["name"], "config": cfg,
        "best_val_data_loss": best_val, "best_epoch": best_epoch,
        "final_epoch": stop_epoch or cfg["max_epochs"],
        "stop_reason": stop_reason or "max_epochs_reached",
        "final_lambda_phys": lambda_history[-1]["lambda_phys"],
        "max_lambda_phys": max(h["lambda_phys"] for h in lambda_history),
        "wall_time_s": wall_time, "peak_vram_MB": peak_vram,
        "n_epochs_run": len(history),
        "final_train_data_loss": history[-1]["train_data_loss"],
        "final_train_physics_loss": history[-1]["train_physics_loss"],
        "final_val_data_loss": history[-1]["val_data_loss"],
        "checkpoint_best": f"models/sc_fno/phase_5/step_5_1/{variant.lower()}_best.pt",
        "checkpoint_final": f"models/sc_fno/phase_5/step_5_1/{variant.lower()}_final.pt",
    }
    json.dump(summary, open(
        f"results/phase_5/step_5_1/training_logs/{variant.lower()}_summary.json", "w"), indent=2)

    print(f"\n{cfg['name']} COMPLETE:")
    print(f"  best_val = {best_val:.6f} at epoch {best_epoch}")
    print(f"  final_lambda = {lambda_history[-1]['lambda_phys']:.6f}")
    print(f"  max_lambda = {summary['max_lambda_phys']:.6f}")
    print(f"  epochs run = {len(history)}")
    print(f"  wall time = {wall_time/60:.1f} min")
    print(f"  peak VRAM = {peak_vram:.1f} MB")
    print(f"  stop reason = {summary['stop_reason']}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "P5-A"
    train_variant(variant)
