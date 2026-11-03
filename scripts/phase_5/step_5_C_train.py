"""Step 5.C — Causal decoder training (exploratory seed).

One-seed exploratory run of SCFNOModelV2 with CausalCoordinateDecoder.
If promising, extend to 5 seeds in step 5.C.E.
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
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
OUT = Path("results/phase_5/step_5_C/seeds")
CKPT = Path("models/sc_fno/phase_5/step_5_C")
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

PHASE3_CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"


def fixed_ramp(epoch, init=1e-4, target=1.0, ramp=100):
    if epoch >= ramp:
        return target
    return init * (10.0 ** (epoch / ramp * np.log10(target / init)))


def load_data():
    norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
    tr = pd.read_csv("results/phase_1/splitting/train_ids.csv")["scenario_id"].tolist()
    va = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()

    def load_split(ids):
        X = np.zeros((len(ids), 200), dtype=np.float32)
        Y = np.zeros((len(ids), 9, 200), dtype=np.float32)
        for i, sid in enumerate(ids):
            d = RAW / sid
            f = np.load(d/"forcing.npy")
            t = np.load(d/"trajectory.npy")
            X[i] = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
            for j, k in enumerate(STATES):
                Y[i, j] = (t[:, j] - norm[k]["mean"]) / norm[k]["std"]
        return torch.tensor(X), torch.tensor(Y), norm

    X_tr, Y_tr, norm = load_split(tr)
    X_va, Y_va, _ = load_split(va)
    return X_tr, Y_tr, X_va, Y_va, norm


def build_causal_model(seed=42):
    from training.adaptive_optimizer import load_phase3_pi_fno_model
    from phase_5_corrected.model_v2 import SCFNOModelV2
    from phase_5_corrected.decoder_v2_causal import build_causal_decoder

    model_p3, ckpt = load_phase3_pi_fno_model(PHASE3_CKPT, device=device)
    raw_fno = model_p3.fno.fno if hasattr(model_p3.fno, "fno") else model_p3.fno

    # Replace decoder with causal decoder (fresh init — must retrain decoder)
    causal_decoder = build_causal_decoder(seed=seed).to(device)
    model_v2 = SCFNOModelV2(raw_fno, causal_decoder, pad_size=0).to(device)
    return model_v2


def boundary_check(model, X_va, Y_va, n_scenarios=20):
    """Compute boundary ratio B_n = MAE[start]/MAE[interior]."""
    model.eval()
    START = slice(0, 20)   # first 20 timesteps
    INTERIOR = slice(20, 180)
    END = slice(180, 200)

    mae_start = {k: 0.0 for k in STATES}
    mae_interior = {k: 0.0 for k in STATES}
    mae_end = {k: 0.0 for k in STATES}
    n = 0

    with torch.no_grad():
        for vi in range(0, min(n_scenarios * 16, X_va.shape[0]), 16):
            v_end = min(vi + 16, X_va.shape[0])
            xb = X_va[vi:v_end].unsqueeze(1).to(device)
            yb = Y_va[vi:v_end].to(device)
            pred = model.forward_data(xb)
            diff = (pred - yb).abs()  # [B, 9, 200]
            for sj, k in enumerate(STATES):
                mae_start[k] += diff[:, sj, START].mean().item() * (v_end - vi)
                mae_interior[k] += diff[:, sj, INTERIOR].mean().item() * (v_end - vi)
                mae_end[k] += diff[:, sj, END].mean().item() * (v_end - vi)
            n += v_end - vi

    ratios = {}
    for k in STATES:
        s = mae_start[k] / max(n, 1)
        i = mae_interior[k] / max(n, 1)
        e = mae_end[k] / max(n, 1)
        ratios[k] = {
            "mae_start": s, "mae_interior": i, "mae_end": e,
            "start_interior": s / i if i > 1e-30 else None,
            "end_interior": e / i if i > 1e-30 else None,
        }
    return ratios


def decoder_health(model, X_va, n_batches=8):
    from phase_5_corrected.derivatives_v2 import compute_du_dt_per_channel
    model.eval()
    T_Q = torch.linspace(0.5, 9.5, 50, device=device)
    du_stds, var_vals = [], []
    with torch.enable_grad():
        for b in range(n_batches):
            i0 = b * 16
            i1 = min(i0 + 16, X_va.shape[0])
            if i0 >= X_va.shape[0]: break
            xb = X_va[i0:i1].unsqueeze(1).to(device)
            B = xb.shape[0]
            tq = T_Q.unsqueeze(0).expand(B, -1)
            try:
                u_hat, du_dt = compute_du_dt_per_channel(model, xb, tq)
                du_stds.append(du_dt.std().item())
                var_vals.append(u_hat.var(dim=-1).mean().item())
            except Exception as e:
                print(f"  health fail: {e}")
    return {"du_dt_std": float(np.nanmean(du_stds)),
            "u_hat_variance": float(np.nanmean(var_vals))}


def train_seed_causal(seed):
    print(f"\n{'='*70}\nStep 5.C — Causal Mask, Seed {seed}\n{'='*70}")

    # Load data first (no randomness)
    X_tr, Y_tr, X_va, Y_va, norm_all = load_data()
    norm_rho = norm_all["rho_ext"]

    # Build model with seed
    model = build_causal_model(seed)

    # Re-seed AFTER model build (for data shuffling)
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"Train: {X_tr.shape[0]} | Val: {X_va.shape[0]}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    from physics_informed.losses import load_composite_loss
    from training.train_sc_fno import get_kin, get_th
    from phase_5_corrected.losses_v2 import compute_composite_loss_v2
    from phase_5_corrected.residuals_v2 import make_rho_ext_fn

    loss_fn = load_composite_loss()
    kin, th = get_kin(), get_th()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    T_Q = torch.linspace(0.5, 9.5, 50, device=device)
    BATCH = 16; MAX_EPOCHS = 300; PATIENCE = 100; MIN_EPOCHS = 100

    best_val = float("inf"); best_ep = 0
    epoch_since_best = 0
    history = []; peak_vram = 0.0
    stop_ep = None; stop_reason = None
    t_start = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        lam_phys = fixed_ramp(epoch)
        perm = torch.randperm(X_tr.shape[0])
        Xp, Yp = X_tr[perm], Y_tr[perm]

        sums = {"L_data": 0, "L_phys": 0, "L_ic": 0, "L_data_decoder": 0, "L_total": 0}
        n_batches = 0

        for bi in range(0, X_tr.shape[0], BATCH):
            b_end = min(bi + BATCH, X_tr.shape[0])
            xb = Xp[bi:b_end].unsqueeze(1).to(device)
            yb = Yp[bi:b_end].to(device)
            B = xb.shape[0]
            tq = T_Q.unsqueeze(0).expand(B, -1)
            rho_fn = make_rho_ext_fn(xb[:, 0, :], norm_rho["mean"], norm_rho["std"])

            optimizer.zero_grad()
            losses = compute_composite_loss_v2(
                model, xb, yb, tq, rho_fn, kin, th, loss_fn,
                lambda_phys=lam_phys, lambda_dec=1.0, lambda_ic=1.0,
            )
            if not torch.isfinite(losses["L_total"]):
                continue
            losses["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in sums:
                if k in losses: sums[k] += float(losses[k].detach())
            n_batches += 1

        scheduler.step()
        for k in sums: sums[k] /= max(n_batches, 1)

        # Validation
        model.eval()
        val_dl = 0.0; n_val = 0
        with torch.no_grad():
            for vi in range(0, X_va.shape[0], BATCH):
                v_end = min(vi + BATCH, X_va.shape[0])
                xb = X_va[vi:v_end].unsqueeze(1).to(device)
                yb = Y_va[vi:v_end].to(device)
                pred = model.forward_data(xb)
                val_dl += float(((pred - yb) ** 2).mean())
                n_val += 1
        val_dl /= max(n_val, 1)

        vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        peak_vram = max(peak_vram, vram)

        if epoch % 10 == 0 or epoch < 5:
            print(f"  ep {epoch:3d}: data={sums['L_data']:.5f} phys={sums['L_phys']:.3e} "
                  f"dec={sums['L_data_decoder']:.4f} lam={lam_phys:.4f} val={val_dl:.5f}")

        history.append({"epoch": epoch, "lambda_phys": lam_phys,
                        **{k: sums[k] for k in sums}, "val_data_loss": val_dl,
                        "peak_vram_MB": vram})

        if val_dl < best_val:
            best_val = val_dl; best_ep = epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "val_loss": val_dl, "epoch": epoch, "seed": seed,
                        "variant": "causal_mask"},
                       CKPT / f"seed_{seed}_causal_best.pt")

        if epoch >= MIN_EPOCHS:
            if val_dl >= best_val - 1e-6:
                epoch_since_best += 1
            else:
                epoch_since_best = 0
            if epoch_since_best >= PATIENCE:
                stop_ep = epoch
                stop_reason = f"patience={PATIENCE} after epoch {MIN_EPOCHS}"
                print(f"  EARLY STOP at ep {epoch}: {stop_reason}")
                break

    wall = time.time() - t_start

    torch.save({"model_state_dict": model.state_dict(),
                "val_loss": best_val, "epoch": stop_ep or MAX_EPOCHS,
                "seed": seed, "variant": "causal_mask"},
               CKPT / f"seed_{seed}_causal_final.pt")

    # Post-training checks
    health = decoder_health(model, X_va, n_batches=8)
    boundary = boundary_check(model, X_va, Y_va, n_scenarios=20)

    summary = {
        "seed": seed,
        "variant": "causal_mask",
        "best_val_data_loss": best_val, "best_epoch": best_ep,
        "final_epoch": stop_ep or MAX_EPOCHS,
        "stop_reason": stop_reason or "max_epochs",
        "wall_time_s": wall, "peak_vram_MB": peak_vram,
        "post_training_decoder_health": health,
        "boundary_ratios": boundary,
    }
    json.dump(summary, open(OUT / f"seed_{seed}_causal_summary.json","w"), indent=2)
    pd.DataFrame(history).to_csv(OUT / f"seed_{seed}_causal_training_log.csv", index=False)

    print(f"\n{'='*70}")
    print(f"SEED {seed} CAUSAL COMPLETE:")
    print(f"  best_val = {best_val:.6f} @ ep {best_ep}")
    print(f"  final_ep = {stop_ep or MAX_EPOCHS}, wall = {wall/60:.1f} min")
    print(f"  peak VRAM = {peak_vram:.0f} MB")
    print(f"  decoder health: du_dt_std={health['du_dt_std']:.3f}, uvar={health['u_hat_variance']:.3f}")
    print(f"  boundary n:  start/interior = {boundary['n']['start_interior']:.2f}x, "
          f"end/interior = {boundary['n']['end_interior']:.2f}x")
    print(f"  boundary Tc: start/interior = {boundary['Tc']['start_interior']:.2f}x")
    print(f"  ⚠ target B_n < 4.0x: "
          f"{'PASS' if boundary['n']['start_interior'] < 4.0 else 'FAIL'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    train_seed_causal(seed)
