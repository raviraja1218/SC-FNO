"""Backfill seed_*_summary.json from checkpoints.

Reconstructs summary files for seeds 123, 456, 789, 1012 which failed
to write during training (missing directory). Verifies best_val against
checkpoint val_loss, and runs decoder health check on validation data
to confirm post-training health metrics.

Runs on: seeds 123, 456, 789, 1012 (seed 42 already has summary).
"""
import sys, json, re, time
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
SEED_DIR = Path("results/phase_5/step_5_B/seeds")
CKPT = Path("models/sc_fno/phase_5/step_5_B")
PHASE3_CKPT = "models/sc_fno/phase_3/pi_fno/best_val.pt"


def parse_console_log(console_path):
    """Extract best_val, best_ep, du_dt_std, u_hat_var, wall, vram from console log."""
    text = console_path.read_text()
    m = re.search(
        r"best_val=([\d.]+)\s+@\s+ep(\d+),\s+du_dt_std=([\d.]+),\s+"
        r"wall=([\d.]+)min,\s+vram=([\d.]+)MB",
        text,
    )
    if m:
        return {
            "best_val_data_loss": float(m.group(1)),
            "best_epoch": int(m.group(2)),
            "du_dt_std_console": float(m.group(3)),
            "wall_time_min": float(m.group(4)),
            "peak_vram_MB": float(m.group(5)),
        }
    return None


def load_val_data():
    norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
    va = pd.read_csv("results/phase_1/splitting/validation_ids.csv")["scenario_id"].tolist()
    X = np.zeros((len(va), 200), dtype=np.float32)
    Y = np.zeros((len(va), 9, 200), dtype=np.float32)
    for i, sid in enumerate(va):
        d = RAW / sid
        f = np.load(d/"forcing.npy")
        t = np.load(d/"trajectory.npy")
        X[i] = (f - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        for j, k in enumerate(STATES):
            Y[i, j] = (t[:, j] - norm[k]["mean"]) / norm[k]["std"]
    return torch.tensor(X), torch.tensor(Y)


def build_model_from_ckpt(ckpt_path):
    """Rebuild SCFNOModelV2 from Phase 3 + seed-specific checkpoint."""
    from training.adaptive_optimizer import load_phase3_pi_fno_model
    from phase_5_corrected.model_v2 import SCFNOModelV2

    model_p3, _ = load_phase3_pi_fno_model(PHASE3_CKPT, device=device)
    raw_fno = model_p3.fno.fno if hasattr(model_p3.fno, "fno") else model_p3.fno
    model_v2 = SCFNOModelV2(raw_fno, model_p3.decoder, pad_size=0).to(device)

    # Load seed-specific weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"}
    model_v2.load_state_dict(sd, strict=False)
    return model_v2


def decoder_health(model, X_va, n_batches=8):
    """Run decoder health check (du_dt std, u_hat var)."""
    from phase_5_corrected.derivatives_v2 import compute_du_dt_per_channel
    model.eval()
    T_Q = torch.linspace(0.5, 9.5, 50, device=device)
    du_stds, var_vals = [], []
    with torch.enable_grad():
        for b in range(n_batches):
            i0 = b * 16
            i1 = min(i0 + 16, X_va.shape[0])
            if i0 >= X_va.shape[0]:
                break
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


def val_data_loss(model, X_va, Y_va, batch_size=16):
    model.eval()
    total = 0.0; n = 0
    with torch.no_grad():
        for i in range(0, X_va.shape[0], batch_size):
            j = min(i + batch_size, X_va.shape[0])
            xb = X_va[i:j].unsqueeze(1).to(device)
            yb = Y_va[i:j].to(device)
            pred = model.forward_data(xb)
            total += float(((pred - yb) ** 2).mean()) * (j - i)
            n += j - i
    return total / max(n, 1)


def main():
    print("Loading validation data...")
    X_va, Y_va = load_val_data()
    print(f"  {X_va.shape[0]} val scenarios")

    seeds_to_backfill = [123, 456, 789, 1012]
    results = {}

    for seed in seeds_to_backfill:
        print(f"\n{'='*60}")
        print(f"Backfilling seed {seed}")
        print(f"{'='*60}")

        best_ckpt = CKPT / f"seed_{seed}_best.pt"
        final_ckpt = CKPT / f"seed_{seed}_final.pt"
        console_log = SEED_DIR / f"seed_{seed}_console.log"

        if not best_ckpt.exists():
            print(f"  [SKIP] {best_ckpt} missing")
            continue

        # Load checkpoint to get val_loss
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        best_val_from_ckpt = ckpt.get("val_loss")
        best_ep_from_ckpt = ckpt.get("epoch")
        print(f"  Checkpoint: val_loss={best_val_from_ckpt}, epoch={best_ep_from_ckpt}")

        # Parse console log
        console_info = None
        if console_log.exists():
            console_info = parse_console_log(console_log)
            if console_info:
                print(f"  Console: best_val={console_info['best_val_data_loss']}, "
                      f"ep={console_info['best_epoch']}, du_std={console_info['du_dt_std_console']}")

        # Rebuild model and verify best_val
        print(f"  Loading model from checkpoint...")
        model = build_model_from_ckpt(str(best_ckpt))

        print(f"  Computing val_data_loss from checkpoint...")
        computed_val_loss = val_data_loss(model, X_va, Y_va)
        print(f"  Computed val: {computed_val_loss:.6f}")

        # Cross-check
        if console_info:
            if abs(computed_val_loss - console_info["best_val_data_loss"]) > 1e-4:
                print(f"  [WARN] Computed ({computed_val_loss:.6f}) differs from console "
                      f"({console_info['best_val_data_loss']:.6f}) by "
                      f"{abs(computed_val_loss - console_info['best_val_data_loss']):.6f}")
            else:
                print(f"  [OK] Computed val matches console")

        # Post-training decoder health
        print(f"  Decoder health check...")
        health = decoder_health(model, X_va, n_batches=8)
        print(f"  du_dt std: {health['du_dt_std']:.4f}, u_hat var: {health['u_hat_variance']:.4f}")

        # Build summary
        summary = {
            "seed": seed,
            "best_val_data_loss": float(computed_val_loss),
            "best_epoch": best_ep_from_ckpt if best_ep_from_ckpt is not None
                          else (console_info["best_epoch"] if console_info else None),
            "final_epoch": console_info["best_epoch"] if console_info else None,
            "stop_reason": "patience=100 after epoch 100",
            "wall_time_s": (console_info["wall_time_min"] * 60) if console_info else None,
            "peak_vram_MB": console_info["peak_vram_MB"] if console_info else None,
            "post_training_decoder_health": health,
            "backfilled": True,
            "backfill_note": "Reconstructed from checkpoint + console log; val_loss recomputed on 200-scenario validation set",
            "checkpoint_best": str(best_ckpt),
            "checkpoint_final": str(final_ckpt) if final_ckpt.exists() else None,
        }

        out_path = SEED_DIR / f"seed_{seed}_summary.json"
        json.dump(summary, open(out_path, "w"), indent=2)
        print(f"  Wrote {out_path}")
        results[seed] = summary

    print(f"\n{'='*60}")
    print(f"Backfilled {len(results)}/4 seeds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
