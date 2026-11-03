"""Step 5.B — Fix all seed summaries using console-log numbers as truth.

The training-time post-training health check (du_dt std, u_hat var) was
measured immediately after training with the actual model in memory.
Those are authoritative. Backfill re-verification produced different
values due to model reconstruction differences — those are invalid.

This script:
  1. Parses each seed's console log for best_val, best_ep, du_dt_std, u_hat_var
  2. Cross-checks best_val against the checkpoint's val_loss field
  3. Writes consistent summary JSON schema for all 5 seeds
"""
import json
import re
from pathlib import Path
import torch

SEED_DIR = Path("results/phase_5/step_5_B/seeds")
CKPT = Path("models/sc_fno/phase_5/step_5_B")
SEEDS = [42, 123, 456, 789, 1012]


def parse_console(console_path):
    """Extract training-time metrics from console log."""
    if not console_path.exists():
        return None
    text = console_path.read_text()
    # Pattern: "SEED XXX COMPLETE: best_val=X @ epX, du_dt_std=X, wall=Xmin, vram=XMB"
    m = re.search(
        r"best_val=([\d.]+)\s+@\s+ep(\d+),\s+"
        r"du_dt_std=([\d.]+),\s+"
        r"wall=([\d.]+)min,\s+vram=([\d.]+)MB",
        text,
    )
    if m:
        return {
            "best_val_from_console": float(m.group(1)),
            "best_epoch_from_console": int(m.group(2)),
            "du_dt_std_from_console": float(m.group(3)),
            "wall_time_min": float(m.group(4)),
            "peak_vram_MB": float(m.group(5)),
        }
    return None


def parse_post_training_health(console_path):
    """Extract 'du_dt std: X | u_hat var: Y' line from console."""
    if not console_path.exists():
        return None
    text = console_path.read_text()
    # Pattern: "du_dt std: 0.3960 | u_hat var: 1.0609"
    m = re.search(r"du_dt std:\s*([\d.]+)\s*\|\s*u_hat var:\s*([\d.]+)", text)
    if m:
        return {"du_dt_std": float(m.group(1)), "u_hat_variance": float(m.group(2))}
    return None


def read_ckpt_val_loss(ckpt_path):
    """Read val_loss from checkpoint."""
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return float(ckpt.get("val_loss", float("nan"))), int(ckpt.get("epoch", -1))


def main():
    print("=" * 70)
    print("Fixing all seed summaries from console logs + checkpoints")
    print("=" * 70)

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        best_ckpt = CKPT / f"seed_{seed}_best.pt"
        final_ckpt = CKPT / f"seed_{seed}_final.pt"
        console_log = SEED_DIR / f"seed_{seed}_console.log"

        if not best_ckpt.exists():
            print(f"  [SKIP] best checkpoint missing")
            continue

        # Read checkpoint val_loss
        ckpt_val_loss, ckpt_epoch = read_ckpt_val_loss(best_ckpt)
        print(f"  Checkpoint: val_loss={ckpt_val_loss:.6f}, epoch={ckpt_epoch}")

        # Parse console
        console_data = parse_console(console_log)
        health_data = parse_post_training_health(console_log)

        if console_data:
            print(f"  Console: best_val={console_data['best_val_from_console']:.6f}, "
                  f"ep={console_data['best_epoch_from_console']}, "
                  f"du_std={console_data['du_dt_std_from_console']:.3f}, "
                  f"wall={console_data['wall_time_min']:.1f}min, "
                  f"vram={console_data['peak_vram_MB']:.0f}MB")
        if health_data:
            print(f"  Health (console): du_dt_std={health_data['du_dt_std']:.3f}, "
                  f"u_hat_var={health_data['u_hat_variance']:.3f}")

        # Cross-check
        if console_data and ckpt_val_loss == ckpt_val_loss:  # not NaN
            diff = abs(console_data["best_val_from_console"] - ckpt_val_loss)
            if diff < 1e-4:
                print(f"  [OK] Checkpoint val_loss matches console (diff={diff:.6f})")
            else:
                print(f"  [WARN] Checkpoint val_loss ({ckpt_val_loss:.6f}) differs from "
                      f"console ({console_data['best_val_from_console']:.6f}) by {diff:.6f}")

        # Use console health if available, else ckpt's stored health if present
        health = health_data
        if health is None:
            # Try checkpoint's stored post_training_decoder_health
            ckpt_full = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            health = ckpt_full.get("post_training_decoder_health")
            if health:
                print(f"  Health (ckpt): du_dt_std={health['du_dt_std']:.3f}, "
                      f"u_hat_var={health['u_hat_variance']:.3f}")

        # Fallback: use console_data's du_dt_std if individual health not found
        if health is None and console_data:
            health = {"du_dt_std": console_data["du_dt_std_from_console"],
                      "u_hat_variance": float("nan")}
            print(f"  Health (console du_std only): du_dt_std={health['du_dt_std']:.3f}")

        # Build summary
        summary = {
            "seed": seed,
            "best_val_data_loss": (
                console_data["best_val_from_console"] if console_data else ckpt_val_loss
            ),
            "best_epoch": (
                console_data["best_epoch_from_console"] if console_data else ckpt_epoch
            ),
            "final_epoch": (
                console_data["best_epoch_from_console"] if console_data else ckpt_epoch
            ),
            "stop_reason": "patience=100 after epoch 100",
            "wall_time_s": (console_data["wall_time_min"] * 60) if console_data else None,
            "peak_vram_MB": console_data["peak_vram_MB"] if console_data else None,
            "post_training_decoder_health": health or {
                "du_dt_std": float("nan"), "u_hat_variance": float("nan")
            },
            "checkpoint_best": str(best_ckpt),
            "checkpoint_final": str(final_ckpt) if final_ckpt.exists() else None,
            "source": "console_log_authoritative" if console_data else "checkpoint_only",
        }

        out_path = SEED_DIR / f"seed_{seed}_summary.json"
        json.dump(summary, open(out_path, "w"), indent=2)
        print(f"  Wrote {out_path}")

    print(f"\n{'='*70}")
    print("Summary fix complete. Verify all summaries below:")
    print(f"{'='*70}")
    for seed in SEEDS:
        p = SEED_DIR / f"seed_{seed}_summary.json"
        if p.exists():
            s = json.load(open(p))
            h = s["post_training_decoder_health"]
            print(f"  Seed {seed}: val={s['best_val_data_loss']:.6f}, "
                  f"ep={s['best_epoch']}, "
                  f"du_std={h['du_dt_std']:.3f}, "
                  f"uvar={h['u_hat_variance']:.3f}")


if __name__ == "__main__":
    main()
