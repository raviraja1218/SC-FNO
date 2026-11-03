"""Step 5.B — Multi-seed summary (validation only)."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

SEEDS = [42, 123, 456, 789, 1012]
OUT = Path("results/phase_5/step_5_B")
SEED_DIR = OUT / "seeds"
MULTI = OUT / "multi_seed"; MULTI.mkdir(parents=True, exist_ok=True)

rows = []
for seed in SEEDS:
    p = SEED_DIR / f"seed_{seed}_summary.json"
    if not p.exists():
        print(f"[SKIP] seed {seed}: summary missing")
        continue
    s = json.load(open(p))
    rows.append({
        "seed": seed,
        "best_val_data_loss": s["best_val_data_loss"],
        "best_epoch": s["best_epoch"],
        "final_epoch": s.get("final_epoch"),
        "wall_time_min": (s.get("wall_time_s") or 0) / 60,
        "peak_vram_MB": s.get("peak_vram_MB", 0),
        "post_du_dt_std": s["post_training_decoder_health"]["du_dt_std"],
        "post_u_hat_variance": s["post_training_decoder_health"]["u_hat_variance"],
    })

df = pd.DataFrame(rows)
df.to_csv(MULTI / "multi_seed_summary.csv", index=False)

if len(df) > 0:
    vals = df["best_val_data_loss"].values
    mean_v = float(vals.mean())
    std_v = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    cv = std_v / mean_v if mean_v > 0 else None
    
    phase3 = 0.066331
    phase4a1 = 0.065438
    
    summary = {
        "n_seeds_completed": len(df),
        "seeds_completed": df["seed"].tolist(),
        "best_val_data_loss_mean": mean_v,
        "best_val_data_loss_std": std_v,
        "best_val_data_loss_min": float(vals.min()),
        "best_val_data_loss_max": float(vals.max()),
        "best_val_data_loss_cv": cv,
        "phase3_baseline": phase3,
        "phase4a1_validation": phase4a1,
        "delta_vs_phase3_pct": (mean_v - phase3) / phase3 * 100,
        "delta_vs_phase4a1_pct": (mean_v - phase4a1) / phase4a1 * 100,
        "decoder_health_all_pass": bool(
            (df["post_du_dt_std"] > 0.05).all() and
            (df["post_u_hat_variance"] > 0.3).all()
        ),
        "peak_vram_max_MB": float(df["peak_vram_MB"].max()),
        "wall_time_total_min": float(df["wall_time_min"].sum()),
    }
    json.dump(summary, open(MULTI / "multi_seed_aggregate.json","w"), indent=2)
    
    print("\n" + "="*70)
    print("MULTI-SEED SUMMARY (5 seeds)")
    print("="*70)
    print(df.to_string(index=False))
    print("\n" + "="*70)
    print(f"Mean best_val:  {mean_v:.6f}")
    print(f"Std:            {std_v:.6f}")
    print(f"CV:             {cv*100:.2f}%  (target < 5%)")
    print(f"Delta vs Phase 3 (0.066331):  {summary['delta_vs_phase3_pct']:+.2f}%")
    print(f"Delta vs Phase 4A.1 (0.065438): {summary['delta_vs_phase4a1_pct']:+.2f}%")
    print(f"Decoder health: {'ALL PASS' if summary['decoder_health_all_pass'] else 'SOME FAIL'}")
    print(f"Peak VRAM:      {summary['peak_vram_max_MB']:.0f} MB")
    print(f"Total wall:     {summary['wall_time_total_min']:.1f} min")
    print("="*70)
    print(f"\nWritten: {MULTI}/multi_seed_aggregate.json")
else:
    print("No seeds completed")
