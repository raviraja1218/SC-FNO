"""Step 5.C — Multi-seed causal summary vs Phase 5.B non-causal."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

SEEDS = [42, 123, 456, 789, 1012]
SEED_DIR = Path("results/phase_5/step_5_C/seeds")
BOUNDARY_DIR = Path("results/phase_5/step_5_C/boundary_analysis")

rows = []
for seed in SEEDS:
    p = SEED_DIR / f"seed_{seed}_causal_summary.json"
    if not p.exists():
        print(f"[SKIP] seed {seed}: missing")
        continue
    s = json.load(open(p))
    b = s["boundary_ratios"]
    rows.append({
        "seed": seed,
        "best_val": s["best_val_data_loss"],
        "best_epoch": s["best_epoch"],
        "B_n": b["n"]["start_interior"],
        "B_Tf": b["Tf"]["start_interior"],
        "B_Tc": b["Tc"]["start_interior"],
        "du_dt_std": s["post_training_decoder_health"]["du_dt_std"],
        "u_hat_var": s["post_training_decoder_health"]["u_hat_variance"],
    })

df = pd.DataFrame(rows)
df.to_csv(BOUNDARY_DIR / "causal_multi_seed.csv", index=False)

if len(df) > 0:
    print("="*70)
    print("Step 5.C — 5-seed causal mask summary")
    print("="*70)
    print(df.to_string(index=False))
    print()
    
    # Pass/fail criteria
    n_pass = (df["B_n"] < 4.0).sum()
    tf_pass = (df["B_Tf"] < 4.0).sum()
    tc_pass = (df["B_Tc"] < 4.0).sum()
    health_pass = ((df["du_dt_std"] > 0.05) & (df["u_hat_var"] > 0.3)).sum()
    
    print(f"B_n < 4.0:  {n_pass}/5 seeds")
    print(f"B_Tf < 4.0: {tf_pass}/5 seeds")
    print(f"B_Tc < 4.0: {tc_pass}/5 seeds")
    print(f"Decoder health pass: {health_pass}/5 seeds")
    print()
    
    # Non-causal baseline
    nc = json.load(open(BOUNDARY_DIR / "phase5b_noncausal_baseline.json"))
    nc_b = nc["boundary_ratios"]
    print(f"Non-causal baseline (seed 42):")
    print(f"  B_n = {nc_b['n']['start_interior']:.2f}x")
    print(f"  B_Tf = {nc_b['Tf']['start_interior']:.2f}x")
    print(f"  B_Tc = {nc_b['Tc']['start_interior']:.2f}x")
    print()
    
    # Causal mean
    print(f"Causal (5-seed mean):")
    print(f"  B_n = {df['B_n'].mean():.2f}x ± {df['B_n'].std():.2f}")
    print(f"  B_Tf = {df['B_Tf'].mean():.2f}x ± {df['B_Tf'].std():.2f}")
    print(f"  B_Tc = {df['B_Tc'].mean():.2f}x ± {df['B_Tc'].std():.2f}")
    print()
    print(f"  best_val = {df['best_val'].mean():.6f} ± {df['best_val'].std():.6f}")
    print(f"  du_dt std = {df['du_dt_std'].mean():.3f} ± {df['du_dt_std'].std():.3f}")
    print(f"  u_hat var = {df['u_hat_var'].mean():.3f} ± {df['u_hat_var'].std():.3f}")

    # Save aggregate
    agg = {
        "n_seeds": len(df),
        "B_n_mean": float(df["B_n"].mean()),
        "B_n_std": float(df["B_n"].std(ddof=1)),
        "B_Tf_mean": float(df["B_Tf"].mean()),
        "B_Tc_mean": float(df["B_Tc"].mean()),
        "n_seeds_B_n_pass": int(n_pass),
        "best_val_mean": float(df["best_val"].mean()),
        "best_val_std": float(df["best_val"].std(ddof=1)),
        "du_dt_std_mean": float(df["du_dt_std"].mean()),
        "u_hat_var_mean": float(df["u_hat_var"].mean()),
        "u_hat_var_min": float(df["u_hat_var"].min()),
        "noncausal_B_n": float(nc_b["n"]["start_interior"]),
        "B_n_improvement_pct": (float(nc_b["n"]["start_interior"]) - float(df["B_n"].mean())) / float(nc_b["n"]["start_interior"]) * 100,
    }
    json.dump(agg, open(BOUNDARY_DIR / "causal_multi_seed_aggregate.json", "w"), indent=2)
    print(f"\nSaved aggregate: causal_multi_seed_aggregate.json")
