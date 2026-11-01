"""Step 4A.3.R.2 — Paired bootstrap statistical analysis.

Addresses reviewer point 2: quantify the paired distribution of data accuracy
differences between Phase 3 and Phase 4A.1, both overall and per family.

Uses existing per_scenario_comparison.csv — no new measurements.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results/phase_4/phase_4a/residual_analysis")
df = pd.read_csv(OUT / "per_scenario_comparison.csv")

# df has: scenario_id, family, phase3_rel_l2, phase4a1_rel_l2, delta, pct_change
# delta = phase3 - phase4a1 (positive = Phase 4A.1 better)
# Confirm column ordering
print("Loaded per_scenario_comparison.csv:", df.shape)
print("Columns:", df.columns.tolist())
print()

def paired_stats(df_subset, label):
    """Compute paired statistics for a scenario subset."""
    deltas = df_subset["delta"].values  # phase3 - phase4a1 (positive = 4A.1 better)
    n = len(deltas)
    wins = int((deltas > 0).sum())    # Phase 4A.1 better
    losses = int((deltas < 0).sum())  # Phase 3 better
    ties = int((deltas == 0).sum())

    # Bootstrap 95% CI for mean and median
    n_boot = 10000
    rng = np.random.default_rng(42)
    boot_means = np.zeros(n_boot)
    boot_medians = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[i] = sample.mean()
        boot_medians[i] = np.median(sample)

    return {
        "label": label,
        "n": n,
        "mean_paired_delta": float(deltas.mean()),
        "median_paired_delta": float(np.median(deltas)),
        "std_paired_delta": float(deltas.std(ddof=1)),
        "mean_paired_delta_ci95": [float(np.percentile(boot_means, 2.5)),
                                    float(np.percentile(boot_means, 97.5))],
        "median_paired_delta_ci95": [float(np.percentile(boot_medians, 2.5)),
                                      float(np.percentile(boot_medians, 97.5))],
        "wins_phase4a1": wins,
        "losses_phase3": losses,
        "ties": ties,
        "win_rate_phase4a1": wins / n,
        "win_rate_phase3": losses / n,
        # For context: rel L2 comparison
        "phase3_rel_l2_mean": float(df_subset["phase3_rel_l2"].mean()),
        "phase4a1_rel_l2_mean": float(df_subset["phase4a1_rel_l2"].mean()),
        "mean_rel_l2_pct": float((df_subset["phase4a1_rel_l2"].mean() - df_subset["phase3_rel_l2"].mean()) / df_subset["phase3_rel_l2"].mean() * 100),
    }

# Overall
overall = paired_stats(df, "overall")

# Per family
by_family = {}
for fam in sorted(df["family"].unique()):
    subset = df[df["family"] == fam]
    by_family[fam] = paired_stats(subset, fam)

result = {
    "description": "Paired bootstrap analysis of relative L2 differences",
    "sign_convention": "delta = phase3_rel_l2 - phase4a1_rel_l2. Positive = Phase 4A.1 better",
    "n_bootstrap_samples": 10000,
    "overall": overall,
    "per_family": by_family,
}
with open(OUT / "paired_bootstrap_stats.json", "w") as f:
    json.dump(result, f, indent=2)

# Print summary
print("=" * 80)
print("PAIRED BOOTSTRAP STATISTICAL ANALYSIS")
print("=" * 80)
print(f"Sign: delta = Phase3_relL2 − Phase4A.1_relL2  (positive = 4A.1 better)")
print()
print(f"{'Group':<15s} {'n':>4s} {'mean_Δ':>12s} {'95% CI':>30s} {'win_rate':>10s}")
print(f"{'':15s} {'':4s} {'median_Δ':>12s} {'95% CI':>30s} {'':>10s}")
print("-" * 80)
for group in [overall] + list(by_family.values()):
    mean_ci = group["mean_paired_delta_ci95"]
    med_ci = group["median_paired_delta_ci95"]
    print(f"{group['label']:<15s} {group['n']:>4d} {group['mean_paired_delta']:>+12.4e} "
          f"[{mean_ci[0]:>+11.3e},{mean_ci[1]:>+11.3e}] {group['win_rate_phase4a1']:>9.1%}")
    print(f"{'':15s} {'':4s} {group['median_paired_delta']:>+12.4e} "
          f"[{med_ci[0]:>+11.3e},{med_ci[1]:>+11.3e}]")
    print()

# Interpretation
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(f"Overall mean delta: {overall['mean_paired_delta']:+.4e}")
print(f"  If 95% CI crosses zero → mean improvement NOT statistically significant")
print(f"  CI: [{overall['mean_paired_delta_ci95'][0]:+.3e}, {overall['mean_paired_delta_ci95'][1]:+.3e}]")
print(f"  Zero in CI? {overall['mean_paired_delta_ci95'][0] <= 0 <= overall['mean_paired_delta_ci95'][1]}")
print()
print(f"Overall median delta: {overall['median_paired_delta']:+.4e}")
print(f"  CI: [{overall['median_paired_delta_ci95'][0]:+.3e}, {overall['median_paired_delta_ci95'][1]:+.3e}]")
print(f"  Zero in CI? {overall['median_paired_delta_ci95'][0] <= 0 <= overall['median_paired_delta_ci95'][1]}")
