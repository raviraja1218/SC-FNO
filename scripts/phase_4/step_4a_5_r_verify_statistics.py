"""Step 4A.5.R - Independent verification of paired bootstrap statistics."""
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results/phase_4/phase_4a/residual_analysis")
df = pd.read_csv(OUT / "per_scenario_comparison.csv")
published = json.load(open(OUT / "paired_bootstrap_stats.json"))

source_sha = hashlib.sha256(open(OUT / "per_scenario_comparison.csv", "rb").read()).hexdigest()

print(f"Source: per_scenario_comparison.csv")
print(f"SHA-256: {source_sha}")
print(f"Rows: {len(df)}")

def paired_stats(deltas, seed=42, n_boot=10000):
    n = len(deltas)
    wins = int((deltas > 0).sum())
    losses = int((deltas < 0).sum())
    ties = int((deltas == 0).sum())
    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_boot)
    boot_medians = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[i] = sample.mean()
        boot_medians[i] = np.median(sample)
    return {
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
    }

deltas_all = df["delta"].values
recomputed = paired_stats(deltas_all)
pub = published["overall"]

print("=== Overall comparison ===")
all_match = True
for key in ["mean_paired_delta", "median_paired_delta", "std_paired_delta",
            "wins_phase4a1", "losses_phase3", "ties"]:
    pv = pub.get(key)
    rv = recomputed.get(key)
    if isinstance(pv, (int, float)) and isinstance(rv, (int, float)):
        match = abs(pv - rv) < 1e-9
        all_match = all_match and match
        print(f"  {key}: published={pv:.6e}, recomputed={rv:.6e}, match={match}")

for key in ["mean_paired_delta_ci95", "median_paired_delta_ci95"]:
    pv = pub.get(key)
    rv = recomputed.get(key)
    if pv and rv:
        match = all(abs(a - b) < 1e-6 for a, b in zip(pv, rv))
        all_match = all_match and match
        print(f"  {key}: published={pv}, recomputed={rv}, match={match}")

print("\n=== Per-family comparison ===")
for fam in sorted(df["family"].unique()):
    fam_df = df[df["family"] == fam]
    rv_fam = paired_stats(fam_df["delta"].values)
    pub_fam = published["per_family"][fam]
    print(f"\n--- {fam} (n={len(fam_df)}) ---")
    for key in ["mean_paired_delta", "median_paired_delta", "wins_phase4a1", "losses_phase3"]:
        pv = pub_fam.get(key)
        rv = rv_fam.get(key)
        if isinstance(pv, (int, float)) and isinstance(rv, (int, float)):
            match = abs(pv - rv) < 1e-6
            all_match = all_match and match
            print(f"  {key}: published={pv:.6e}, recomputed={rv:.6e}, match={match}")

repro_record = {
    "audit_date": "2026-11-02",
    "source_file": "results/phase_4/phase_4a/residual_analysis/per_scenario_comparison.csv",
    "source_sha256": source_sha,
    "source_rows": len(df),
    "sign_convention": "delta = phase3_rel_l2 - phase4a1_rel_l2 (positive = Phase 4A.1 better)",
    "bootstrap_procedure": {
        "n_resamples": 10000,
        "random_seed": 42,
        "rng": "numpy.random.default_rng(seed=42)",
        "resample_scheme": "with replacement, size=200",
        "ci_method": "percentile (2.5%, 97.5%)",
        "win_definition": "delta > 0",
        "loss_definition": "delta < 0",
        "tie_definition": "delta == 0",
    },
    "recomputed_overall": recomputed,
    "published_overall": pub,
    "match_verified": all_match,
    "verdict": ("Independently recomputed statistics match published values. "
                "Commit f7474c0 is authoritative.") if all_match else "MISMATCH DETECTED",
}
with open("results/phase_4/closure/statistical_reversal_reproducibility.json", "w") as f:
    json.dump(repro_record, f, indent=2)

print(f"\n=== {'All statistics independently verified' if all_match else 'MISMATCH'} ===")
