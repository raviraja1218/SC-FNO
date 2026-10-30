"""Audit item 4 — Verify aggregate Rel L2 population (fixed version)."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("results/phase_2/supplemental_audit")
scen = pd.read_csv("results/phase_2/evaluation/scenario_level_metrics.csv")
print(f"scenario_level_metrics.csv rows: {len(scen)}")

rel_l2 = scen["rel_l2"].dropna()

median_over_1800 = float(rel_l2.median())
mean_over_1800 = float(rel_l2.mean())

scen_means = scen.groupby("scenario_id")["rel_l2"].mean()
median_over_200_scen_means = float(scen_means.median())

reported = 9.232e-4
candidates = {
    "A_median_over_1800_pairs": median_over_1800,
    "B_median_over_200_scen_means": median_over_200_scen_means,
    "C_mean_over_1800_pairs": mean_over_1800,
}

print("\n=== Aggregate Rel L2 candidates ===")
for k, v in candidates.items():
    diff = abs(v - reported) / reported
    print(f"  {k}: {v:.6e}  (diff from reported {reported:.6e}: {diff:.4%})")

best_match = min(candidates.items(), key=lambda kv: abs(kv[1] - reported))
best_name, best_value = best_match
print(f"\nBest match: {best_name} = {best_value:.6e}")

# Build sub-dicts BEFORE main dict
population_description = {
    "A_median_over_1800_pairs": "Median of all 200 scenarios × 9 states = 1800 per-pair Rel L2 values",
    "B_median_over_200_scen_means": "Median over 200 scenarios of the per-scenario mean Rel L2 across 9 states",
    "C_mean_over_1800_pairs": "Mean of all 1800 per-pair Rel L2 values",
}

recommendation = (
    f"Manuscript should state: 'Overall Rel L2 median = {best_value:.4e}, "
    f"computed as {population_description[best_name]}. "
    f"Per-state and per-family breakdowns preserved (Table 4 + per_family_metrics_table.md).'"
)

result = {
    "reported_value": reported,
    "candidates": candidates,
    "primary_population": best_name,
    "primary_value": best_value,
    "population_description": population_description,
    "recommendation": recommendation,
}
json.dump(result, open(OUT / "04_aggregate_metric_definition_data.json", "w"), indent=2)
print(f"\n=== Saved to {OUT}/04_aggregate_metric_definition_data.json ===")
print(f"\n{recommendation}")
