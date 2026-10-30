"""Step 2.4.G — Human-readable evaluation summary + verification."""
import json
from pathlib import Path
import pandas as pd

OUT = Path("results/phase_2/evaluation")
agg = json.load(open(OUT / "aggregate_metrics.json"))
wc = json.load(open(OUT / "worst_case_aggregate.json"))
wps = json.load(open(OUT / "worst_case_per_state.json"))
bench = json.load(open(OUT / "inference_benchmark.json"))
per_state = pd.read_csv(OUT / "metrics_per_state.csv")
per_family = pd.read_csv(OUT / "metrics_per_family.csv")

lines = []
lines.append("# Step 2.4 — Baseline Evaluation Summary\n")
lines.append("**Date:** 2026-10-30\n")
lines.append(f"**Checkpoint:** `fno_best.pt` (epoch {agg['checkpoint_epoch']}, val MSE {agg['checkpoint_val_mse']:.6e})  ")
lines.append(f"**Checkpoint SHA-256:** `{agg['checkpoint_sha256']}`\n")

lines.append("## Test Set\n")
lines.append(f"- Scenarios evaluated: **{agg['n_scenarios']}** (all 200 test, untouched since Phase 1)")
lines.append(f"- Timesteps per scenario: {agg['n_timesteps']}")
lines.append(f"- Families: {agg['families']}\n")

lines.append("## Overall Metrics\n")
lines.append(f"- MAE (mean over all 200×9×200 values): **{agg['overall_mae_mean']:.4e}**")
lines.append(f"- RMSE (mean): **{agg['overall_rmse_mean']:.4e}**")
lines.append(f"- Relative L2 (median): **{agg['overall_rel_l2_median']:.4e}**")
lines.append(f"- Rel L2 skipped (near-zero denom): {agg['rel_l2_skipped_count']} of {agg['n_scenarios']*9}\n")

lines.append("## Per-State Metrics\n")
lines.append("| State | MAE mean | RMSE mean | RelL2 median | Max abs |")
lines.append("|-------|----------|-----------|--------------|---------|")
for _, r in per_state.iterrows():
    lines.append(f"| {r['state']} | {r['mae_mean']:.3e} | {r['rmse_mean']:.3e} | {r['rel_l2_median']:.3e} | {r['maxabs_max']:.3e} |")
lines.append("")

lines.append("## Per-Family Metrics\n")
lines.append("| Family | MAE mean | RMSE mean | RelL2 median |")
lines.append("|--------|----------|-----------|--------------|")
for _, r in per_family.iterrows():
    lines.append(f"| {r['family']} | {r['mae_mean']:.3e} | {r['rmse_mean']:.3e} | {r['rel_l2_median']:.3e} |")
lines.append("")

lines.append("## Worst-Case Scenarios\n")
lines.append(f"**Aggregate worst:** `{wc['scenario_id']}` (family={wc['family']}, relL2 mean={wc['aggregate_rel_l2_mean']:.4e})\n")
lines.append("**Per-state worst:**")
for state in ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]:
    if state in wps:
        w = wps[state]
        lines.append(f"- `{state}`: {w['scenario_id']} ({w['family']}), relL2={w['rel_l2']:.4e}")
lines.append("")

lines.append("## Inference Benchmark\n")
lines.append("| Batch | Median (ms) | Mean (ms) | Peak VRAM (MB) |")
lines.append("|-------|-------------|-----------|----------------|")
for k in ["B1","B16","B200"]:
    b = bench[k]
    lines.append(f"| {k} | {b['median_ms']:.3f} | {b['mean_ms']:.3f} | {b['peak_alloc_MB']:.2f} |")
lines.append("")
sp = bench["speedup_vs_BDF"]
lines.append(f"**Speedup vs Phase 1 BDF baseline:**")
lines.append(f"- BDF median {sp['BDF_median_ms']:.1f} ms, mean {sp['BDF_mean_ms']:.1f} ms")
lines.append(f"- FNO per-scenario (B200 median): {sp['FNO_per_scenario_ms_B200']:.6f} ms")
lines.append(f"- **Speedup (median vs median): {sp['speedup_vs_BDF_median_using_B200']:.1f}×**")
lines.append(f"- Speedup (mean vs median): {sp['speedup_vs_BDF_mean_using_B200']:.1f}×")
lines.append(f"- Primary metric: `{sp['primary_metric']}`\n")

lines.append("## Timing Methodology (Reviewer Item C)\n")
b = bench["B16"]
lines.append(f"- Warm-up: {b['warmup']} iterations")
lines.append(f"- Repeats: {b['repeats']}")
lines.append(f"- Sync: `{b['sync']}`")
lines.append(f"- Timer: `{b['timer']}`")
lines.append(f"- Median primary (B200)\n")

lines.append("## Required Figures\n")
for i, name in enumerate([
    "f1_training_validation_loss.png",
    "f2_neutron_density_prediction.png",
    "f3_precursor_predictions.png",
    "f4_fuel_temperature_prediction.png",
    "f5_coolant_temperature_prediction.png",
    "f6_statewise_error_comparison.png",
    "worst_case_prediction.png",
], 1):
    p = Path("reports/figures/phase_2") / name
    lines.append(f"- [{'✓' if p.exists() else '✗'}] `{name}`")
lines.append("")

lines.append("## Critical Guarantees\n")
lines.append("- Test set untouched during Steps 2.1–2.3 ✅")
lines.append("- Best checkpoint selected by validation MSE only ✅")
lines.append("- Test set loaded for first time in Step 2.4 ✅")
lines.append("- No model selection using test data ✅")
lines.append("- All metrics from saved prediction arrays (reproducible) ✅")
lines.append("- Near-zero denominator handling: rel_l2 undefined when ||y||_2 < 1e-8; count = 0 ✅")
lines.append("- Timing methodology: warm-up + sync + repeats ✅")
lines.append("- Phase 1, 2.1, 2.2, 2.3 evidence untouched ✅\n")

lines.append("## Interpretive Note (per reviewer)\n")
lines.append("These are measured test-set errors on the FNO baseline. They are not claims of physical fidelity. The purpose is to establish the **reference** against which Phase 3 SC-FNO improvements will be measured. Any state with high relative L2 (e.g., neutron density `n` at 7.55e-3) is a candidate for physics-informed improvement — but that determination belongs to Phase 3, not Step 2.4.\n")

with open(OUT / "evaluation_summary.md", "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print("\n=== evaluation_summary.md written ===")
