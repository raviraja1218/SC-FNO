"""Step 4.6.B — Assemble unified Phase 4 comparison from committed evidence."""
import json
import csv
from pathlib import Path

p4b_metrics = json.load(open("results/phase_4/phase_4b/evaluation/all_models_metrics.json"))
p4b_bootstrap = json.load(open("results/phase_4/phase_4b/evaluation/paired_bootstrap_vs_p2.json"))["vs_P2-FNO"]
corr = json.load(open("results/phase_4/phase_4b/evaluation/physics_revalidation_addendum/physics_revalidation_v2_zero_forcing.json"))

MODELS = {
    "P2-FNO": {"phase": "Phase 2", "role": "data-only baseline"},
    "P3-PI-FNO": {"phase": "Phase 3", "role": "physics baseline"},
    "Phase4A.1": {"phase": "Phase 4A", "role": "adaptive optimizer (SELECTED)"},
    "P4-A": {"phase": "Phase 4B", "role": "PCGrad only (invalid)"},
    "P4-B": {"phase": "Phase 4B", "role": "UnifiedPadding only (invalid)"},
    "P4-C": {"phase": "Phase 4B", "role": "AdaptiveLambda only (invalid)"},
    "P4-D": {"phase": "Phase 4B", "role": "Combined (invalid)"},
}

rows = []
for model_name in ["P2-FNO", "P3-PI-FNO", "Phase4A.1", "P4-A", "P4-B", "P4-C", "P4-D"]:
    m = p4b_metrics[model_name]
    bs = p4b_bootstrap.get(model_name, {})
    corr_phys = corr.get(model_name, {}).get("physics_loss_mean") if model_name != "P2-FNO" else None

    rows.append({
        "model": model_name,
        "phase": MODELS[model_name]["phase"],
        "role": MODELS[model_name]["role"],
        "rel_l2_median": m["rel_l2_overall_median"],
        "rel_l2_mean": m["rel_l2_overall_mean"],
        "physics_loss_per_sum": m.get("physics_loss"),
        "physics_loss_per_channel": corr_phys,
        "ic_loss": m.get("ic_loss"),
        "paired_wins_vs_p2": bs.get("wins"),
        "paired_win_rate": bs.get("win_rate"),
        "median_paired_delta": bs.get("median_delta"),
        "ci95_lo": bs.get("ci95", [None])[0] if bs.get("ci95") else None,
        "ci95_hi": bs.get("ci95", [None, None])[1] if bs.get("ci95") else None,
        "boundary_n_ratio": m["boundary_metrics"].get("n", {}).get("start_interior_ratio"),
    })

out_csv = Path("results/phase_4/final_comparison/unified_phase_4_comparison.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

summary = {
    "phase": "Phase 4",
    "timestamp": "2026-11-02",
    "models_compared": 7,
    "phase_4a": {
        "status": "POSITIVE",
        "selected_model": "Phase4A.1",
        "rel_l2_median": p4b_metrics["Phase4A.1"]["rel_l2_overall_median"],
        "paired_wins_vs_p2": p4b_bootstrap["Phase4A.1"]["wins"],
        "win_rate": p4b_bootstrap["Phase4A.1"]["win_rate"],
        "physics_loss_per_sum": p4b_metrics["Phase4A.1"]["physics_loss"],
        "physics_loss_per_channel": corr["Phase4A.1"]["physics_loss_mean"],
    },
    "phase_4b": {
        "status": "NEGATIVE — training-protocol limitation",
        "variants": ["P4-A", "P4-B", "P4-C", "P4-D"],
        "all_physics_diverged": True,
    },
    "gate_4_verdict": "PHASE_4_LOCKED — MIXED RESULT: PHASE 4A SUCCESS, PHASE 4B TRAINING-PROTOCOL LIMITATION",
}
json.dump(summary, open("results/phase_4/final_comparison/unified_comparison_summary.json", "w"), indent=2)

print(f"[OK] Unified comparison: {out_csv}")
print(f"     Phase 4A: POSITIVE (Phase4A.1 win rate {summary['phase_4a']['win_rate']*100:.1f}%)")
print(f"     Phase 4B: NEGATIVE (all variants physics-diverged)")
