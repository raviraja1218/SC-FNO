"""Step 3.5 reporting corrections — statistical rigor + evidence fills."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
from step3_5_eval_utils import STATES, get_family, rel_l2, load_test_data, load_norm, load_test_ids

base = Path("results/phase_3/evaluation")

# Load scenario metrics for all 4
dfs = {}
for v in ["p2", "p_fno", "pi_fno", "p_pi_fno"]:
    dfs[v] = pd.read_csv(base / f"{v}_scenario_metrics.csv")

# ============================================================
# CORRECTION 1: Paired bootstrap statistical test (PI-FNO vs P2-FNO)
# ============================================================
print("=" * 70)
print("CORRECTION 1: Paired bootstrap (PI-FNO vs P2-FNO)")
print("=" * 70)

p2_df = dfs["p2"].set_index("scenario_id")
pi_df = dfs["pi_fno"].set_index("scenario_id")
p_pi_df = dfs["p_pi_fno"].set_index("scenario_id")
pfno_df = dfs["p_fno"].set_index("scenario_id")

# Align on scenario IDs
common = sorted(set(p2_df.index) & set(pi_df.index))
p2_agg = p2_df.loc[common, "aggregate_rel_l2"].values
pi_agg = pi_df.loc[common, "aggregate_rel_l2"].values
delta = p2_agg - pi_agg  # positive = PI wins

# Bootstrap
rng = np.random.default_rng(42)
N = len(delta)
n_boot = 10000
boot_medians = np.zeros(n_boot)
for i in range(n_boot):
    idx = rng.integers(0, N, N)
    boot_medians[i] = np.median(delta[idx])

ci_lo, ci_hi = np.percentile(boot_medians, [2.5, 97.5])

# Win rates
pi_wins = int(np.sum(pi_agg < p2_agg))
p2_wins = int(np.sum(p2_agg < pi_agg))
ties = int(np.sum(p2_agg == pi_agg))

pct = lambda arr, p: float(np.percentile(arr, p))

stat = {
    "comparison": "PI-FNO vs P2-FNO (aggregate rel L2 per scenario)",
    "n_scenarios": N,
    "pi_fno_wins": pi_wins,
    "p2_fno_wins": p2_wins,
    "ties": ties,
    "win_rate_pi_fno": pi_wins / N,
    "median_paired_delta": float(np.median(delta)),
    "mean_paired_delta": float(np.mean(delta)),
    "median_paired_delta_95ci": [float(ci_lo), float(ci_hi)],
    "delta_percentiles": {
        "p25": pct(delta, 25), "p50": pct(delta, 50),
        "p75": pct(delta, 75), "p90": pct(delta, 90), "p95": pct(delta, 95),
        "p05": pct(delta, 5), "p10": pct(delta, 10),
    },
    "statistically_significant": bool(ci_lo > 0 or ci_hi < 0),
    "interpretation": (
        "Statistically significant improvement (95% CI excludes 0)"
        if (ci_lo > 0 or ci_hi < 0) else
        "Observed improvement, not statistically established at 95% CI"
    ),
}

# Same for P-PI vs P2, P-FNO vs P2
for other, other_name in [("p_pi_fno", "P-PI-FNO"), ("p_fno", "P-FNO")]:
    o_df = dfs[other].set_index("scenario_id")
    o_agg = o_df.loc[common, "aggregate_rel_l2"].values
    d = p2_agg - o_agg
    bm = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, N)
        bm[i] = np.median(d[idx])
    lo, hi = np.percentile(bm, [2.5, 97.5])
    stat[f"{other}_vs_p2"] = {
        "wins": int(np.sum(o_agg < p2_agg)),
        "win_rate": float(np.mean(o_agg < p2_agg)),
        "median_delta": float(np.median(d)),
        "median_delta_95ci": [float(lo), float(hi)],
        "statistically_significant": bool(lo > 0 or hi < 0),
    }

print(json.dumps(stat, indent=2))
(base / "statistical_significance.json").write_text(json.dumps(stat, indent=2))
print("Saved statistical_significance.json\n")

# ============================================================
# CORRECTION 2: P2-FNO per-family table
# ============================================================
print("=" * 70)
print("CORRECTION 2: Full per-family table (all 4 models)")
print("=" * 70)

per_family_full = {}
for fam in ["step", "ramp", "sinusoidal", "grf"]:
    per_family_full[fam] = {}
    for v, name in [("p2","P2-FNO"), ("p_fno","P-FNO"), ("pi_fno","PI-FNO"), ("p_pi_fno","P-PI-FNO")]:
        sub = dfs[v][dfs[v]["family"] == fam]
        if len(sub) > 0:
            per_family_full[fam][name] = float(np.median(sub["aggregate_rel_l2"]))

print(json.dumps(per_family_full, indent=2))

# Four comparisons
comparisons = {}
for (a, b, label) in [
    ("P2-FNO", "PI-FNO", "physics_effect_no_padding"),
    ("P-FNO", "P-PI-FNO", "physics_effect_with_padding"),
    ("P2-FNO", "P-FNO", "padding_effect_no_physics"),
    ("PI-FNO", "P-PI-FNO", "padding_effect_with_physics"),
]:
    comparisons[label] = {}
    for fam in per_family_full:
        if a in per_family_full[fam] and b in per_family_full[fam]:
            va = per_family_full[fam][a]
            vb = per_family_full[fam][b]
            pct = (va - vb) / va * 100
            comparisons[label][fam] = {"a": va, "b": vb, "pct_change_b_vs_a": pct}

print(json.dumps(comparisons, indent=2))

corr2 = {
    "per_family_full": per_family_full,
    "four_comparisons": comparisons,
    "hypothesis_note": (
        "Family-level differences are observations, not proven mechanisms. "
        "The delayed-neutron-response hypothesis for sinusoidal improvement "
        "is plausible but not established by family-level measurements alone."
    ),
}
(base / "per_family_full.json").write_text(json.dumps(corr2, indent=2))
print("Saved per_family_full.json\n")

# ============================================================
# CORRECTION 3: P2-FNO scenario_0458 (same pipeline)
# ============================================================
print("=" * 70)
print("CORRECTION 3: P2-FNO scenario_0458 recomputed")
print("=" * 70)

# P2 predictions already exist in p2_predictions.json
p2_preds = json.load(open(base / "p2_predictions.json"))
test_data = load_test_data()

sid = "scenario_0458"
p2_pred_458 = np.array(p2_preds["predictions"][sid])  # [9, 200]
gt_458 = test_data[sid]["trajectory"].T  # [9, 200]
p2_rel_l2_458 = float(rel_l2(p2_pred_458.flatten(), gt_458.flatten()))

# Also compute SC-FNO variants for direct comparison
sc_results = {}
for v in ["p_fno", "pi_fno", "p_pi_fno"]:
    v_preds = json.load(open(base / f"{v}_predictions.json"))
    v_pred = np.array(v_preds["predictions"][sid])
    sc_results[v] = float(rel_l2(v_pred.flatten(), gt_458.flatten()))

corr3 = {
    "scenario_id": sid,
    "P2_FNO_recomputed_rel_l2": p2_rel_l2_458,
    "P_FNO_rel_l2": sc_results["p_fno"],
    "PI_FNO_rel_l2": sc_results["pi_fno"],
    "P_PI_FNO_rel_l2": sc_results["p_pi_fno"],
    "note": (
        "P2-FNO recomputed under identical pipeline (aggregate rel L2 over "
        "[9, 200]). The prior 1.85e-2 value from Phase 2 aggregate metrics "
        "was computed under a different convention and is NOT directly comparable."
    ),
}
print(json.dumps(corr3, indent=2))
(base / "scenario_0458_recomputed.json").write_text(json.dumps(corr3, indent=2))
print("Saved scenario_0458_recomputed.json\n")

# ============================================================
# CORRECTION 6: Gradient cosine evolution (from training logs)
# ============================================================
print("=" * 70)
print("CORRECTION 6: Gradient cos evolution")
print("=" * 70)

grad_ev = {}
for v in ["pi_fno", "p_pi_fno"]:
    log = pd.read_csv(f"results/phase_3/training/{v}/training_log.csv")
    cos_vals = log["cos_data_phys"].tolist()
    # Skip epoch 0 (warmup)
    valid = [c for c in cos_vals[1:] if not np.isnan(c)]
    if valid:
        grad_ev[v] = {
            "n_epochs_with_cos": len(valid),
            "initial_cos_after_warmup": float(valid[0]),
            "min_cos": float(min(valid)),
            "max_cos": float(max(valid)),
            "final_cos": float(valid[-1]),
            "epochs_below_-0.15": int(sum(1 for c in valid if c < -0.15)),
            "epochs_below_-0.25": int(sum(1 for c in valid if c < -0.25)),
            "note": "Reference cos at Step 3.3 init was −0.239. Do not claim 'resolved' — state the trajectory."
        }
print(json.dumps(grad_ev, indent=2))
(base / "gradient_conflict_evidence.json").write_text(json.dumps(grad_ev, indent=2))
print("Saved gradient_conflict_evidence.json\n")

print("=" * 70)
print("CORRECTIONS 1, 2, 3, 6 COMPLETE")
print("=" * 70)
print("\nNext: Correction 4 (throughput reconciliation) and 5 (resolution terminology)")
print("require documentation edits — delivered separately.")
