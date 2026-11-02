"""Step 4B.5 — Generate 10 figures + 5 tables from all_models_metrics.json."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results/phase_4/phase_4b/evaluation")
FIG = Path("reports/figures/phase_4b"); FIG.mkdir(parents=True, exist_ok=True)
TAB = Path("reports/tables/phase_4b");  TAB.mkdir(parents=True, exist_ok=True)

metrics = json.load(open(OUT / "all_models_metrics.json"))
models = ["P2-FNO", "P3-PI-FNO", "Phase4A.1", "P4-A", "P4-B", "P4-C", "P4-D"]
STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
families = ["step", "ramp", "sinusoidal", "grf"]

COLORS = {
    "P2-FNO": "#2E86AB", "P3-PI-FNO": "#A23B72", "Phase4A.1": "#18A558",
    "P4-A": "#E67E22", "P4-B": "#C0392B", "P4-C": "#8E44AD", "P4-D": "#D35400",
}

# F1 — Overall rel L2 bar
fig, ax = plt.subplots(figsize=(11, 6))
vals = [metrics[m]["rel_l2_overall_median"] for m in models]
bars = ax.bar(models, vals, color=[COLORS[m] for m in models])
ax.set_ylabel("Rel L2 median (physical units)")
ax.set_title("F1 — Test-set overall rel L2 median (7 models, 200 scenarios)")
ax.tick_params(axis='x', rotation=45)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v * 1.02, f"{v:.4e}", ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "F1_overall_rel_l2.png", dpi=120); plt.close()

# F2 — Per-state heatmap
mat = np.zeros((7, 9))
for i, m in enumerate(models):
    for j, k in enumerate(STATES):
        mat[i, j] = metrics[m]["rel_l2_per_state_median"][k]
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(mat, aspect="auto", cmap="viridis")
ax.set_yticks(range(7)); ax.set_yticklabels(models)
ax.set_xticks(range(9)); ax.set_xticklabels(STATES)
ax.set_title("F2 — Per-state rel L2 median heatmap (7 models × 9 states)")
plt.colorbar(im, ax=ax, label="Rel L2 median")
for i in range(7):
    for j in range(9):
        ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                color="white" if mat[i,j] < mat.max()*0.6 else "black", fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "F2_per_state_heatmap.png", dpi=120); plt.close()

# F3 — Per-family grouped bars
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(families)); width = 0.11
for i, m in enumerate(models):
    vals = [metrics[m]["per_family"][f]["rel_l2_median"] for f in families]
    ax.bar(x + i*width, vals, width, label=m, color=COLORS[m])
ax.set_xticks(x + 3*width); ax.set_xticklabels(families)
ax.set_ylabel("Rel L2 median"); ax.set_yscale("log")
ax.set_title("F3 — Per-family rel L2 median (log scale)")
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "F3_per_family.png", dpi=120); plt.close()

# F4 — Physics loss comparison
fig, ax = plt.subplots(figsize=(11, 6))
vals = [metrics[m]["physics_loss"] if metrics[m]["physics_loss"] is not None else np.nan for m in models]
bars = ax.bar(models, vals, color=[COLORS[m] for m in models])
ax.axhline(1e-5, color="red", linestyle="--", linewidth=2, label="10^-5 target")
ax.axhline(1.68e-3, color="orange", linestyle=":", linewidth=1.5, label="P3 baseline (1.68e-3)")
ax.set_yscale("log")
ax.set_ylabel("Physics loss (9-equation, log scale)")
ax.set_title("F4 — Physics loss comparison vs 10^-5 target")
ax.tick_params(axis='x', rotation=45)
for b, v in zip(bars, vals):
    if not np.isnan(v):
        ax.text(b.get_x() + b.get_width()/2, v * 1.5, f"{v:.2e}", ha="center", fontsize=8)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "F4_physics_loss.png", dpi=120); plt.close()

# F5 — Boundary ratio n + Tc
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, state in zip(axes, ["n", "Tc"]):
    ratios = []
    for m in models:
        r = metrics[m]["boundary_metrics"].get(state, {}).get("start_interior_ratio")
        ratios.append(r if r is not None else np.nan)
    bars = ax.bar(models, ratios, color=[COLORS[m] for m in models])
    ax.axhline(4.0, color="red", linestyle="--", label="Target < 4.0")
    ax.axhline(8.0, color="gray", linestyle=":", label="Phase 2 audit approx 8")
    ax.set_title(f"F5 — {state} start/interior boundary ratio")
    ax.set_ylabel("Ratio")
    ax.tick_params(axis='x', rotation=45)
    for b, v in zip(bars, ratios):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, v * 1.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "F5_boundary_ratio.png", dpi=120); plt.close()

# F6 — Endpoint errors t=0, t=10
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, ep in zip(axes, ["mae_t0", "mae_t10"]):
    vals_n = [metrics[m]["endpoint_metrics"].get("n", {}).get(ep, np.nan) for m in models]
    vals_tf = [metrics[m]["endpoint_metrics"].get("Tf", {}).get(ep, np.nan) for m in models]
    x = np.arange(len(models))
    ax.bar(x - 0.2, vals_n, 0.4, label="n", color="steelblue")
    ax.bar(x + 0.2, vals_tf, 0.4, label="Tf (K)", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=45)
    ax.set_title(f"F6 — {ep} endpoint error")
    ax.set_yscale("log"); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "F6_endpoint_errors.png", dpi=120); plt.close()

# F7 — Paired bootstrap CI forest plot
bs = json.load(open(OUT / "paired_bootstrap_vs_p2.json"))["vs_P2-FNO"]
fig, ax = plt.subplots(figsize=(10, 5))
names = list(bs.keys())
medians = [bs[n]["median_delta"] for n in names]
los = [bs[n]["ci95"][0] for n in names]
his = [bs[n]["ci95"][1] for n in names]
y_pos = np.arange(len(names))
ax.errorbar(medians, y_pos, xerr=[np.array(medians)-np.array(los), np.array(his)-np.array(medians)],
            fmt="o", capsize=5, color="black", ecolor="steelblue")
ax.axvline(0, color="red", linestyle="--", label="No difference")
ax.set_yticks(y_pos); ax.set_yticklabels(names)
ax.set_xlabel("Median paired delta (P2 - variant). Positive = variant better.")
ax.set_title("F7 — Paired bootstrap 95% CI vs P2-FNO")
ax.legend(); ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "F7_paired_bootstrap.png", dpi=120); plt.close()

# F8 — Scenario-level paired delta histogram
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, m in zip(axes.flat, ["P3-PI-FNO", "Phase4A.1", "P4-A", "P4-C"]):
    p2 = np.array(metrics["P2-FNO"]["per_scenario_rel_l2"])
    mn = np.array(metrics[m]["per_scenario_rel_l2"])
    delta = p2 - mn
    ax.hist(delta, bins=40, color=COLORS[m], alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.axvline(np.median(delta), color="black", linestyle="-",
               label=f"median = {np.median(delta):.2e}")
    wins = (delta > 0).sum()
    ax.set_title(f"F8 — {m} vs P2-FNO: delta distribution (wins {wins}/200)")
    ax.set_xlabel("Delta = P2 rel L2 - variant rel L2")
    ax.set_ylabel("Count"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "F8_delta_histogram.png", dpi=120); plt.close()

# F9 — Worst-case scenario per model
worst_scn = {}
for m in models:
    per_scn = np.array(metrics[m]["per_scenario_rel_l2"])
    worst_idx = int(np.argmax(per_scn))
    worst_scn[m] = {"idx": worst_idx, "rel_l2": float(per_scn[worst_idx]),
                    "per_state": metrics[m]["per_scenario_per_state_rel_l2"][worst_idx]}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(9); width = 0.11
for i, m in enumerate(models):
    vals = worst_scn[m]["per_state"]
    ax.bar(x + i*width, vals, width, label=f"{m} (worst={worst_scn[m]['rel_l2']:.2e})", color=COLORS[m])
ax.set_xticks(x + 3*width); ax.set_xticklabels(STATES)
ax.set_ylabel("Per-state rel L2")
ax.set_title("F9 — Per-state rel L2 at each model's worst scenario")
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "F9_worst_case.png", dpi=120); plt.close()

# F10 — Intervention isolation (P4-A/B/C/D vs P3 baseline)
fig, ax = plt.subplots(figsize=(11, 6))
baseline_p3 = metrics["P3-PI-FNO"]["rel_l2_overall_median"]
variants_only = ["P4-A", "P4-B", "P4-C", "P4-D"]
deltas = [metrics[v]["rel_l2_overall_median"] - baseline_p3 for v in variants_only]
colors_v = [COLORS[v] for v in variants_only]
bars = ax.bar(variants_only, deltas, color=colors_v)
ax.axhline(0, color="black", linestyle="-", label="P3-PI-FNO baseline")
ax.set_ylabel("Delta rel L2 median vs P3 (positive = worse)")
ax.set_title("F10 — Intervention isolation: P4-A/B/C/D vs P3-PI-FNO")
for b, v in zip(bars, deltas):
    sign = "+" if v > 0 else ""
    ax.text(b.get_x() + b.get_width()/2, v, f"{sign}{v:.4e}", ha="center", fontsize=9)
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(FIG / "F10_intervention_isolation.png", dpi=120); plt.close()

print(f"[OK] 10 figures written to {FIG}/")

# T1 — Overall metrics
rows = []
for m in models:
    r = metrics[m]
    rows.append({"model": m,
                 "rel_l2_median": r["rel_l2_overall_median"],
                 "rel_l2_mean": r["rel_l2_overall_mean"],
                 "physics_loss": r["physics_loss"],
                 "ic_loss": r["ic_loss"],
                 "eval_time_s": r.get("eval_time_s")})
pd.DataFrame(rows).to_csv(TAB / "T1_overall_metrics.csv", index=False)
print(f"[OK] T1 written")

# T2 — Per-state
rows = []
for m in models:
    for k in STATES:
        rows.append({"model": m, "state": k,
                     "rel_l2_median": metrics[m]["rel_l2_per_state_median"][k],
                     "rel_l2_mean": metrics[m]["rel_l2_per_state_mean"][k]})
pd.DataFrame(rows).to_csv(TAB / "T2_per_state.csv", index=False)
print(f"[OK] T2 written")

# T3 — Per-family
rows = []
for m in models:
    for fam in families:
        fm = metrics[m]["per_family"].get(fam, {})
        rows.append({"model": m, "family": fam,
                     "n_scenarios": fm.get("n_scenarios"),
                     "rel_l2_median": fm.get("rel_l2_median"),
                     "rel_l2_mean": fm.get("rel_l2_mean")})
pd.DataFrame(rows).to_csv(TAB / "T3_per_family.csv", index=False)
print(f"[OK] T3 written")

# T4 — Paired bootstrap vs P2
bs_p2 = json.load(open(OUT / "paired_bootstrap_vs_p2.json"))["vs_P2-FNO"]
rows = []
for m, v in bs_p2.items():
    rows.append({"model": m, "wins": v["wins"], "losses": v["losses"],
                 "win_rate": v["win_rate"], "median_delta": v["median_delta"],
                 "ci95_lo": v["ci95"][0], "ci95_hi": v["ci95"][1],
                 "significant_at_95": not (v["ci95"][0] <= 0 <= v["ci95"][1])})
pd.DataFrame(rows).to_csv(TAB / "T4_paired_bootstrap_vs_p2.csv", index=False)
print(f"[OK] T4 written")

# T5 — Endpoint
rows = []
for m in models:
    for k in STATES:
        ep = metrics[m]["endpoint_metrics"].get(k, {})
        rows.append({"model": m, "state": k,
                     "mae_t0": ep.get("mae_t0"), "mae_t10": ep.get("mae_t10"),
                     "mae_near_start_0_05": ep.get("mae_near_start"),
                     "mae_near_end_095_10": ep.get("mae_near_end")})
pd.DataFrame(rows).to_csv(TAB / "T5_endpoint.csv", index=False)
print(f"[OK] T5 written")

print(f"\n[OK] 5 tables written to {TAB}/")
