"""Step 4A.2.R2 — Finalize corrections: regenerate figures, write addendum,
update manifest."""
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path("results/phase_4/phase_4a/residual_analysis")
FIGURES = Path("reports/figures/phase_4/phase_4a")
FIGURES.mkdir(parents=True, exist_ok=True)
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

# ---- Load corrected data ----
corrected = json.load(open(ANALYSIS / "residual_comparison_CORRECTED.json"))
boundary = json.load(open(ANALYSIS / "boundary_comparison.json"))

p3 = corrected["phase3"]
p4 = corrected["phase4a1"]

# ---- Regenerate Figure 1: per-state residuals (CORRECTED) ----
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(STATES))
w = 0.35
p3_vals = [p3["per_state_normalized"][s] for s in STATES]
p4_vals = [p4["per_state_normalized"][s] for s in STATES]
ax.bar(x - w/2, p3_vals, w, label="Phase 3 PI-FNO", color="#d9534f")
ax.bar(x + w/2, p4_vals, w, label="Phase 4A.1 final", color="#5cb85c")
ax.set_yscale("log")
ax.set_ylabel("Normalized residual (R_j/s_j)²")
ax.set_title("Per-State Physics Residuals (CORRECTED): all 9 states improved 177×–5,672×")
ax.set_xticks(x); ax.set_xticklabels(STATES, rotation=0)
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGURES / "residual_comparison_per_state_CORRECTED.png", dpi=150)
plt.close()

# ---- Regenerate Figure 2: aggregate (unchanged number, clearer annotation) ----
fig, ax = plt.subplots(figsize=(6, 5))
vals = [p3["L_physics_aggregate"], p4["L_physics_aggregate"]]
bars = ax.bar(["Phase 3\nPI-FNO", "Phase 4A.1\nfinal"], vals, color=["#d9534f", "#5cb85c"])
ax.axhline(1e-5, color="black", linestyle="--", label="10⁻⁵ target")
ax.set_yscale("log")
ax.set_ylabel("Aggregate L_physics (9-equation mean)")
ax.set_title("Physics Loss: 828× improvement, 9-equation, 10⁻⁵ target achieved")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v*1.5, f"{v:.3e}", ha="center", fontsize=10)
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGURES / "physics_loss_comparison_CORRECTED.png", dpi=150)
plt.close()

# ---- Regenerate Figure 3: boundary (with delta annotations) ----
fig, ax = plt.subplots(figsize=(10, 6))
p3_ratios = [boundary["phase3"]["states"][s]["start_interior_ratio"] for s in STATES]
p4_ratios = [boundary["phase4a1"]["states"][s]["start_interior_ratio"] for s in STATES]
ax.bar(x - w/2, p3_ratios, w, label="Phase 3", color="#d9534f")
ax.bar(x + w/2, p4_ratios, w, label="Phase 4A.1", color="#5cb85c")
ax.axhline(2.0, color="black", linestyle="--", label="Phase 4A boundary target (< 2.0)")
# Annotate deltas
for i, s in enumerate(STATES):
    d = p4_ratios[i] - p3_ratios[i]
    ax.text(i, max(p3_ratios[i], p4_ratios[i]) + 0.15,
            f"{d:+.2f}", ha="center", fontsize=9,
            color="red" if d > 0.1 else "black")
ax.set_ylabel("Start/interior MAE ratio")
ax.set_title("Boundary Artifact (CORRECTED): persists; 5 states slightly worsened, 4 unchanged")
ax.set_xticks(x); ax.set_xticklabels(STATES)
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGURES / "boundary_error_comparison_CORRECTED.png", dpi=150)
plt.close()

print("✓ Generated 3 corrected figures")

# ---- Write corrected residual_comparison.md ----
md = []
md.append("# Step 4A.2 — Residual Convergence Assessment (CORRECTED)")
md.append("")
md.append("**Phase 4A.1 v2 vs Phase 3 PI-FNO — 200 validation scenarios**")
md.append("")
md.append(f"- Phase 3 PI-FNO SHA: `{p3.get('ckpt_path', 'a0522e57...')}`")
md.append(f"- Phase 4A.1 final SHA: `{p4.get('ckpt_path', '681fe67...')}`")
md.append("")
md.append("## Correction Note")
md.append("")
md.append("The original Phase 4A.2 per-state diagnostic (`bf1e6d5`) contained a bug: it")
md.append("manually denormalized `du_hat_dt` by `norm_std`, while `compute_physics_loss`")
md.append("passes `du_hat_dt` directly (already in physical units from `compute_du_dt`).")
md.append("This produced an anomalous R_n = 1897 in the original CSV. **This correction")
md.append("recomputes per-state residuals using the exact same path as `compute_physics_loss`**, ")
md.append("verifying aggregate = mean(9 per-state contributions) to machine precision.")
md.append("")
md.append("## Aggregate Metrics")
md.append("")
md.append("| Metric | Phase 3 | Phase 4A.1 | Δ | Ratio |")
md.append("|--------|---------|------------|---|-------|")
md.append(f"| L_physics (9-eq) | {p3['L_physics_aggregate']:.4e} | {p4['L_physics_aggregate']:.4e} | −99.88% | **{corrected['aggregate']['ratio']:.1f}×** |")
md.append(f"| L_data (final ckpt) | 0.066331 | 0.066678 | +0.52% | 1.005× |")
md.append("")
md.append("## Per-State Residuals (CORRECTED)")
md.append("")
md.append("| State | Phase 3 | Phase 4A.1 | Ratio | Δ% |")
md.append("|-------|---------|------------|-------|-----|")
for s in STATES:
    a = p3["per_state_normalized"][s]
    b = p4["per_state_normalized"][s]
    r = a / b if b > 0 else 0
    pct = (b - a) / a * 100
    md.append(f"| {s} | {a:.4e} | {b:.4e} | {r:.1f}× | {pct:+.2f}% |")
md.append("")
md.append(f"**All 9 per-state residuals improved by 177×–5,672×.** Aggregate: {corrected['aggregate']['ratio']:.1f}×.")
md.append("")
md.append("## Physics Target (10⁻⁵)")
md.append("")
md.append(f"- Phase 3: **{'ACHIEVED' if corrected['target_1e-5']['phase3_achieved'] else 'NOT ACHIEVED'}** "
          f"({corrected['target_1e-5']['phase3_ratio']:.2f}× target)")
md.append(f"- Phase 4A.1: **{'ACHIEVED' if corrected['target_1e-5']['phase4a1_achieved'] else 'NOT ACHIEVED'}** "
          f"({1/corrected['target_1e-5']['phase4a1_ratio']:.1f}× margin)")
md.append("")
md.append("## Data Loss Reconciliation")
md.append("")
md.append("- Phase 4A.1 `training_summary.json` recorded: 0.066702 (at plateau epoch 54)")
md.append("- Phase 4A.2 fresh evaluation of `phase_4a_final.pt`: 0.066678 (post-L-BFGS)")
md.append("- Difference: 3.6e-5 — L-BFGS slightly improved val_data_loss")
md.append("- Both vs Phase 3 baseline (0.066331): +0.56% (plateau) / +0.52% (final)")
md.append("- **Comparison uses final checkpoint: +0.52%**")
md.append("")
md.append("## Boundary Behavior (REVISED)")
md.append("")
md.append("| State | Phase 3 | Phase 4A.1 | Δ | Verdict |")
md.append("|-------|---------|------------|---|---------|")
for s in STATES:
    a = boundary["phase3"]["states"][s]["start_interior_ratio"]
    b = boundary["phase4a1"]["states"][s]["start_interior_ratio"]
    d = b - a
    verdict = "Worsened" if d > 0.1 else ("Slight" if d > 0.02 else "Unchanged")
    md.append(f"| {s} | {a:.3f} | {b:.3f} | {d:+.3f} | {verdict} |")
md.append("")
md.append("**Boundary artifact persists. 5 of 9 states slightly worsened** (T_c, C_5, T_f, C_6, n).")
md.append("Adaptive fine-tuning did not address the FFT wraparound issue.")
md.append("")
md.append("## Scientific Summary")
md.append("")
md.append("**Phase 4A.1 achieves a genuine 828× reduction in 9-equation physics loss**")
md.append("(1.68e-3 → 2.03e-6), satisfying the 10⁻⁵ target with 4.9× margin, at a data")
md.append("accuracy cost of 0.52%. **Every one of the 9 PKE+TH residuals improved by**")
md.append("**177×–5,672×.** The R_n bottleneck claimed in the original report was a")
md.append("diagnostic bug, not a physical or architectural limitation.")
md.append("")
Path(ANALYSIS / "residual_comparison_CORRECTED.md").write_text("\n".join(md))
print("✓ Wrote residual_comparison_CORRECTED.md")

# ---- Update manifest ----
artifacts = {}
for p in sorted(ANALYSIS.glob("*")):
    if p.is_file() and p.name != "step_4a_2_manifest.json":
        artifacts[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted(FIGURES.glob("*CORRECTED*.png")):
    artifacts[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

manifest = {
    "phase": "4A.2.R",
    "gate": "PASS (corrected)",
    "correction_reason": "Original per-state diagnostic double-denormalized du_hat_dt; corrected path matches compute_physics_loss exactly",
    "phase3_ckpt_sha": "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
    "phase4a1_ckpt_sha": "681fe6789dd2911a43991db0242a20f8a08d5ba75f82b28f9dbbeaba88ad744f",
    "aggregate_ratio": corrected["aggregate"]["ratio"],
    "range_x": [min(p3["per_state_normalized"][s] / p4["per_state_normalized"][s] for s in STATES),
                max(p3["per_state_normalized"][s] / p4["per_state_normalized"][s] for s in STATES)],
    "physics_target_achieved": corrected["target_1e-5"]["phase4a1_achieved"],
    "reconciliation_verified": True,
    "artifacts": artifacts,
}
with open(ANALYSIS / "step_4a_2_r_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"✓ Manifest: {len(artifacts)} artifacts")
print("\n=== Phase 4A.2.R finalize: COMPLETE ===")
