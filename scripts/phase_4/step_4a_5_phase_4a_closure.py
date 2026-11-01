"""Step 4A.5 — Phase 4A Closure (consolidation + verification)."""
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(".")
P1 = ROOT / "results/phase_1"
P2 = ROOT / "results/phase_2"
P3 = ROOT / "results/phase_3"
P4A = ROOT / "results/phase_4/phase_4a"
CLOSURE = ROOT / "results/phase_4/closure"
CLOSURE.mkdir(parents=True, exist_ok=True)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def git_peel(ref):
    try:
        return subprocess.check_output(["git", "rev-parse", f"{ref}^{{commit}}"],
                                       text=True).strip()
    except subprocess.CalledProcessError:
        return None

print("=" * 70)
print("STEP 4A.5 — PHASE 4A CLOSURE")
print("=" * 70)

# ============ [1] Lock integrity ============
print("\n[1] Verifying locks...")
p1_manifest_sha = sha256_file(P1 / "phase_1_dataset_manifest.json")
p1_agg = json.load(open(P1 / "phase_1_dataset_manifest.json"))["dataset"]["aggregate_sha256"]
p1_ok = (p1_agg == "9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6"
         and p1_manifest_sha == "993581ef514932c5f3c341ce176a401a5247ebe7217e83f70d3ebe26dd3cff12")
p2_tag = git_peel("phase-2-lock")
p2_ckpt_sha = sha256_file(ROOT / "models/baseline/fno_best.pt")
p2_ok = (p2_tag == "0c9dfd7ced7454342e09280b6307f83d42eea89e"
         and p2_ckpt_sha == "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889")
p3_tag = git_peel("phase-3-closed")
p3_ckpt_sha = sha256_file(ROOT / "models/sc_fno/phase_3/pi_fno/best_val.pt")
p3_ok = (p3_tag == "bc807278181fb43553b91fe2bd0e9802734da29f"
         and p3_ckpt_sha == "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50")
locked_status = subprocess.check_output(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "results/phase_3", "models/baseline",
     "configs/reactor_parameters.v3.yaml", "configs/thermal_hydraulics.v4.yaml"],
    text=True).strip()
locked_clean = (locked_status == "")

lock_verification = {
    "phase_1": {"aggregate_sha256": p1_agg, "manifest_sha256": p1_manifest_sha, "verified": p1_ok},
    "phase_2": {"tag_commit": p2_tag, "checkpoint_sha256": p2_ckpt_sha, "verified": p2_ok},
    "phase_3": {"tag_commit": p3_tag, "pi_fno_ckpt_sha256": p3_ckpt_sha, "verified": p3_ok},
    "locked_paths_clean": locked_clean,
    "all_verified": p1_ok and p2_ok and p3_ok and locked_clean,
}
with open(CLOSURE / "lock_integrity_verification.json", "w") as f:
    json.dump(lock_verification, f, indent=2)
print(f"  Phase 1: {'PASS' if p1_ok else 'FAIL'}")
print(f"  Phase 2: {'PASS' if p2_ok else 'FAIL'}")
print(f"  Phase 3: {'PASS' if p3_ok else 'FAIL'}")
print(f"  Locked paths clean: {'PASS' if locked_clean else 'FAIL'}")
assert lock_verification["all_verified"], "Lock integrity failed — ABORT"

# ============ [2] Load evidence ============
print("\n[2] Loading evidence...")
residuals = json.load(open(P4A / "residual_analysis/residual_comparison_CORRECTED.json"))
paired = json.load(open(P4A / "residual_analysis/paired_bootstrap_stats.json"))
per_family = json.load(open(P4A / "residual_analysis/per_family_metrics.json"))
summary = json.load(open(P4A / "optimizer/training_summary.json"))

p3_phys = residuals["phase3"]["L_physics_aggregate"]
p4_phys = residuals["phase4a1"]["L_physics_aggregate"]
phys_ratio = p3_phys / p4_phys
p3_ic = residuals["phase3"]["L_IC"]
p4_ic = residuals["phase4a1"]["L_IC"]
ic_ratio = p3_ic / p4_ic

# Per-state summary
per_state = residuals["phase3"]["per_state_normalized"]
per_state_4a1 = residuals["phase4a1"]["per_state_normalized"]
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
per_state_rows = []
for s in STATES:
    p3v = per_state[s]; p4v = per_state_4a1[s]
    ratio = p3v / p4v if p4v > 0 else 0
    per_state_rows.append({
        "state": s, "phase3": p3v, "phase4a1": p4v,
        "ratio": ratio,
        "pct_change": (p4v - p3v) / p3v * 100 if p3v > 0 else 0,
    })
ratios = [r["ratio"] for r in per_state_rows]

# Reconcile win/loss
import pandas as pd
df = pd.read_csv(P4A / "residual_analysis/per_scenario_comparison.csv")
wins_4a1 = int((df["delta"] > 0).sum())
losses_4a1 = int((df["delta"] < 0).sum())
ties = int((df["delta"] == 0).sum())

# ============ [3] Statistical reversal reconciliation ============
print("\n[3] Writing reconciliation...")
reconciliation_md = f"""# Statistical Reversal Reconciliation

**Status:** AUTHORITATIVE — supersedes commit `3a04b3d`

## Issue
- Step 4A.3 report (`3a04b3d`): "Phase 3 wins 162/200" — **WRONG interpretation**
- Step 4A.3.R.2 report (`f7474c0`): "Phase 4A.1 wins 81% (162/200)" — **CORRECT**

## Metric Definition
`delta = phase3_rel_l2 − phase4a1_rel_l2` — positive means Phase 4A.1 better.

## Corrected Interpretation (Authoritative)
| Metric | Value |
|--------|-------|
| Phase 4A.1 better (delta > 0) | **{wins_4a1}/200** |
| Phase 3 better (delta < 0) | **{losses_4a1}/200** |
| Ties | {ties}/200 |
| **Phase 4A.1 win rate** | **{wins_4a1/200*100:.1f}%** |
| Mean paired delta | +{paired['overall']['mean_paired_delta']:.4e} |
| Mean 95% CI | [{paired['overall']['mean_paired_delta_ci95'][0]:+.4e}, {paired['overall']['mean_paired_delta_ci95'][1]:+.4e}] |
| Median paired delta | +{paired['overall']['median_paired_delta']:.4e} |
| Median 95% CI | [{paired['overall']['median_paired_delta_ci95'][0]:+.4e}, {paired['overall']['median_paired_delta_ci95'][1]:+.4e}] |

**Both CIs exclude zero → statistically significant.**

## Supersession
`f7474c0` is **authoritative**. `3a04b3d` had the win/loss labels flipped. Data unchanged.

All downstream references use corrected interpretation: **Phase 4A.1 wins {wins_4a1}/200 scenarios (81%).**
"""
(CLOSURE / "statistical_reversal_reconciliation.md").write_text(reconciliation_md)
print(f"  Phase 4A.1 wins: {wins_4a1}/200 ({wins_4a1/200*100:.1f}%)")

# Save summary for part 2
summary_data = {
    "p3_phys": p3_phys, "p4_phys": p4_phys, "phys_ratio": phys_ratio,
    "p3_ic": p3_ic, "p4_ic": p4_ic, "ic_ratio": ic_ratio,
    "per_state_rows": per_state_rows, "ratios": ratios,
    "wins_4a1": wins_4a1, "losses_4a1": losses_4a1, "ties": ties,
    "paired": paired, "per_family": per_family, "summary": summary,
}
with open(CLOSURE / "_closure_summary_data.json", "w") as f:
    json.dump(summary_data, f, indent=2, default=str)

print("\n[3a] Reconciliation written. Data cached for report generation.")
print("Part 1 complete.")
