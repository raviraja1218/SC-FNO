"""Step 4A.0.E — Extract Phase 3 test metrics as frozen reference for 4A.5."""
import json
from pathlib import Path

EVAL = Path("results/phase_3/evaluation")
out = {"reference_files": {}, "extracted": {}}

def safe_load(name):
    p = EVAL / name
    if not p.exists():
        out["reference_files"][name] = "MISSING"
        return None
    try:
        d = json.load(open(p))
        out["reference_files"][name] = "loaded"
        return d
    except Exception as e:
        out["reference_files"][name] = f"ERROR: {e}"
        return None

fc = safe_load("final_comparison.json")
if fc: out["extracted"]["final_comparison"] = fc

ss = safe_load("statistical_significance.json")
if ss:
    out["extracted"]["paired_bootstrap"] = {
        "pi_fno_wins": ss.get("pi_fno_wins"),
        "p2_fno_wins": ss.get("p2_fno_wins"),
        "win_rate": ss.get("win_rate_pi_fno"),
        "median_paired_delta": ss.get("median_paired_delta"),
        "ci95": ss.get("median_paired_delta_95ci"),
    }

pf = safe_load("per_family_full.json")
if pf: out["extracted"]["per_family"] = pf

bc = safe_load("boundary_comparison.json")
if bc: out["extracted"]["boundary"] = bc

gc = safe_load("gate_checklist.json")
if gc: out["extracted"]["gate_checklist"] = gc

wc = safe_load("scenario_0458_recomputed.json")
if wc: out["extracted"]["worst_case_0458"] = wc

with open("results/phase_4/phase_4a/preflight/phase_3_test_reference.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out["reference_files"], indent=2))
print(f"\nExtracted keys: {list(out['extracted'].keys())}")
print("=== 4A.0.E Phase 3 test reference: recorded ===")
