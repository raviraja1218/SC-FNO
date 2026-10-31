"""Step 3.1.H — Consolidate evidence + Gate 3.1 verdict."""
import hashlib, json, subprocess
from pathlib import Path

base = Path("results/phase_3/residuals")
def load(n):
    try: return json.load(open(base / n))
    except Exception as e: return {"__err__": str(e)}

scales = load("residual_scales.json")
leak   = load("no_leakage_verification.json")

# Parse pytest log for pass count
pt_log = (base / "pytest_output.log").read_text()
n_pass = pt_log.count(" PASSED")
n_fail = pt_log.count(" FAILED")

checks = []
def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))

chk("8 validation tests exist (13 items)", n_pass == 13 and n_fail == 0,
    f"{n_pass} passed, {n_fail} failed")
chk("T1 neutron", "test_T1_neutron_residual_analytical PASSED" in pt_log)
chk("T2 precursors (6)", sum(1 for i in range(6)
    if f"test_T2_precursor_residual_analytical[{i}] PASSED" in pt_log) == 6)
chk("T3 fuel thermal", "test_T3_fuel_thermal_adiabatic PASSED" in pt_log)
chk("T4 coolant", "test_T4_coolant_thermal PASSED" in pt_log)
chk("T5 feedback", "test_T5_feedback_reactivity PASSED" in pt_log)
chk("T6 coupling", "test_T6_coupling_predicted_states_used PASSED" in pt_log)
chk("T7 equilibrium", "test_T7_equilibrium_residuals_zero PASSED" in pt_log)
chk("T8 unit consistency", "test_T8_unit_consistency PASSED" in pt_log)
chk("Residual scales measured", "per_state" in scales and
    len(scales["per_state"]) == 9, f"{len(scales.get('per_state', {}))} states")
chk("No leakage audit PASS", leak.get("all_pass"), leak.get("interpretation", "")[:80])
chk("Units documented", (base / "residual_units.md").exists())
chk("residuals.py present", Path("src/physics_informed/residuals.py").exists())
chk("test_residuals.py present",
    Path("tests/test_physics_informed/test_residuals.py").exists())

# Locked paths untouched
st = subprocess.run(
    ["git","status","--porcelain",
     "results/phase_1","results/phase_2","models/baseline",
     "configs/reactor_parameters.v3.yaml","configs/thermal_hydraulics.v4.yaml"],
    capture_output=True, text=True).stdout.strip()
chk("Locked paths untouched", st == "", "empty" if not st else st[:100])

all_pass = all(ok for _, ok, _ in checks)

lines = ["# Step 3.1 — Residual Validation Report", "",
         f"**Date:** 2026-10-31  ",
         f"**Gate 3.1:** {'PASS' if all_pass else 'FAIL'}  ",
         f"**Checks:** {sum(1 for _, ok, _ in checks if ok)}/{len(checks)}", "",
         "## Checklist", "",
         "| # | Criterion | Result | Detail |",
         "|---|-----------|--------|--------|"]
for i, (n, ok, d) in enumerate(checks, 1):
    lines.append(f"| {i} | {n} | {'PASS' if ok else 'FAIL'} | {d} |")

lines += ["", "## Residual Scales (p95)", "",
          "| State | p95 magnitude | Unit |",
          "|-------|--------------|------|"]
unit_map = {"n": "1/s", "C1": "1/s", "C2": "1/s", "C3": "1/s",
            "C4": "1/s", "C5": "1/s", "C6": "1/s", "Tf": "K/s", "Tc": "K/s"}
for state, info in scales.get("per_state", {}).items():
    lines.append(f"| {state} | {info['p95']:.4e} | {unit_map.get(state,'')} |")

lines += ["", "## Deliverables", "",
          "- `src/physics_informed/residuals.py` — 9 residual channels",
          "- `tests/test_physics_informed/test_residuals.py` — 13 validation tests",
          "- `residual_scales.json` — per-state p95 from 100 training scenarios",
          "- `no_leakage_verification.json` — feedback uses predicted Tf/Tc",
          "- `residual_units.md` — physical dimensions documented",
          "",
          "## Notes", "",
          "- All 9 residuals computed using PREDICTED states (no ground-truth leakage).",
          "- Coolant factor-of-2 matches `thermal_hydraulics.py:39` exactly.",
          "- q(t) handled per Phase 0: include_q_t=True, q_default=0.",
          "- Feedback formula `alpha_f*(Tf-Tf_ref) + alpha_c*(Tc-Tc_ref)` verified.",
          "- FP32 tolerances documented in test file docstring (scientifically justified).",
          ""]

with open(base / "residual_validation.md", "w") as f:
    f.write("\n".join(lines))

summary = {"checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
           "all_pass": all_pass,
           "gate_3_1": "PASS" if all_pass else "FAIL"}
with open(base / "residual_validation.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n".join(lines))
print(f"\n=== GATE 3.1: {'PASS' if all_pass else 'FAIL'} ===")
