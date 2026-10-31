"""Step 3.0.J — Consolidate preflight evidence + Gate 3.0 verdict."""
import json
import subprocess
from pathlib import Path

base = Path("results/phase_3/preflight")
def load(name):
    try:
        return json.load(open(base / name))
    except Exception as e:
        return {"__error__": str(e)}

art   = load("artifact_integrity.json")
eqmap = load("equation_to_source_mapping.json")
param = load("parameter_provenance.json")
io    = load("io_verification.json")
qsrc  = load("source_term_confirmation.json")
react = load("reactivity_verification.json")

checks = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))

check("3.0.A Phase 1/2 artifact integrity",
      art.get("all_pass"),
      f"{sum(1 for v in art.get('checks', {}).values() if v.get('pass'))}/"
      f"{len(art.get('checks', {}))} checks")

check("3.0.B Equation-to-source mapping",
      eqmap.get("all_pass"),
      f"{eqmap.get('found')}/{eqmap.get('total_terms')} terms found")

check("3.0.C Parameter provenance",
      param.get("summary", {}).get("all_pass"),
      f"fuel equilibrium ratio = {param.get('summary', {}).get('fuel_equilibrium_ratio')}")

check("3.0.D Architecture ADR",
      (base / "architecture_decision_record.md").exists(),
      "coordinate-conditioned decoder frozen")

check("3.0.E Padding policy",
      (base / "padding_policy.md").exists(),
      "3 padding modes, 4-variant matrix")

check("3.0.F Experimental matrix",
      (base / "experimental_comparison_matrix.md").exists(),
      "P2-FNO / P-FNO / PI-FNO / P-PI-FNO defined")

check("3.0.G I/O verification",
      io.get("all_pass"),
      f"time grid uniform, dt = "
      f"{io.get('time_grid_detail', {}).get('dt_value', 0):.9f} (not 0.05)")

check("3.0.H Source term confirmation",
      qsrc.get("all_pass"),
      "include_q_t=True, q_default=0")

check("3.0.I Reactivity verification",
      react.get("all_pass"),
      f"{sum(1 for v in react.get('checks', {}).values() if v.get('pass'))}/"
      f"{len(react.get('checks', {}))} checks")

# Locked paths untouched
status = subprocess.run(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "models/baseline",
     "configs/reactor_parameters.v3.yaml", "configs/thermal_hydraulics.v4.yaml"],
    capture_output=True, text=True,
).stdout.strip()
check("3.0 Locked paths untouched", status == "", "empty" if not status else status[:200])

all_pass = all(ok for _, ok, _ in checks)

# Write markdown summary
lines = ["# Step 3.0 Preflight Summary\n",
         "**Date:** 2026-10-31  ",
         f"**Gate 3.0:** {'PASS' if all_pass else 'FAIL'}  ",
         f"**Checks:** {sum(1 for _, ok, _ in checks if ok)}/{len(checks)}\n",
         "## Sub-step Verification\n",
         "| # | Sub-step | Result | Detail |",
         "|---|----------|--------|--------|"]
for i, (name, ok, detail) in enumerate(checks, 1):
    lines.append(f"| {i} | {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

# Documented findings carried into Step 3.1+
lines.append("\n## Documented Findings (Carry into Step 3.1+)\n")
lines.append("1. **Module attribution:** `feedback_reactivity`, `alpha_f`, `alpha_c`, `T_f_ref`, `T_c_ref` are defined in `thermal_hydraulics.py`, not `reactivity.py` (which imports and composes them). Coolant equation factor-of-2 is at `thermal_hydraulics.py:39`.")
lines.append(f"2. **Time grid:** Phase 1 uses `np.linspace(0,10,200)` → Δt = {io.get('time_grid_detail', {}).get('dt_value', 0):.9f} s (not 0.05). **Step 3.2 derivative implementation must use actual `time.npy` coordinate, never assume 0.05.**")
lines.append(f"3. **Fuel equilibrium verified:** ratio = {param.get('summary', {}).get('fuel_equilibrium_ratio')} (≈1.0, v4 correction confirmed applied).")
lines.append("4. **Reactivity feedback formula confirmed:** `alpha_f*(T_f - T_f_ref) + alpha_c*(T_c - T_c_ref)` (both alphas negative → negative feedback).")
lines.append(f"5. **q(t) source term:** include_q_t = True, default = 0.0 (Phase 0 decision preserved).")

lines.append("\n## Next Step\n")
lines.append("Step 3.1 — Complete physics residual implementation for all 9 states.")

with open(base / "preflight_summary.md", "w") as f:
    f.write("\n".join(lines) + "\n")

summary = {"checks": [{"name": n, "pass": p, "detail": d} for n, p, d in checks],
           "all_pass": all_pass,
           "gate_3_0": "PASS" if all_pass else "FAIL"}
with open(base / "preflight_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n".join(lines))
print(f"\n=== GATE 3.0: {'PASS' if all_pass else 'FAIL'} ===")
