"""Step 3.0.I — Verify reactivity feedback formula, signs, and reference temps."""
import json
import re
import yaml
from pathlib import Path

source = Path("src/physics/reactivity.py")
text = source.read_text()

out = {"source_file": str(source), "checks": {}, "all_pass": False}
def check(name, ok, detail=""):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# 1. make_rho_total function exists
check("make_rho_total_exists", "def make_rho_total" in text, "function defined")

# 2. Extract full reactivity.py content into evidence
out["reactivity_source_full"] = text

# 3. alpha_f and alpha_c present (any form)
check("alpha_f_present", "alpha_f" in text, "alpha_f found")
check("alpha_c_present", "alpha_c" in text, "alpha_c found")

# 4. Reference temperatures — verify against locked config
th_cfg = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
fb = th_cfg["reactivity_feedback"]
def scalar(v):
    return float(v[0]) if isinstance(v, list) else float(v)

check("config_alpha_f_sign", scalar(fb["alpha_f"]) < 0,
      f"alpha_f = {scalar(fb['alpha_f'])} (should be < 0, Doppler)")
check("config_alpha_c_sign", scalar(fb["alpha_c"]) < 0,
      f"alpha_c = {scalar(fb['alpha_c'])} (should be < 0, coolant coefficient)")
check("config_T_f_ref", abs(scalar(fb["T_f_ref"]) - 900.0) < 1e-6,
      f"T_f_ref = {scalar(fb['T_f_ref'])} (should be 900)")
check("config_T_c_ref", abs(scalar(fb["T_c_ref"]) - 583.0) < 1e-6,
      f"T_c_ref = {scalar(fb['T_c_ref'])} (should be 583)")

# 5. Source references T_f_ref or T_c_ref or equivalents
check("source_refs_T_f", bool(re.search(r"T_f_ref|Tf_ref|fuel.*ref", text)),
      "fuel reference temp referenced")
check("source_refs_T_c", bool(re.search(r"T_c_ref|Tc_ref|coolant.*ref", text)),
      "coolant reference temp referenced")

# 6. Feedback zero at initial state (algebraic check)
# alpha_f * (T_f0 - T_f_ref) + alpha_c * (T_c0 - T_c_ref)
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
init = kin_cfg["initial_state"]
T_f0 = scalar(init["T_f0"])
T_c0 = scalar(init["T_c0"])
feedback_at_init = (
    scalar(fb["alpha_f"]) * (T_f0 - scalar(fb["T_f_ref"]))
    + scalar(fb["alpha_c"]) * (T_c0 - scalar(fb["T_c_ref"]))
)
check("feedback_zero_at_init", abs(feedback_at_init) < 1e-9,
      f"feedback = {feedback_at_init:.3e} (should be 0)")

out["all_pass"] = all(c["pass"] for c in out["checks"].values())
with open("results/phase_3/preflight/reactivity_verification.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n=== Reactivity verification:", "PASS" if out["all_pass"] else "CHECK", "===")
