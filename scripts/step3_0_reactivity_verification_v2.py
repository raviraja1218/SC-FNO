"""Step 3.0.I (v2) — Verify reactivity feedback across both modules."""
import json
import re
import yaml
from pathlib import Path

out = {"checks": {}, "sources": {}, "all_pass": False}
def check(name, ok, detail=""):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

reactivity_src = Path("src/physics/reactivity.py").read_text()
th_src         = Path("src/physics/thermal_hydraulics.py").read_text()
out["sources"]["reactivity_py"] = reactivity_src
out["sources"]["thermal_hydraulics_py"] = th_src

# --- reactivity.py: composition only ---
check("make_rho_total_exists", "def make_rho_total" in reactivity_src, "function defined")
check("imports_feedback_from_th",
      bool(re.search(r"from\s+\.thermal_hydraulics\s+import.*feedback_reactivity", reactivity_src)),
      "imports feedback_reactivity from thermal_hydraulics")
check("composes_rho_ext_plus_feedback",
      bool(re.search(r"rho_ext_fn\(t\).*feedback_reactivity|feedback_reactivity\(T_f,\s*T_c", reactivity_src)),
      "rho_total = rho_ext + feedback")
check("feedback_uses_y_minus_2_y_minus_1",
      ("y[-2]" in reactivity_src) and ("y[-1]" in reactivity_src),
      "T_f, T_c at positions -2, -1")

# --- thermal_hydraulics.py: parameter definitions and feedback formula ---
check("alpha_f_in_TH", "alpha_f" in th_src, "alpha_f field in THParameters")
check("alpha_c_in_TH", "alpha_c" in th_src, "alpha_c field in THParameters")
check("T_f_ref_in_TH", "T_f_ref" in th_src, "T_f_ref field in THParameters")
check("T_c_ref_in_TH", "T_c_ref" in th_src, "T_c_ref field in THParameters")
check("feedback_reactivity_defined_in_TH",
      "def feedback_reactivity" in th_src, "feedback_reactivity defined")
check("feedback_formula_uses_alphas_and_refs",
      bool(re.search(r"alpha_f\s*\*\s*\(T_f\s*-\s*.*T_f_ref", th_src)),
      "formula: alpha_f*(T_f - T_f_ref) + alpha_c*(T_c - T_c_ref)")

# --- Coolant equation factor of 2 ---
check("factor_2_in_coolant_equation",
      bool(re.search(r"2\.0\s*\*\s*th\.m_dot\s*\*\s*th\.c_p_c", th_src)),
      "dT_c = (... - 2.0*m_dot*c_p_c*(T_c - T_in)) / (M_c*c_p_c)")

# --- Config cross-check ---
th_cfg = yaml.safe_load(open("configs/thermal_hydraulics.v4.yaml"))
fb = th_cfg["reactivity_feedback"]
def scalar(v):
    return float(v[0]) if isinstance(v, list) else float(v)

check("config_alpha_f_sign_negative", scalar(fb["alpha_f"]) < 0,
      f"alpha_f = {scalar(fb['alpha_f'])} (Doppler < 0)")
check("config_alpha_c_sign_negative", scalar(fb["alpha_c"]) < 0,
      f"alpha_c = {scalar(fb['alpha_c'])} (coolant < 0)")
check("config_T_f_ref_900", abs(scalar(fb["T_f_ref"]) - 900.0) < 1e-6,
      f"T_f_ref = {scalar(fb['T_f_ref'])}")
check("config_T_c_ref_583", abs(scalar(fb["T_c_ref"]) - 583.0) < 1e-6,
      f"T_c_ref = {scalar(fb['T_c_ref'])}")

# --- Feedback zero at initial state ---
kin_cfg = yaml.safe_load(open("configs/reactor_parameters.v3.yaml"))
init = kin_cfg["initial_state"]
T_f0 = scalar(init["T_f0"])
T_c0 = scalar(init["T_c0"])
feedback_at_init = (
    scalar(fb["alpha_f"]) * (T_f0 - scalar(fb["T_f_ref"]))
    + scalar(fb["alpha_c"]) * (T_c0 - scalar(fb["T_c_ref"]))
)
check("feedback_zero_at_initial_state", abs(feedback_at_init) < 1e-9,
      f"feedback(T_f0, T_c0) = {feedback_at_init:.3e}")

out["all_pass"] = all(c["pass"] for c in out["checks"].values())
with open("results/phase_3/preflight/reactivity_verification.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n=== Reactivity verification:", "PASS" if out["all_pass"] else "FAIL", "===")
