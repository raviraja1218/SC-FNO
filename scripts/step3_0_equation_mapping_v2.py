"""Step 3.0.B (v2) — Correct module attribution for each equation term."""
import csv
import json
import re
from pathlib import Path

# CORRECTED: terms are attributed to the module that ACTUALLY DEFINES them,
# not the module that composes them. reactivity.py imports feedback_reactivity
# from thermal_hydraulics.py; it does not redefine alphas.
checks = {
    "kinetics": [
        ("pke_rhs function",     [r"def\s+pke_rhs"]),
        ("beta_eff term",        [r"beta_eff"]),
        ("Lambda term",          [r"\bLambda\b"]),
        ("precursor lambda_i",   [r"lambda_"]),
        ("q(t) source term",     [r"include_q", r"q_t"]),
        ("equilibrium C_i0",     [r"equilibrium_precursors", r"C_eq", r"beta_i.*n0"]),
    ],
    "thermal_hydraulics": [
        ("th_rhs function",      [r"def\s+th_rhs"]),
        ("fuel heat balance",    [r"M_f", r"c_p_f", r"\bf_f\b"]),
        ("coolant balance",      [r"M_c", r"c_p_c", r"m_dot"]),
        ("heat transfer UA",     [r"\bU\s*\*\s*A\b", r"U\s*\*\s*th\.A", r"th\.U\s*\*\s*th\.A"]),
        # CORRECTED: factor-2 is in thermal_hydraulics.py line 39
        ("factor 2 mass flow",   [r"2\.0\s*\*\s*th\.m_dot", r"2\.0\s*\*\s*m_dot", r"2\s*\*\s*th\.m_dot"]),
        ("T_in inlet",           [r"T_in"]),
        # CORRECTED: feedback_reactivity is defined here, not in reactivity.py
        ("feedback_reactivity",  [r"def\s+feedback_reactivity"]),
        # CORRECTED: alphas are defined as fields here
        ("alpha_f Doppler",      [r"alpha_f"]),
        ("alpha_c coolant",      [r"alpha_c"]),
        ("T_f_ref",              [r"T_f_ref"]),
        ("T_c_ref",              [r"T_c_ref"]),
    ],
    "reactivity": [
        ("make_rho_total",       [r"def\s+make_rho_total"]),
        # reactivity.py composes: imports feedback_reactivity from thermal_hydraulics
        ("feedback import",      [r"from\s+\.thermal_hydraulics\s+import.*feedback_reactivity"]),
        ("rho_total composition",[r"rho_ext_fn\(t\).*feedback_reactivity", r"feedback_reactivity\(T_f,\s*T_c"]),
    ],
    "solver": [
        ("coupled_rhs",          [r"def\s+coupled_rhs"]),
        ("9-state ordering",     [r"y\[0\]", r"y\[1:7\]", r"y\[-2\]", r"y\[-1\]"]),
    ],
}

sources = {name: Path(f"src/physics/{name}.py") if name != "solver" else Path("src/data/solver.py")
           for name in checks}
for name, p in sources.items():
    assert p.exists(), f"Missing: {p}"

def find_lines(path, patterns):
    hits = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        for pat in patterns:
            if re.search(pat, line):
                hits.append((i, line.rstrip()))
                break
    return hits

mapping = []
for module, pats in checks.items():
    path = sources[module]
    for label, patterns in pats:
        hits = find_lines(path, patterns)
        if hits:
            lineno, line = hits[0]
            mapping.append({
                "module": str(path), "term": label,
                "line_number": lineno, "source_line": line.strip()[:200],
                "status": "FOUND",
            })
        else:
            mapping.append({
                "module": str(path), "term": label,
                "line_number": None, "source_line": "",
                "status": "NOT_FOUND",
            })

with open("results/phase_3/preflight/equation_to_source_mapping.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["module","term","line_number","source_line","status"])
    w.writeheader()
    w.writerows(mapping)

found = sum(1 for m in mapping if m["status"] == "FOUND")
missing = [f"{m['module']}::{m['term']}" for m in mapping if m["status"] == "NOT_FOUND"]

summary = {
    "total_terms": len(mapping),
    "found": found,
    "not_found": len(missing),
    "missing_terms": missing,
    "all_pass": len(missing) == 0,
    "note": "feedback_reactivity and alphas are defined in thermal_hydraulics.py, "
            "reactivity.py imports and composes them. Attribution corrected.",
}
with open("results/phase_3/preflight/equation_to_source_mapping.json", "w") as f:
    json.dump(summary, f, indent=2)

with open("results/phase_3/preflight/equation_to_source_mapping.md", "w") as f:
    f.write("# Equation-to-Source-Code Mapping (v2)\n\n")
    f.write(f"**Total:** {len(mapping)} | **Found:** {found} | **Missing:** {len(missing)}\n\n")
    f.write(f"> **Note:** `feedback_reactivity`, `alpha_f`, `alpha_c`, `T_f_ref`, `T_c_ref` "
            f"are defined in `thermal_hydraulics.py` (not `reactivity.py`, which imports them). "
            f"The coolant equation factor-of-2 is at `thermal_hydraulics.py:39`.\n\n")
    f.write("| Module | Term | Line | Source | Status |\n")
    f.write("|--------|------|------|--------|--------|\n")
    for m in mapping:
        ln = m["line_number"] if m["line_number"] else "—"
        src = m["source_line"].replace("|", "\\|") if m["source_line"] else "—"
        f.write(f"| {m['module']} | {m['term']} | {ln} | `{src}` | {m['status']} |\n")

print(json.dumps(summary, indent=2))
print("\n=== Equation mapping:", "PASS" if summary["all_pass"] else "INCOMPLETE", "===")
