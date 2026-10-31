"""Step 3.0.B — Map every governing equation term to its source in locked Phase 1 modules."""
import csv
import json
import re
from pathlib import Path

sources = {
    "kinetics":            Path("src/physics/kinetics.py"),
    "thermal_hydraulics":  Path("src/physics/thermal_hydraulics.py"),
    "reactivity":          Path("src/physics/reactivity.py"),
    "solver":              Path("src/data/solver.py"),
}

# Sanity: files exist
for name, p in sources.items():
    assert p.exists(), f"Missing: {p}"

# Extract source excerpts around key terms
def find_lines(path, patterns):
    """Return (line_number, line_text) for lines matching any pattern."""
    hits = []
    text = path.read_text().splitlines()
    for i, line in enumerate(text, 1):
        for pat in patterns:
            if re.search(pat, line):
                hits.append((i, line.rstrip()))
                break
    return hits

# Required terms per module
checks = {
    "kinetics": [
        ("pke_rhs function",     r"def\s+pke_rhs"),
        ("beta_eff term",         r"beta_eff"),
        ("Lambda term",           r"Lambda"),
        ("precursor lambda_i",    r"lambda_decay|lambda_i"),
        ("q(t) source term",      r"\bq\b.*source|q_t|q\(t\)|include_q"),
        ("equilibrium C_i0",      r"equilibrium_precursors|C_eq|beta_i.*n0"),
    ],
    "thermal_hydraulics": [
        ("th_rhs function",       r"def\s+th_rhs"),
        ("fuel heat balance",     r"M_f|c_p_f|f_f"),
        ("coolant balance",       r"M_c|c_p_c|m_dot"),
        ("heat transfer UA",      r"\bU\b.*\bA\b|UA|U\s*\*\s*A"),
        ("factor 2 mass flow",    r"2\s*\*\s*m_dot|2\s*\*\s*m|coolant.*2"),
        ("T_in inlet",            r"T_in"),
        ("feedback_reactivity",   r"def\s+feedback_reactivity"),
    ],
    "reactivity": [
        ("make_rho_total",        r"def\s+make_rho_total"),
        ("alpha_f Doppler",       r"alpha_f"),
        ("alpha_c coolant",       r"alpha_c"),
        ("T_f_ref",               r"T_f_ref"),
        ("T_c_ref",               r"T_c_ref"),
    ],
    "solver": [
        ("coupled_rhs",           r"def\s+coupled_rhs"),
        ("9-state ordering",      r"y\[0\]|y\[1:7\]|y\[7\]|y\[8\]|n,?\s*C|C1.*C6.*Tf.*Tc"),
        ("state vector doc",      r"state.*vector|\[n,.*C1|n,\s*C"),
    ],
}

mapping = []
for module, pats in checks.items():
    path = sources[module]
    for label, pattern in pats:
        hits = find_lines(path, [pattern])
        status = "FOUND" if hits else "NOT_FOUND"
        for lineno, line in hits[:1]:
            mapping.append({
                "module": str(path),
                "term": label,
                "pattern": pattern,
                "line_number": lineno,
                "source_line": line.strip()[:200],
                "status": status,
            })
        if not hits:
            mapping.append({
                "module": str(path),
                "term": label,
                "pattern": pattern,
                "line_number": None,
                "source_line": "",
                "status": "NOT_FOUND",
            })

with open("results/phase_3/preflight/equation_to_source_mapping.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["module","term","pattern","line_number","source_line","status"])
    w.writeheader()
    w.writerows(mapping)

# Summary
found = sum(1 for m in mapping if m["status"] == "FOUND")
missing = [m["term"] for m in mapping if m["status"] == "NOT_FOUND"]

summary = {
    "total_terms": len(mapping),
    "found": found,
    "not_found": len(missing),
    "missing_terms": missing,
    "all_pass": len(missing) == 0,
}
with open("results/phase_3/preflight/equation_to_source_mapping.json", "w") as f:
    json.dump(summary, f, indent=2)

# Write markdown summary
with open("results/phase_3/preflight/equation_to_source_mapping.md", "w") as f:
    f.write("# Equation-to-Source-Code Mapping\n\n")
    f.write(f"**Total terms:** {len(mapping)}  \n")
    f.write(f"**Found:** {found}  \n")
    f.write(f"**Not found:** {len(missing)}  \n\n")
    f.write("## Full Mapping\n\n")
    f.write("| Module | Term | Line | Source Excerpt | Status |\n")
    f.write("|--------|------|------|----------------|--------|\n")
    for m in mapping:
        ln = m["line_number"] if m["line_number"] else "—"
        src = m["source_line"].replace("|", "\\|") if m["source_line"] else "—"
        f.write(f"| {m['module']} | {m['term']} | {ln} | `{src}` | {m['status']} |\n")
    if missing:
        f.write(f"\n## ⚠ Missing terms\n\n{missing}\n")

print(json.dumps(summary, indent=2))
print("\n=== Equation mapping:", "PASS" if summary["all_pass"] else "INCOMPLETE", "===")
if missing:
    print("Missing terms:", missing)
