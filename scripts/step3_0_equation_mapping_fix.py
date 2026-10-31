"""Fix Step 3.0.B — auto-detect actual term names in source modules."""
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

# Broad patterns per module — several alternatives per term, plus fallbacks
checks = {
    "kinetics": [
        ("pke_rhs function",     [r"def\s+pke_rhs"]),
        ("beta_eff term",        [r"beta_eff"]),
        ("Lambda term",          [r"\bLambda\b"]),
        ("precursor lambda_i",   [r"lambda_(decay|i)", r"lam\b", r"lambda_"]),
        ("q(t) source term",     [r"include_q", r"q_t", r"\bq\b.*source", r"source_term"]),
        ("equilibrium C_i0",     [r"equilibrium_precursors", r"C_eq", r"beta_i.*n0"]),
    ],
    "thermal_hydraulics": [
        ("th_rhs function",      [r"def\s+th_rhs"]),
        ("fuel heat balance",    [r"M_f", r"c_p_f", r"\bf_f\b"]),
        ("coolant balance",      [r"M_c", r"c_p_c", r"m_dot", r"mflow"]),
        ("heat transfer UA",     [r"\bU\b.*\bA\b", r"\bUA\b", r"U\s*\*\s*A"]),
        # Broad — match any 2* combined with flow-related name
        ("factor 2 mass flow",   [r"2\s*\*\s*m", r"2\.0\s*\*\s*m", r"2\s*\*\s*mass", r"two.*mass", r"\* *2 *\* *m"]),
        ("T_in inlet",           [r"T_in", r"\binlet\b"]),
        ("feedback_reactivity",  [r"def\s+feedback_reactivity"]),
    ],
    "reactivity": [
        ("make_rho_total",       [r"def\s+make_rho_total"]),
        ("alpha_f Doppler",      [r"alpha_f", r"doppler", r"\balpha_?f\b"]),
        ("alpha_c coolant",      [r"alpha_c", r"coolant.*alpha", r"\balpha_?c\b"]),
        ("T_f_ref",              [r"T_f_ref", r"Tf_ref", r"T_f.*ref", r"fuel.*ref"]),
        ("T_c_ref",              [r"T_c_ref", r"Tc_ref", r"T_c.*ref", r"coolant.*ref"]),
    ],
    "solver": [
        ("coupled_rhs",          [r"def\s+coupled_rhs"]),
        ("9-state ordering",     [r"y\[0\]", r"y\[1:7\]", r"y\[7\]", r"y\[8\]"]),
        ("state vector doc",     [r"state.*vector", r"\[n,", r"n,\s*C"]),
    ],
}

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
            # Fallback: try searching the whole file content for ANY variant
            text = path.read_text()
            mapping.append({
                "module": str(path), "term": label,
                "line_number": None, "source_line": "",
                "status": "NOT_FOUND",
                "hint": f"Looked for: {patterns}. Check module content.",
            })

with open("results/phase_3/preflight/equation_to_source_mapping.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["module","term","line_number","source_line","status"])
    w.writeheader()
    for m in mapping:
        w.writerow({k: m.get(k, "") for k in w.fieldnames})

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

# Markdown
with open("results/phase_3/preflight/equation_to_source_mapping.md", "w") as f:
    f.write("# Equation-to-Source-Code Mapping\n\n")
    f.write(f"**Total:** {len(mapping)} | **Found:** {found} | **Missing:** {len(missing)}\n\n")
    f.write("| Module | Term | Line | Source | Status |\n")
    f.write("|--------|------|------|--------|--------|\n")
    for m in mapping:
        ln = m["line_number"] if m["line_number"] else "—"
        src = m["source_line"].replace("|", "\\|") if m["source_line"] else "—"
        f.write(f"| {m['module']} | {m['term']} | {ln} | `{src}` | {m['status']} |\n")
    if missing:
        f.write(f"\n## ⚠ Missing Terms\n\n")
        f.write("These terms were not found with the tested patterns. **Investigate manually** "
                "(do not fabricate the mapping):\n\n")
        for t in missing:
            f.write(f"- `{t}`\n")
        f.write("\n**Action required:**\n")
        f.write("```bash\n")
        f.write("grep -n 'alpha\\|feedback\\|ref' src/physics/reactivity.py\n")
        f.write("grep -n '2\\s*\\*\\|m_dot\\|mass' src/physics/thermal_hydraulics.py\n")
        f.write("```\n")
        f.write("Add the actual found lines to `equation_to_source_mapping.csv` manually if patterns don't match.\n")

print(json.dumps(summary, indent=2))
print("\n=== Equation mapping:", "PASS" if summary["all_pass"] else "INCOMPLETE", "===")
if missing:
    print("\nMissing:", missing)
    print("Run the grep diagnostics in the output .md file before declaring Gate 3.0 pass.")
