"""Step 4.6.E — Gate 4 verification (13 criteria)."""
import json
from pathlib import Path
import subprocess

checks = []
def check(name, ok, detail=""):
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# 1. Phase 4A tag
try:
    subprocess.check_call(["git", "rev-parse", "phase-4a-locked"],
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    check("phase-4a-locked tag", True, "present")
except subprocess.CalledProcessError:
    check("phase-4a-locked tag", False, "missing")

# 2. Phase 4B tag
try:
    subprocess.check_call(["git", "rev-parse", "phase-4b-locked"],
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    check("phase-4b-locked tag", True, "present")
except subprocess.CalledProcessError:
    check("phase-4b-locked tag", False, "missing")

# 3. All 7 model SHAs — verify via locked_verification
lv = json.load(open("results/phase_4/closure/lock_verification.json"))
sha_checks = [c for k, c in lv["checks"].items() if "ckpt SHA" in k]
check("7 model SHAs unchanged", all(c["pass"] for c in sha_checks),
      f"{sum(c['pass'] for c in sha_checks)}/{len(sha_checks)}")

# 4. Test set SHA
ts_ok = lv["checks"].get("Test set", {}).get("pass", False)
check("Test set SHA", ts_ok, "a71514ea...")

# 5. Unified comparison CSV
comp = Path("results/phase_4/final_comparison/unified_phase_4_comparison.csv")
check("Unified comparison CSV", comp.exists() and comp.stat().st_size > 100,
      f"{comp.stat().st_size} bytes" if comp.exists() else "missing")

# 6. Completion report 6 sections
report = Path("reports/phase_reports/phase_4_completion_report.md")
if report.exists():
    text = report.read_text()
    has_6 = all(f"Section {i}" in text for i in range(1, 7))
else:
    has_6 = False
check("Completion report 6 sections", has_6, str(report))

# 7. Reviewer's Phase 4B wording verbatim
reviewer_wording = "Phase 4B did not achieve satisfactory physics-residual convergence under the frozen training protocol"
check("Reviewer Phase 4B verbatim", report.exists() and reviewer_wording in report.read_text(),
      "verbatim" if report.exists() else "missing")

# 8. Final model selection documented
has_selection = report.exists() and "Phase 4A.1" in report.read_text() and "selected final model" in report.read_text().lower()
check("Final model selection documented", has_selection, "Phase4A.1")

# 9. Master manifest
man = Path("results/phase_4/phase_4_final_manifest.json")
man_data = json.load(open(man)) if man.exists() else {}
check("Phase 4 master manifest", man.exists() and man_data.get("selected_model") == "Phase4A.1",
      f"selected: {man_data.get('selected_model')}")

# 10. Locked paths clean
status = subprocess.check_output(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "results/phase_3",
     "results/phase_4/phase_4a", "results/phase_4/phase_4b",
     "models/baseline", "models/sc_fno/phase_3",
     "models/sc_fno/phase_4/phase_4a", "models/sc_fno/phase_4/phase_4b"],
    text=True).strip()
check("Locked paths clean", status == "", "empty" if not status else status[:100])

# 11. No retraining after Phase 4B closure
import os
latest_ckpt = max(os.path.getmtime(p) for p in Path("models/sc_fno/phase_4").rglob("*.pt"))
last_commit = float(subprocess.check_output(["git", "log", "-1", "--format=%ct", "dd392bb"], text=True).strip())
check("No retraining after Phase 4B closure", latest_ckpt <= last_commit + 60, "verified")

# 12. Both closure reports present
p4a_report = Path("results/phase_4/closure/phase_4a_completion_report.md")
p4b_report = Path("reports/phase_reports/phase_4b_completion_report.md")
check("Both Phase 4A and 4B closure reports present",
      p4a_report.exists() and p4b_report.exists(),
      f"4A: {p4a_report.exists()}, 4B: {p4b_report.exists()}")

# 13. Phase 5 authorization declared
check("Phase 5 authorization", man_data.get("phase_5_authorized") is True, "true")

n_pass = sum(1 for c in checks if c["pass"])
result = {"gate": "4", "n_checks": len(checks), "n_pass": n_pass,
          "all_pass": n_pass == len(checks), "checks": checks}
json.dump(result, open("results/phase_4/closure/gate_4_verification.json", "w"), indent=2)

print(f"\n=== Gate 4: {n_pass}/{len(checks)} PASS ===")
assert result["all_pass"], "Gate 4 verification FAILED"
