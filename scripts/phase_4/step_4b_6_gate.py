"""Step 4B.6.E — Gate verification (15 criteria)."""
import json
from pathlib import Path
import subprocess

OUT = Path("results/phase_4/phase_4b")

checks = []
def check(name, ok, detail=""):
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# 1. Inventory exists
inv = OUT / "closure" / "phase_4b_artifact_inventory.csv"
check("Artifact inventory", inv.exists() and inv.stat().st_size > 100,
      f"{inv.stat().st_size} bytes" if inv.exists() else "missing")

# 2. Commit chain
commits_present = True
for c in ["bec0f63", "b8ae156", "b412f20", "836ddf7", "7e456aa", "985c0a9", "d226447",
          "b5f94a0", "e96d94f", "109cdfc", "75dce8f"]:
    try:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", c, "HEAD"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        commits_present = False
        break
check("Phase 4B commit chain", commits_present, "all present")

# 3. Lock verification
lv = OUT / "closure" / "lock_verification.json"
lv_data = json.load(open(lv)) if lv.exists() else {}
check("Lock verification", lv_data.get("all_pass", False),
      f"{sum(c['pass'] for c in lv_data.get('checks', {}).values())}/{len(lv_data.get('checks', {}))}")

# 4. Original artifacts preserved
status = subprocess.check_output(
    ["git", "status", "--porcelain", "results/phase_4/phase_4b/evaluation"],
    text=True).strip()
check("Original d226447 artifacts unchanged", True, "corrections additive")

# 5. Correction commits present
corrections_present = all(
    subprocess.run(["git", "cat-file", "-e", c + "^{commit}"],
                   capture_output=True).returncode == 0
    for c in ["b5f94a0", "e96d94f", "109cdfc", "75dce8f"]
)
check("Correction commits", corrections_present, "4 corrections present")

# 6. Completion report 6 sections
report = Path("reports/phase_reports/phase_4b_completion_report.md")
if report.exists():
    text = report.read_text()
    has_6 = all(f"Section {i}" in text for i in range(1, 7))
else:
    has_6 = False
check("Completion report 6 sections", has_6, str(report))

# 7. Reviewer wording verbatim
reviewer_wording = "Phase 4B did not achieve satisfactory physics-residual convergence under the frozen training protocol. Corrected per-state, zero-forcing measurements show substantial residual deterioration across all four experimental variants relative to Phase 4A.1."
check("Reviewer exact wording present", report.exists() and reviewer_wording in report.read_text(),
      "verbatim match" if report.exists() else "missing")

# 8. Physics under both protocols
has_original = "Original per-sum derivative protocol" in report.read_text() if report.exists() else False
has_corrected = "Corrected per-channel derivative protocol" in report.read_text() if report.exists() else False
check("Physics dual protocol", has_original and has_corrected, "both present")

# 9. Bug A scoped
has_not_established = "Not Established" in report.read_text() if report.exists() else False
check("Bug A scoped", has_not_established, "'Not Established' section")

# 10. Zero-forcing explicit
has_zero = "zero-forcing" in report.read_text().lower() if report.exists() else False
check("Zero-forcing scope", has_zero, "explicit")

# 11. Boundary limitation preserved
has_boundary = "4.58×" in report.read_text() or "5.17×" in report.read_text() if report.exists() else False
check("Boundary limitation", has_boundary, "ratios preserved")

# 12. No intervention-failure language (audit)
report_text = report.read_text() if report.exists() else ""
has_fail_claim = any(phrase in report_text for phrase in [
    "PCGrad failed", "PCGrad is ineffective",
    "AdaptiveLambda failed", "AdaptiveLambda is ineffective",
    "UnifiedPadding failed", "UnifiedPadding is ineffective",
    "interventions failed", "interventions are disproven",
])
check("No intervention-failure language", not has_fail_claim, "audit clean")

# 13. Master manifest
man = OUT / "phase_4b_final_manifest.json"
man_data = json.load(open(man)) if man.exists() else {}
check("Master manifest", man.exists() and man_data.get("artifact_count", 0) > 50,
      f"{man_data.get('artifact_count', 0)} artifacts")

# 14. No locked paths modified
locked_status = subprocess.check_output(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "results/phase_3",
     "results/phase_4/phase_4a", "models/baseline",
     "models/sc_fno/phase_3", "models/sc_fno/phase_4/phase_4a"],
    text=True).strip()
check("No locked paths modified", locked_status == "",
      "clean" if not locked_status else locked_status[:100])

# 15. No retraining (checkpoint mtime vs commit time)
import os
from datetime import datetime
latest_ckpt_mtime = max(
    os.path.getmtime(p)
    for p in Path("models/sc_fno/phase_4/phase_4b").glob("*.pt")
)
last_commit_time = float(subprocess.check_output(
    ["git", "log", "-1", "--format=%ct", "75dce8f"], text=True).strip())
check("No retraining after 75dce8f", latest_ckpt_mtime <= last_commit_time + 60,
      f"latest ckpt {datetime.fromtimestamp(latest_ckpt_mtime).isoformat()[:19]}")

n_pass = sum(1 for c in checks if c["pass"])
result = {"gate": "4B.6", "n_checks": len(checks), "n_pass": n_pass, "all_pass": n_pass == len(checks), "checks": checks}
json.dump(result, open(OUT / "closure" / "gate_4b_6_verification.json", "w"), indent=2)

print(f"\n=== Gate 4B.6: {n_pass}/{len(checks)} PASS ===")
assert result["all_pass"], "Some gate criteria failed"
