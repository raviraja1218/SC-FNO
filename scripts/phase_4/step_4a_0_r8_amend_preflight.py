"""Step 4A.0.R8 — Amend preflight checklist with FIXED baseline.

Updates:
  - preflight_checklist.json — replaces buggy baseline with FIXED, adds note
  - step_4a_0_manifest.json — regenerated with all current artifacts
"""
import json
import hashlib
from pathlib import Path

BASE = Path("results/phase_4/phase_4a/preflight")

# Load FIXED baseline
fixed = json.load(open(BASE / "baseline_metrics_canonical_FIXED.json"))

# Load existing checklist
checklist = json.load(open(BASE / "preflight_checklist.json"))

# Amend checklist
checklist["loader_bug_disclosure"] = {
    "bug_id": "4A.0.R7",
    "discovered": "2026-11-01",
    "description": (
        "Phase 3 PI-FNO checkpoint saved state_dict with `fno.fno.*` prefix "
        "(SCFNOModel wrapping PaddedFNO(fno, pad_size=0)). The Phase 4A.0.F "
        "baseline script and Phase 4A.1 training script loaded into a "
        "single-wrapped SCFNOModel (`fno.*` prefix). Result: 40 checkpoint "
        "keys (`fno.fno.*`) were unexpected and never loaded; the FNO "
        "backbone stayed at Phase 2 `fno_best.pt` initialization. Phase 3 "
        "FNO weights differ from Phase 2 by up to 20% per key "
        "(max diff 0.203 on one weight, sum diff 2.49 over 40 keys). "
        "The bug was real and consequential."
    ),
    "fix": "load_phase3_pi_fno_model (PaddedFNO-wrapped, 0 missing / 0 unexpected)",
    "regression_test": "test_t10_phase3_pi_fno_loads_bit_exact",
    "baseline_recomputed": "baseline_metrics_canonical_FIXED.json",
    "invalid_artifacts_archived": [
        "results/phase_4/phase_4a/preflight/_INVALID_buggy_load/",
        "results/phase_4/phase_4a/optimizer/_INVALID_buggy_load/",
        "models/sc_fno/phase_4/phase_4a/_INVALID_buggy_load/",
    ],
}

# Update the baseline reference in the checklist
if "baseline_metrics_summary" in checklist:
    checklist["baseline_metrics_summary_original_buggy"] = checklist["baseline_metrics_summary"]
checklist["baseline_metrics_summary_FIXED"] = {
    "loader": fixed["loader"],
    "n_val_scenarios": fixed["n_val_scenarios"],
    "initial_data_loss": fixed["initial_data_loss"],
    "initial_physics_loss": fixed["initial_physics_loss"],
    "initial_ic_loss": fixed["initial_ic_loss"],
    "lam_phys_from_ckpt": fixed["lam_phys_from_ckpt"],
    "delta_data_vs_phase3": fixed["delta_data_vs_phase3"],
}

# Note on the 4A.0.F check
for c in checklist.get("checks", []):
    if c["name"] == "4A.0.F Baseline metrics":
        c["detail"] = (
            f"FIXED loader; L_data={fixed['initial_data_loss']:.6f}, "
            f"L_phys={fixed['initial_physics_loss']:.6e} "
            f"(buggy version archived)"
        )

with open(BASE / "preflight_checklist.json", "w") as f:
    json.dump(checklist, f, indent=2)
print("Amended preflight_checklist.json")

# Recompute manifest (SHA-256 over all current artifacts, excluding _INVALID and manifest itself)
artifacts = {}
for p in sorted(BASE.rglob("*")):
    if p.is_file() and "_INVALID" not in str(p) and p.name != "step_4a_0_manifest.json":
        artifacts[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

manifest = {
    "phase": "4A.0",
    "gate_4a_0": checklist.get("gate_4a_0", "PASS"),
    "amendments": [
        "R7 — loader bug fix (PaddedFNO wrap)",
        "R8 — preflight checklist amended with FIXED baseline",
    ],
    "phase_1_lock_reference": "1d9b320",
    "phase_3_lock_reference": "bc80727",
    "starting_checkpoint_sha": "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
    "invalid_artifacts_note": (
        "Baseline measurements and Phase 4A.1 training that used the buggy "
        "single-wrap loader are archived under _INVALID_buggy_load/. They are "
        "preserved as evidence of the bug, not as valid measurements."
    ),
    "artifacts": artifacts,
}
with open(BASE / "step_4a_0_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest: {len(artifacts)} artifacts (excluding _INVALID)")
print("Amendments: R7, R8")
