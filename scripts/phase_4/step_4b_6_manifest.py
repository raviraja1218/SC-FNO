"""Step 4B.6.D — Build Phase 4B final manifest."""
import hashlib
import json
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# Load inventory
import csv
inventory = list(csv.DictReader(open("results/phase_4/phase_4b/closure/phase_4b_artifact_inventory.csv")))

manifest = {
    "phase": "4B",
    "timestamp": "2026-11-02",
    "milestone": "PHASE_4B_LOCKED — NEGATIVE RESULT: TRAINING-PROTOCOL LIMITATION",
    "reviewer_acceptance": {
        "step_4b_5_accepted_at": "75dce8f",
        "conditions_closed": 5,
        "corrections_applied": 8,
    },
    "lock_references": {
        "phase_1": "1d9b320",
        "phase_2": "0c9dfd7ced7454342e09280b6307f83d42eea89e",
        "phase_3": "bc80727",
        "phase_4a": "dea218a",
    },
    "phase_4b_commit_chain": [
        "bec0f63", "b8ae156", "3fd0af0", "b412f20", "10a7e6d", "9d0e0da",
        "836ddf7", "3dd0ed4", "7e456aa", "985c0a9", "d226447",
        "b5f94a0", "e96d94f", "109cdfc", "75dce8f",
    ],
    "model_checkpoints": {
        "P2-FNO": {"path": "models/baseline/fno_best.pt", "sha256": sha("models/baseline/fno_best.pt")},
        "P3-PI-FNO": {"path": "models/sc_fno/phase_3/pi_fno/best_val.pt", "sha256": sha("models/sc_fno/phase_3/pi_fno/best_val.pt")},
        "Phase4A.1": {"path": "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt", "sha256": sha("models/sc_fno/phase_4/phase_4a/phase_4a_final.pt")},
        "P4-A": {"path": "models/sc_fno/phase_4/phase_4b/p4_a_best.pt", "sha256": sha("models/sc_fno/phase_4/phase_4b/p4_a_best.pt")},
        "P4-B": {"path": "models/sc_fno/phase_4/phase_4b/p4_b_best.pt", "sha256": sha("models/sc_fno/phase_4/phase_4b/p4_b_best.pt")},
        "P4-C": {"path": "models/sc_fno/phase_4/phase_4b/p4_c_best.pt", "sha256": sha("models/sc_fno/phase_4/phase_4b/p4_c_best.pt")},
        "P4-D": {"path": "models/sc_fno/phase_4/phase_4b/p4_d_best.pt", "sha256": sha("models/sc_fno/phase_4/phase_4b/p4_d_best.pt")},
    },
    "test_set_sha": "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2",
    "outstanding_phase_5_questions": [
        "True-forcing physics evaluation (requires retraining)",
        "Boundary artifact resolution (FFT wraparound 4.5–5.2×)",
        "Bug A root-cause isolation (patience vs lambda schedule vs other)",
    ],
    "artifacts": {r["path"]: r["sha256"] for r in inventory},
    "artifact_count": len(inventory),
    "completion_report": "reports/phase_reports/phase_4b_completion_report.md",
}

out = Path("results/phase_4/phase_4b/phase_4b_final_manifest.json")
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[OK] Manifest: {out} ({len(inventory)} artifacts)")
print(f"     Milestone: {manifest['milestone']}")
