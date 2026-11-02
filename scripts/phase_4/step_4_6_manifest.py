"""Step 4.6.D — Build Phase 4 final manifest."""
import hashlib, json
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# Verify comparison CSV exists
comp = Path("results/phase_4/final_comparison/unified_phase_4_comparison.csv")
assert comp.exists(), "Comparison CSV missing — run Step 4.6.B first"

manifest = {
    "phase": "Phase 4",
    "timestamp": "2026-11-02",
    "milestone": "PHASE_4_LOCKED — MIXED RESULT: PHASE 4A SUCCESS, PHASE 4B TRAINING-PROTOCOL LIMITATION",
    "selected_model": "Phase4A.1",
    "selected_model_sha": sha("models/sc_fno/phase_4/phase_4a/phase_4a_final.pt"),
    "lock_references": {
        "phase_1": "1d9b320",
        "phase_2": "0c9dfd7ced7454342e09280b6307f83d42eea89e",
        "phase_3": "bc80727",
        "phase_4a_closure": "dea218a",
        "phase_4a_tag_target": "2c8443e0234ab5670386bfc6755cc8d157ea7de0",
        "phase_4b_closure": "dd392bb",
        "phase_4b_tag_target": "fbb36c98995d40f8a23215435f7c45c88df3e6f4",
    },
    "model_checkpoints": {
        n: {"path": p, "sha256": sha(p)}
        for n, p in {
            "P2-FNO": "models/baseline/fno_best.pt",
            "P3-PI-FNO": "models/sc_fno/phase_3/pi_fno/best_val.pt",
            "Phase4A.1": "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt",
            "P4-A": "models/sc_fno/phase_4/phase_4b/p4_a_best.pt",
            "P4-B": "models/sc_fno/phase_4/phase_4b/p4_b_best.pt",
            "P4-C": "models/sc_fno/phase_4/phase_4b/p4_c_best.pt",
            "P4-D": "models/sc_fno/phase_4/phase_4b/p4_d_best.pt",
        }.items()
    },
    "test_set_sha": "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2",
    "outstanding_phase_5_questions": [
        "True-forcing physics evaluation (requires retraining)",
        "Boundary artifact resolution (FFT wraparound 4.58×–5.17×)",
        "Bug A root-cause isolation (patience vs lambda schedule vs other)",
    ],
    "phase_5_authorized": True,
    "completion_report": "reports/phase_reports/phase_4_completion_report.md",
    "unified_comparison_csv": str(comp),
    "unified_comparison_sha256": sha(comp),
}

out = Path("results/phase_4/phase_4_final_manifest.json")
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[OK] Manifest: {out}")
print(f"     Selected model: Phase4A.1 ({manifest['selected_model_sha'][:16]})")
print(f"     Phase 5 authorized: {manifest['phase_5_authorized']}")
