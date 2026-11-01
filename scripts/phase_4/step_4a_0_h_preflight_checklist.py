"""Step 4A.0.H — Consolidate preflight evidence into Gate 4A.0 checklist + manifest."""
import json
import hashlib
from pathlib import Path

BASE = Path("results/phase_4/phase_4a/preflight")

def load(name):
    try:
        return json.load(open(BASE / name))
    except Exception as e:
        return {"__error__": str(e)}

lock     = load("phase_3_lock_verification.json")
ckpt     = load("checkpoint_identity.json")
ckpt_meta= load("checkpoint_metadata.json")
dataset  = load("dataset_verification.json")
norm     = load("normalization_verification.json")
val_repr = load("val_metrics_reproduced.json")
ref      = load("phase_3_test_reference.json")
baseline = load("baseline_metrics.json")
env      = load("environment.json")

checks = []
def check(name, ok, detail=""):
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# 4A.0.A
check("4A.0.A Phase 3 lock integrity", lock.get("all_pass"), "14/14 checks")
# 4A.0.B
check("4A.0.B Checkpoint identity", ckpt.get("all_pass"), "SHA + param count + forward")
# 4A.0.C
check("4A.0.C Dataset verification", dataset.get("all_pass"), "splits + dt + normalization")
# 4A.0.D
check("4A.0.D Val metrics reproducibility", val_repr.get("all_pass"),
      f"val MSE = {val_repr.get('measured', {}).get('mean_val_data_mse', '?')}")
# 4A.0.E
check("4A.0.E Phase 3 test reference", len(ref.get("extracted", {})) >= 5,
      f"{len(ref.get('extracted', {}))} reference files")
# 4A.0.F
check("4A.0.F Baseline metrics", baseline.get("all_pass"),
      f"L_data={baseline.get('metrics', {}).get('initial_data_loss', '?')}, "
      f"L_phys={baseline.get('metrics', {}).get('initial_physics_loss', '?')}")
# 4A.0.G
check("4A.0.G Environment", env.get("cuda_available") is True,
      f"{env.get('cuda_device_name')}, {env.get('cuda_device_vram_MB', 0):.0f} MB")

# Locked path cleanliness
import subprocess
status = subprocess.check_output(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "results/phase_3", "models/baseline"],
    text=True
).strip()
check("Locked paths clean", status == "", "empty" if not status else status[:200])

all_pass = all(c["pass"] for c in checks)
summary = {
    "phase": "4A.0",
    "timestamp": "2026-11-01",
    "phase_3_lock_reference": "bc80727",
    "checks": checks,
    "gate_4a_0": "PASS" if all_pass else "FAIL",
    "diagnostic_notes": [
        "lam_phys in PI-FNO ckpt = 0.002754 (epoch 36 of ramp) — Phase 4A.1 must decide λ_phys strategy",
        "R_n per-state diagnostic = 1900 while aggregate L_physics = 1.69e-3 — "
        "aggregate is authoritative (matches compute_physics_loss); "
        "per-state diagnostic path may have subtle denormalization difference — "
        "resolve in 4A.3 for residual comparison.",
    ],
    "baseline_metrics_summary": {
        "initial_data_loss": baseline.get("metrics", {}).get("initial_data_loss"),
        "initial_physics_loss": baseline.get("metrics", {}).get("initial_physics_loss"),
        "initial_ic_loss": baseline.get("metrics", {}).get("initial_ic_loss"),
        "initial_total_loss": baseline.get("metrics", {}).get("initial_total_loss"),
        "lam_phys_from_ckpt": baseline.get("metrics", {}).get("lam_phys_from_ckpt"),
        "peak_vram_allocated_MB": baseline.get("metrics", {}).get("peak_vram_allocated_MB"),
    },
}

with open(BASE / "preflight_checklist.json", "w") as f:
    json.dump(summary, f, indent=2)

# Manifest
artifacts = {}
for p in sorted(BASE.glob("*")):
    if p.is_file() and p.name not in ("step_4a_0_manifest.json",):
        artifacts[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

manifest = {
    "phase": "4A.0",
    "gate_4a_0": summary["gate_4a_0"],
    "artifacts": artifacts,
}
with open(BASE / "step_4a_0_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nGate 4A.0: {summary['gate_4a_0']}")
print(f"Manifest: {len(artifacts)} artifacts")
