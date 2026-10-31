"""Step 3.0.A — Verify all Phase 1 + Phase 2 locked artifacts."""
import json
import hashlib
import subprocess
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def git_rev_parse(ref):
    try:
        return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    except subprocess.CalledProcessError:
        return None

EXPECTED = {
    "phase1_aggregate": "9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6",
    "phase1_manifest":  "993581ef514932c5f3c341ce176a401a5247ebe7217e83f70d3ebe26dd3cff12",
    "baseline_ckpt":    "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889",
    "lock_tag":         "0c9dfd7ced7454342e09280b6307f83d42eea89e",
    "train_ids":        "16ad4670eee9555aec237c94d2d241e90b74b50ce8ea57eb07efe209fe852834",
    "val_ids":          "19b781283503b1a1cdb313dc68ce153086f18c25043f1ff5123e3cb3e3205bb6",
    "test_ids":         "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2",
    "normalization":    "4e832aa5cdaa940cb2e07bf64c5741cc5124ad40242dd9532da231ff257f6727",
}

result = {"checks": {}, "all_pass": False}

def check(name, condition, detail=""):
    result["checks"][name] = {"pass": bool(condition), "detail": detail}
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}: {detail}")

# 1. Phase 1 manifest SHA-256
actual = sha256_file("results/phase_1/phase_1_dataset_manifest.json")
check("phase1_manifest_sha256", actual == EXPECTED["phase1_manifest"], actual[:16] + "...")

# 2. Phase 1 manifest contents
m = json.load(open("results/phase_1/phase_1_dataset_manifest.json"))
check("phase1_aggregate_in_manifest",
      m["dataset"]["aggregate_sha256"] == EXPECTED["phase1_aggregate"],
      m["dataset"]["aggregate_sha256"][:16] + "...")
check("phase1_configs",
      m["config_versions"] == {"reactor_parameters": "v3", "thermal_hydraulics": "v4"},
      str(m["config_versions"]))

# 3. Baseline checkpoint
actual = sha256_file("models/baseline/fno_best.pt")
check("baseline_ckpt_sha256", actual == EXPECTED["baseline_ckpt"], actual[:16] + "...")

# 4. Lock tag
lock_sha = git_rev_parse("phase-2-lock")
check("phase2_lock_tag", lock_sha == EXPECTED["lock_tag"], (lock_sha or "MISSING")[:8])

# 5. Split files
for name, path in [
    ("train_ids", "results/phase_1/splitting/train_ids.csv"),
    ("val_ids",   "results/phase_1/splitting/validation_ids.csv"),
    ("test_ids",  "results/phase_1/splitting/test_ids.csv"),
    ("normalization", "results/phase_1/splitting/normalization_statistics.json"),
]:
    actual = sha256_file(path)
    check(name, actual == EXPECTED[name], actual[:16] + "...")

# 6. Locked paths clean
try:
    status = subprocess.check_output(
        ["git", "status", "--porcelain",
         "results/phase_1", "results/phase_2",
         "models/baseline",
         "configs/reactor_parameters.v3.yaml", "configs/thermal_hydraulics.v4.yaml"],
        text=True
    ).strip()
    check("locked_paths_clean", status == "", "empty" if not status else status[:200])
except subprocess.CalledProcessError as e:
    check("locked_paths_clean", False, f"git error: {e}")

# 7. Phase 1 ancestor
try:
    subprocess.check_call(["git", "merge-base", "--is-ancestor", "1d9b320", "HEAD"])
    check("phase1_ancestor", True, "1d9b320 is ancestor")
except subprocess.CalledProcessError:
    check("phase1_ancestor", False, "1d9b320 NOT ancestor")

# 8. Phase 2 lock ancestor
try:
    subprocess.check_call(["git", "merge-base", "--is-ancestor",
                           "0c9dfd7ced7454342e09280b6307f83d42eea89e", "HEAD"])
    check("phase2_lock_ancestor", True, "0c9dfd7 is ancestor")
except subprocess.CalledProcessError:
    check("phase2_lock_ancestor", False, "0c9dfd7 NOT ancestor")

# 9. Dataset directory
check("dataset_dir", Path.home().joinpath(
    "Research/Datasets/SC-FNO/raw/scenarios_2000").is_dir(), "scenarios_2000 exists")

# 10. Scenario count
n_scen = len(list(Path.home().glob("Research/Datasets/SC-FNO/raw/scenarios_2000/scenario_*")))
check("scenario_count", n_scen == 2000, f"{n_scen} scenarios")

result["all_pass"] = all(c["pass"] for c in result["checks"].values())

with open("results/phase_3/preflight/artifact_integrity.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n=== Artifact integrity:", "PASS" if result["all_pass"] else "FAIL", "===")
