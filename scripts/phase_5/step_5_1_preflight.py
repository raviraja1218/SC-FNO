"""Step 5.1 preflight — verify Phase 0-4 locks intact before retraining."""
import json
import hashlib
import subprocess
import platform
from pathlib import Path

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def is_ancestor(commit):
    try:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

EXPECTED = {
    "phase1_agg": "9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6",
    "phase1_manifest": "993581ef514932c5f3c341ce176a401a5247ebe7217e83f70d3ebe26dd3cff12",
    "phase2_ckpt": "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889",
    "phase3_ckpt": "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
    "phase4a1_ckpt": "681fe6789dd2911a43991db0242a20f8a08d5ba75f82b28f9dbbeaba88ad744f",
    "train_ids": "16ad4670eee9555aec237c94d2d241e90b74b50ce8ea57eb07efe209fe852834",
    "val_ids": "19b781283503b1a1cdb313dc68ce153086f18c25043f1ff5123e3cb3e3205bb6",
    "test_ids": "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2",
    "normalization": "4e832aa5cdaa940cb2e07bf64c5741cc5124ad40242dd9532da231ff257f6727",
    "phase4_lock": "3b622f7",
}

out = {"checks": {}, "all_pass": False}

def check(name, ok, detail=""):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# --- Ancestry ---
for commit, label in [
    ("1d9b320", "phase1_ancestor"),
    ("0c9dfd7", "phase2_ancestor"),
    ("bc80727", "phase3_ancestor"),
    ("dea218a", "phase4a_ancestor"),
    ("dd392bb", "phase4b_ancestor"),
    ("3b622f7", "phase4_ancestor"),
]:
    check(label, is_ancestor(commit), f"{commit} is ancestor")

# --- Tag ---
try:
    tag = subprocess.check_output(["git", "rev-parse", "phase-4-locked"], text=True).strip()
    check("phase4_tag", tag.startswith("2f0b53f1"), tag[:12])
except subprocess.CalledProcessError:
    check("phase4_tag", False, "tag missing")

# --- Phase 1 manifest ---
m = json.load(open("results/phase_1/phase_1_dataset_manifest.json"))
check("phase1_aggregate", m["dataset"]["aggregate_sha256"] == EXPECTED["phase1_agg"],
      m["dataset"]["aggregate_sha256"][:16])
check("phase1_configs",
      m["config_versions"] == {"reactor_parameters": "v3", "thermal_hydraulics": "v4"},
      str(m["config_versions"]))

actual = sha256("results/phase_1/phase_1_dataset_manifest.json")
check("phase1_manifest_sha", actual == EXPECTED["phase1_manifest"], actual[:16])

# --- Checkpoints ---
for name, path in [
    ("phase2_ckpt", "models/baseline/fno_best.pt"),
    ("phase3_ckpt", "models/sc_fno/phase_3/pi_fno/best_val.pt"),
    ("phase4a1_ckpt", "models/sc_fno/phase_4/phase_4a/phase_4a_final.pt"),
]:
    actual = sha256(path)
    check(name + "_sha", actual == EXPECTED[name], actual[:16])

# --- Splits ---
for name, path in [
    ("train_ids", "results/phase_1/splitting/train_ids.csv"),
    ("val_ids", "results/phase_1/splitting/validation_ids.csv"),
    ("test_ids", "results/phase_1/splitting/test_ids.csv"),
    ("normalization", "results/phase_1/splitting/normalization_statistics.json"),
]:
    actual = sha256(path)
    check(name + "_sha", actual == EXPECTED[name], actual[:16])

# --- Locked paths clean ---
status = subprocess.check_output(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "results/phase_3", "results/phase_4",
     "models/baseline", "models/sc_fno/phase_3", "models/sc_fno/phase_4",
     "configs/reactor_parameters.v3.yaml", "configs/thermal_hydraulics.v4.yaml"],
    text=True).strip()
check("locked_paths_clean", status == "", "empty" if not status else status[:200])

# --- Dataset dir ---
check("dataset_dir", Path.home().joinpath("Research/Datasets/SC-FNO/raw/scenarios_2000").is_dir(),
      "exists")

# --- Environment ---
env = {
    "python": platform.python_version(),
    "platform": platform.platform(),
}
try:
    import torch, neuralop
    env["torch"] = torch.__version__
    env["neuralop"] = getattr(neuralop, "__version__", "unknown")
    env["cuda_available"] = torch.cuda.is_available()
    env["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    env["cuda_vram_MB"] = torch.cuda.get_device_properties(0).total_memory / 1e6 if torch.cuda.is_available() else 0
    check("cuda_available", torch.cuda.is_available(), env.get("cuda_device"))
except Exception as e:
    check("cuda_available", False, f"import failed: {e}")

out["all_pass"] = all(c["pass"] for c in out["checks"].values())

import os
os.makedirs("results/phase_5/step_5_1/preflight", exist_ok=True)
json.dump(out, open("results/phase_5/step_5_1/preflight/phase_4_lock_verification.json", "w"), indent=2)
json.dump(env, open("results/phase_5/step_5_1/preflight/environment.json", "w"), indent=2)

print(f"\n=== Preflight: {'PASS' if out['all_pass'] else 'FAIL'} ===")
