"""Step 4A.0.A — Verify Phase 1/2/3 locked artifacts intact.

Handles annotated tags by peeling to commit SHA (^{commit}).
"""
import json
import hashlib
import subprocess
from pathlib import Path

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def git_rev_parse(ref):
    try:
        return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    except subprocess.CalledProcessError:
        return None

def git_peel_commit(ref):
    """Peel ref to commit SHA (works for annotated tags, lightweight tags, branches, commits)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"{ref}^{{commit}}"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None

def is_ancestor(commit):
    try:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

EXPECTED = {
    "phase1_aggregate": "9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6",
    "phase1_manifest":  "993581ef514932c5f3c341ce176a401a5247ebe7217e83f70d3ebe26dd3cff12",
    "phase2_ckpt":      "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889",
    "phase2_lock":      "0c9dfd7ced7454342e09280b6307f83d42eea89e",
    "phase3_pi_fno":    "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50",
    "phase3_lock":      "bc80727",
    "decoder_init":     "c30dd6ca0cc7fb87918696265285eb041f8c214264c1ab6622659fd6623c4b3f",
    "train_ids":        "16ad4670eee9555aec237c94d2d241e90b74b50ce8ea57eb07efe209fe852834",
    "val_ids":          "19b781283503b1a1cdb313dc68ce153086f18c25043f1ff5123e3cb3e3205bb6",
    "test_ids":         "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2",
    "normalization":    "4e832aa5cdaa940cb2e07bf64c5741cc5124ad40242dd9532da231ff257f6727",
}

result = {"checks": {}, "all_pass": False}

def check(name, ok, detail=""):
    result["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# 1. Tag exists (any form)
tag_object_sha = git_rev_parse("phase-3-closed")
check("phase3_tag_exists", tag_object_sha is not None,
      (tag_object_sha or "MISSING")[:12])

# 2. Tag PEELED to commit matches bc80727
#    Annotated tags: rev-parse returns tag object; peel with ^{commit}
tag_commit_sha = git_peel_commit("phase-3-closed")
tag_peeled_ok = tag_commit_sha is not None and tag_commit_sha.startswith(EXPECTED["phase3_lock"])
check("phase3_tag_peels_to_bc80727", tag_peeled_ok,
      f"peeled={tag_commit_sha[:12] if tag_commit_sha else 'MISSING'} "
      f"(expected {EXPECTED['phase3_lock']})")

# 3. Tag annotation object (documented for audit)
tag_ann_sha = git_rev_parse(f"refs/tags/phase-3-closed")
result["checks"]["_tag_object_sha"] = {
    "pass": True,
    "detail": tag_ann_sha[:12] if tag_ann_sha else "not a ref",
}
result["checks"]["_tag_is_annotated"] = {
    "pass": tag_object_sha != tag_commit_sha,
    "detail": "annotated" if tag_object_sha != tag_commit_sha else "lightweight",
}

# 4. Phase 3 ancestor of HEAD
check("phase3_ancestor_of_head", is_ancestor(EXPECTED["phase3_lock"]),
      "bc80727 is ancestor" if is_ancestor(EXPECTED["phase3_lock"]) else "NOT ancestor")

# 5. Phase 2 lock ancestor
check("phase2_ancestor_of_head", is_ancestor(EXPECTED["phase2_lock"]),
      "0c9dfd7 is ancestor" if is_ancestor(EXPECTED["phase2_lock"]) else "NOT ancestor")

# 6. Phase 1 aggregate
m = json.load(open("results/phase_1/phase_1_dataset_manifest.json"))
check("phase1_aggregate",
      m["dataset"]["aggregate_sha256"] == EXPECTED["phase1_aggregate"],
      m["dataset"]["aggregate_sha256"][:16] + "...")
check("phase1_configs",
      m["config_versions"] == {"reactor_parameters": "v3", "thermal_hydraulics": "v4"},
      str(m["config_versions"]))

# 7. Phase 1 manifest sha
actual = sha256_file("results/phase_1/phase_1_dataset_manifest.json")
check("phase1_manifest_sha", actual == EXPECTED["phase1_manifest"], actual[:16])

# 8. Phase 2 checkpoint
actual = sha256_file("models/baseline/fno_best.pt")
check("phase2_ckpt_sha", actual == EXPECTED["phase2_ckpt"], actual[:16])

# 9. Phase 3 PI-FNO checkpoint
actual = sha256_file("models/sc_fno/phase_3/pi_fno/best_val.pt")
check("phase3_pi_fno_sha", actual == EXPECTED["phase3_pi_fno"], actual[:16])

# 10. Decoder init
actual = sha256_file("models/sc_fno/phase_3/decoder_init.pt")
check("decoder_init_sha", actual == EXPECTED["decoder_init"], actual[:16])

# 11. Split files
for name, path in [
    ("train_ids",     "results/phase_1/splitting/train_ids.csv"),
    ("val_ids",       "results/phase_1/splitting/validation_ids.csv"),
    ("test_ids",      "results/phase_1/splitting/test_ids.csv"),
    ("normalization", "results/phase_1/splitting/normalization_statistics.json"),
]:
    actual = sha256_file(path)
    check(name, actual == EXPECTED[name], actual[:16])

# 12. Locked paths clean
status = subprocess.check_output(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "results/phase_3",
     "models/baseline",
     "configs/reactor_parameters.v3.yaml", "configs/thermal_hydraulics.v4.yaml"],
    text=True
).strip()
check("locked_paths_clean", status == "", "empty" if not status else status[:200])

# 13. Dataset directory
check("dataset_dir", Path.home().joinpath(
    "Research/Datasets/SC-FNO/raw/scenarios_2000").is_dir(), "exists")

# Ignore the informational keys starting with "_" when determining overall pass
real_checks = {k: v for k, v in result["checks"].items() if not k.startswith("_")}
result["all_pass"] = all(c["pass"] for c in real_checks.values())

with open("results/phase_4/phase_4a/preflight/phase_3_lock_verification.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n=== 4A.0.A Phase 3 lock integrity:",
      "PASS" if result["all_pass"] else "FAIL", "===")
print(f"  Tag object: {tag_object_sha[:12] if tag_object_sha else 'MISSING'}")
print(f"  Tag commit (peeled): {tag_commit_sha[:12] if tag_commit_sha else 'MISSING'}")
print(f"  Expected commit:     {EXPECTED['phase3_lock']}")
