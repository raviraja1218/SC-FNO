"""Step 4B.6.A — Verify all ancestor locks + model SHAs + working tree state."""
import json
import hashlib
import subprocess
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def is_ancestor(commit):
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

result = {"checks": {}, "all_pass": False}
def check(name, ok, detail=""):
    result["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

print("=== Ancestor locks ===")
for label, commit in [
    ("Phase 1 (1d9b320)", "1d9b320"),
    ("Phase 2 (0c9dfd7)", "0c9dfd7ced7454342e09280b6307f83d42eea89e"),
    ("Phase 3 (bc80727)", "bc80727"),
    ("Phase 4A (dea218a)", "dea218a"),
]:
    check(label, is_ancestor(commit), "ancestor of HEAD" if is_ancestor(commit) else "NOT ancestor")

print("\n=== Phase 4B.0–4B.5 commit chain present ===")
commits = ["bec0f63", "b8ae156", "3fd0af0", "b412f20", "10a7e6d", "9d0e0da",
           "836ddf7", "3dd0ed4", "7e456aa", "985c0a9", "d226447",
           "b5f94a0", "e96d94f", "109cdfc", "75dce8f"]
for c in commits:
    ok = is_ancestor(c)
    check(f"commit {c}", ok, "present" if ok else "MISSING")

print("\n=== Model checkpoint SHAs ===")
models = {
    "P2-FNO": ("models/baseline/fno_best.pt", "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889"),
    "P3-PI-FNO": ("models/sc_fno/phase_3/pi_fno/best_val.pt", "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50"),
    "Phase4A.1": ("models/sc_fno/phase_4/phase_4a/phase_4a_final.pt", "681fe6789dd2911a43991db0242a20f8a08d5ba75f82b28f9dbbeaba88ad744f"),
}
for name, (path, expected) in models.items():
    actual = sha(path)
    check(f"{name} ckpt SHA", actual == expected, actual[:16])

print("\n=== Phase 4B variant ckpts (prefix check) ===")
p4_models = {
    "P4-A": ("models/sc_fno/phase_4/phase_4b/p4_a_best.pt", "e7966412"),
    "P4-B": ("models/sc_fno/phase_4/phase_4b/p4_b_best.pt", "1e1e2768"),
    "P4-C": ("models/sc_fno/phase_4/phase_4b/p4_c_best.pt", "d6a3de6d"),
    "P4-D": ("models/sc_fno/phase_4/phase_4b/p4_d_best.pt", "9f0f4a50"),
}
for name, (path, prefix) in p4_models.items():
    actual = sha(path)
    check(f"{name} ckpt SHA", actual.startswith(prefix), actual[:16])

print("\n=== Test set SHA ===")
ts = sha("results/phase_1/splitting/test_ids.csv")
check("Test set", ts == "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2", ts[:16])

print("\n=== Working tree clean for locked paths ===")
status = subprocess.check_output(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "results/phase_3",
     "results/phase_4/phase_4a", "models/baseline",
     "models/sc_fno/phase_3", "models/sc_fno/phase_4/phase_4a"],
    text=True
).strip()
check("locked paths clean", status == "", "empty" if not status else status[:200])

result["all_pass"] = all(c["pass"] for c in result["checks"].values())
with open("results/phase_4/phase_4b/closure/lock_verification.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n=== Lock verification: {sum(c['pass'] for c in result['checks'].values())}/{len(result['checks'])} PASS ===")
assert result["all_pass"], "Lock verification FAILED"
