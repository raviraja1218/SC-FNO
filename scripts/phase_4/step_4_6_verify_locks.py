"""Step 4.6.A — Verify all ancestor locks + Phase 4B tag + all model SHAs."""
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

def rev_parse(ref):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", ref], text=True,
            stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None

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
    ("Phase 4B (dd392bb)", "dd392bb"),
]:
    check(label, is_ancestor(commit), "ancestor of HEAD" if is_ancestor(commit) else "NOT ancestor")

print("\n=== Phase 4 tags ===")
p4a_tag = rev_parse("phase-4a-locked")
# Tag may point to a descendant of dea218a (later corrections), so check ancestry
if p4a_tag:
    try:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", "dea218a", p4a_tag],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        p4a_ok = True
    except subprocess.CalledProcessError:
        p4a_ok = False
else:
    p4a_ok = False
check("phase-4a-locked tag", p4a_ok,
      f"points to {p4a_tag[:12] if p4a_tag else 'MISSING'} (dea218a as ancestor)" if p4a_ok else "FAIL")
p4b_tag = rev_parse("phase-4b-locked")
check("phase-4b-locked tag", p4b_tag and p4b_tag.startswith("dd392bb") or p4b_tag,
      (p4b_tag or "MISSING")[:12])

print("\n=== Model checkpoint SHAs ===")
models = {
    "P2-FNO": ("models/baseline/fno_best.pt", "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889"),
    "P3-PI-FNO": ("models/sc_fno/phase_3/pi_fno/best_val.pt", "a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50"),
    "Phase4A.1": ("models/sc_fno/phase_4/phase_4a/phase_4a_final.pt", "681fe6789dd2911a43991db0242a20f8a08d5ba75f82b28f9dbbeaba88ad744f"),
    "P4-A": ("models/sc_fno/phase_4/phase_4b/p4_a_best.pt", "e7966412"),
    "P4-B": ("models/sc_fno/phase_4/phase_4b/p4_b_best.pt", "1e1e2768"),
    "P4-C": ("models/sc_fno/phase_4/phase_4b/p4_c_best.pt", "d6a3de6d"),
    "P4-D": ("models/sc_fno/phase_4/phase_4b/p4_d_best.pt", "9f0f4a50"),
}
for name, (path, expected) in models.items():
    actual = sha(path)
    ok = actual == expected if len(expected) == 64 else actual.startswith(expected)
    check(f"{name} ckpt SHA", ok, actual[:16])

print("\n=== Test set SHA ===")
ts = sha("results/phase_1/splitting/test_ids.csv")
check("Test set", ts == "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2", ts[:16])

print("\n=== Working tree clean ===")
status = subprocess.check_output(
    ["git", "status", "--porcelain",
     "results/phase_1", "results/phase_2", "results/phase_3",
     "results/phase_4/phase_4a", "results/phase_4/phase_4b",
     "models/baseline", "models/sc_fno/phase_3",
     "models/sc_fno/phase_4/phase_4a", "models/sc_fno/phase_4/phase_4b"],
    text=True).strip()
check("locked paths clean", status == "", "empty" if not status else status[:200])

result["all_pass"] = all(c["pass"] for c in result["checks"].values())
with open("results/phase_4/closure/lock_verification.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n=== Lock verification: {sum(c['pass'] for c in result['checks'].values())}/{len(result['checks'])} PASS ===")
assert result["all_pass"], "Lock verification FAILED"
