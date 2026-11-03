"""Step 5.B preflight — verify Phase 0-5.A locks intact, write checks JSON."""
import json, hashlib, subprocess
from pathlib import Path

OUT = Path("results/phase_5/step_5_B")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "preflight").mkdir(parents=True, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def anc(c):
    try:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", c, "HEAD"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

checks = []
def check(n, ok, d=""):
    checks.append({"name": n, "pass": bool(ok), "detail": d})
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}")

# Ancestry through Phase 5.A
for c, l in [("1d9b320","p1"), ("0c9dfd7","p2"), ("bc80727","p3"),
             ("dea218a","p4a"), ("dd392bb","p4b"), ("3b622f7","p4"),
             ("ec9029e","5A")]:
    check(l+"_ancestor", anc(c), c)

# Phase 4 tag
try:
    t = subprocess.check_output(["git","rev-parse","phase-4-locked"], text=True).strip()
    check("phase4_tag", t.startswith("2f0b53f1"), t[:12])
except:
    check("phase4_tag", False, "missing")

# Checkpoint SHAs
try:
    check("phase3_ckpt", sha("models/sc_fno/phase_3/pi_fno/best_val.pt")
          .startswith("a0522e57"), "a0522e57...")
    check("test_ids", sha("results/phase_1/splitting/test_ids.csv")
          .startswith("a71514ea"), "a71514ea...")
except Exception as e:
    check("checkpoints", False, str(e))

# Locked paths
try:
    status = subprocess.check_output(
        ["git","status","--porcelain","results/phase_1","results/phase_2",
         "results/phase_3","results/phase_4","models/baseline",
         "models/sc_fno/phase_3","models/sc_fno/phase_4"],
        text=True).strip()
    check("locked_clean", status == "", "empty" if not status else status[:100])
except Exception as e:
    check("locked_clean", False, str(e))

# Step 5.A artifacts
check("5A_gate", Path("results/phase_5/step_5_A/gate_5_A_verification.json").exists(), "exists")
check("SCFNOModelV2", Path("src/phase_5_corrected/model_v2.py").exists(), "exists")
check("SCFNOModelV2_decoder_v2", Path("src/phase_5_corrected/decoder_v2.py").exists(), "exists")

all_pass = all(c["pass"] for c in checks)
out = {"checks": checks, "all_pass": all_pass}
json.dump(out, open(OUT/"preflight"/"preflight_checks.json","w"), indent=2)
print(f"\n=== Preflight: {'PASS' if all_pass else 'FAIL'} ===")
print(f"Written: {OUT}/preflight/preflight_checks.json")
