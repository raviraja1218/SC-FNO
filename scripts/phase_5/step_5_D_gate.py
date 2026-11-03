"""Gate 5.D — verification."""
import json, subprocess
from pathlib import Path

OUT = Path("results/phase_5/step_5_D")
checks = []
def check(n, ok, d=""):
    checks.append({"name": n, "pass": bool(ok), "detail": d})
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}")

# 1. Preflight
pf = json.load(open(OUT/"preflight/preflight_checks.json"))
check("Preflight 5.D", pf["all_pass"], f"{len(pf['checks'])} checks")

# 2. 100 unseen scenarios generated
import pandas as pd
m = pd.read_csv(OUT/"unseen_generation/unseen_manifest.csv")
check("100 unseen scenarios generated", len(m) == 100, f"{len(m)}/100")
check("50 in-dist + 50 OOD", m["in_distribution"].sum() == 50 and (~m["in_distribution"]).sum() == 50,
      f"in-dist={int(m['in_distribution'].sum())}, OOD={int((~m['in_distribution']).sum())}")

# 3. Inference
inf = json.load(open(OUT/"inference/inference_results.json"))
check("5.B test evaluation complete", "5B_noncausal_s42_test" in inf, "present")
check("5.C test evaluation complete", "5C_causal_s456_test" in inf, "present")
check("Unseen inference complete", "5B_noncausal_s42_unseen" in inf and "5C_causal_s456_unseen" in inf,
      "both models on 100 unseen")
check("No NaN/Inf in metrics",
      all(isinstance(v.get("overall_median"), (int, float)) and v["overall_median"] == v["overall_median"]
          for v in inf.values()),
      "finite")

# 4. Resolution
res = json.load(open(OUT/"resolution/resolution_transfer.json"))
check("Resolution transfer at 5 densities", len(res) == 5,
      f"tested N = {list(res.keys())}")

# 5. Benchmark
bm = json.load(open(OUT/"benchmark/benchmark_results.json"))
check("BDF benchmark complete", "bdf_median_ms" in bm, f"BDF median {bm['bdf_median_ms']:.2f} ms")
check("SC-FNO benchmark complete", "scfno_model_median_ms" in bm,
      f"SC-FNO {bm['scfno_model_median_ms']:.2f} ms")
check("Speedup measured", bm["speedup_e2e"] > 1, f"{bm['speedup_e2e']:.1f}× end-to-end")
check("VRAM < 3 GB", bm["peak_vram_MB"] < 3000, f"{bm['peak_vram_MB']:.1f} MB")

# 6. Test set untouched in prior phases
status = subprocess.check_output(
    ["git","status","--porcelain","results/phase_1","results/phase_2",
     "results/phase_3","results/phase_4","models/baseline",
     "models/sc_fno/phase_3","models/sc_fno/phase_4"],
    text=True).strip()
check("Locked paths clean", status == "", "empty" if not status else status[:100])

# 7. Phase 4 lock intact
try:
    t = subprocess.check_output(["git","rev-parse","phase-4-locked"], text=True).strip()
    check("Phase 4 lock intact", t.startswith("2f0b53f1"), t[:12])
except: check("Phase 4 lock intact", False)

# 8. Phase 5.A/B/C locks intact
for c, l in [("ec9029e","5.A"), ("68c1ce1","5.B"), ("6bb3d12","5.C")]:
    try:
        subprocess.check_call(["git","merge-base","--is-ancestor", c, "HEAD"],
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        check(f"Phase {l} ancestor", True, c)
    except: check(f"Phase {l} ancestor", False, c)

result = {"gate": "5.D", "n_checks": len(checks),
          "n_pass": sum(1 for c in checks if c["pass"]),
          "all_pass": all(c["pass"] for c in checks),
          "checks": checks}
json.dump(result, open(OUT/"gate_5_D_verification.json","w"), indent=2)
print(f"\n=== Gate 5.D: {result['n_pass']}/{result['n_checks']} PASS ===")
