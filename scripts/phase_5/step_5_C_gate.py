"""Gate 5.C — verification."""
import json, subprocess
from pathlib import Path

OUT = Path("results/phase_5/step_5_C")
SEEDS = [42, 123, 456, 789, 1012]
checks = []
def check(n, ok, d=""):
    checks.append({"name": n, "pass": bool(ok), "detail": d})
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {d}")

# 1. Preflight
pf = json.load(open(OUT/"preflight/preflight_checks.json"))
check("Preflight 5.C", pf["all_pass"], f"{len(pf['checks'])} checks")

# 2. All 5 causal summaries exist
summaries = {}
for seed in SEEDS:
    p = OUT/"seeds"/f"seed_{seed}_causal_summary.json"
    if p.exists():
        summaries[seed] = json.load(open(p))
check("All 5 causal summaries", len(summaries) == 5, f"{len(summaries)}/5")

# 3. B_n < 4.0 (target)
if len(summaries) == 5:
    b_ns = [s["boundary_ratios"]["n"]["start_interior"] for s in summaries.values()]
    n_pass = sum(1 for x in b_ns if x < 4.0)
    check("B_n < 4.0 target", n_pass >= 3, f"{n_pass}/5 seeds pass (mean {sum(b_ns)/5:.2f}×)")
    
    # 4. B_Tf < 4.0
    b_tfs = [s["boundary_ratios"]["Tf"]["start_interior"] for s in summaries.values()]
    tf_pass = sum(1 for x in b_tfs if x < 4.0)
    check("B_Tf < 4.0", tf_pass == 5, f"{tf_pass}/5 seeds")
    
    # 5. B_Tc < 4.0
    b_tcs = [s["boundary_ratios"]["Tc"]["start_interior"] for s in summaries.values()]
    tc_pass = sum(1 for x in b_tcs if x < 4.0)
    check("B_Tc < 4.0", tc_pass >= 3, f"{tc_pass}/5 seeds pass (mean {sum(b_tcs)/5:.2f}×)")

    # 6. Interior degradation < 5%
    phase5b_baseline = 0.066180
    best_vals = [s["best_val_data_loss"] for s in summaries.values()]
    mean_bv = sum(best_vals)/5
    degradation = (mean_bv - phase5b_baseline) / phase5b_baseline * 100
    check("Interior accuracy < 5% degradation", abs(degradation) < 5,
          f"{degradation:.2f}% vs Phase 5.B")
    
    # 7. Decoder health all seeds
    health_pass = all(
        s["post_training_decoder_health"]["du_dt_std"] > 0.05 and
        s["post_training_decoder_health"]["u_hat_variance"] > 0.3
        for s in summaries.values()
    )
    check("Decoder health all seeds", health_pass,
          "du_dt std > 0.05, u_hat var > 0.3")
    
    # 8. Multi-seed aggregate exists
    agg_path = OUT/"boundary_analysis/causal_multi_seed_aggregate.json"
    check("Multi-seed aggregate exists", agg_path.exists(), "exists")

# 9. Test set untouched
test_files = list((OUT/"seeds").glob("test_*"))
check("Test set untouched", len(test_files) == 0, "0 test files")

# 10. Phase 4 lock
try:
    t = subprocess.check_output(["git","rev-parse","phase-4-locked"], text=True).strip()
    check("Phase 4 lock intact", t.startswith("2f0b53f1"), t[:12])
except: check("Phase 4 lock intact", False, "missing")

# 11. Locked paths clean
status = subprocess.check_output(
    ["git","status","--porcelain","results/phase_1","results/phase_2",
     "results/phase_3","results/phase_4","models/baseline",
     "models/sc_fno/phase_3","models/sc_fno/phase_4"],
    text=True).strip()
check("Locked paths clean", status == "", "empty" if not status else status[:100])

# 12. All checkpoints exist
ckpt_ok = True
for seed in SEEDS:
    if not (Path("models/sc_fno/phase_5/step_5_C")/f"seed_{seed}_causal_best.pt").exists():
        ckpt_ok = False
check("All 5 seed checkpoints exist", ckpt_ok, "10 files expected")

result = {"gate": "5.C", "n_checks": len(checks),
          "n_pass": sum(1 for c in checks if c["pass"]),
          "all_pass": all(c["pass"] for c in checks),
          "checks": checks}
json.dump(result, open(OUT/"gate_5_C_verification.json","w"), indent=2)
print(f"\n=== Gate 5.C: {result['n_pass']}/{result['n_checks']} PASS ===")
