"""Step 3.3.A — Merge residual scales (Step 3.1) + normalization (Phase 1)."""
import json
from pathlib import Path

# Load from Step 3.1
res_scales = json.load(open("results/phase_3/residuals/residual_scales.json"))
assert "per_state" in res_scales
res_per_state = res_scales["per_state"]

# Load from Phase 1
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))
assert "training only" in norm["source"].lower(), "Normalization not training-only"
norm_global = norm["global"]

STATES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
RESIDUAL_UNITS = {"n": "1/s", "C1": "1/s", "C2": "1/s", "C3": "1/s",
                  "C4": "1/s", "C5": "1/s", "C6": "1/s", "Tf": "K/s", "Tc": "K/s"}

merged = {}
for state in STATES:
    if state not in res_per_state:
        raise ValueError(f"State {state} missing from residual_scales.json")
    if state not in norm_global:
        raise ValueError(f"State {state} missing from normalization_statistics.json")
    merged[state] = {
        "residual_p95": float(res_per_state[state]["p95"]),
        "residual_mean": float(res_per_state[state]["mean"]),
        "residual_max": float(res_per_state[state]["max"]),
        "residual_unit": RESIDUAL_UNITS[state],
        "normalization_mean": float(norm_global[state]["mean"]),
        "normalization_std": float(norm_global[state]["std"]),
    }
    # Sanity: scale must be positive
    assert merged[state]["residual_p95"] > 0, f"State {state} p95 <= 0"
    assert merged[state]["normalization_std"] > 0, f"State {state} std <= 0"

out = {
    "source_residual_scales": "results/phase_3/residuals/residual_scales.json (Step 3.1)",
    "source_normalization": "results/phase_1/splitting/normalization_statistics.json (Phase 1 training-only)",
    "residual_scale_definition": "P95(|R_j|) — 95th percentile of absolute residual magnitude",
    "per_state": merged,
    "all_pass": True,
}
Path("results/phase_3/loss_integration/per_state_scaling.json").write_text(
    json.dumps(out, indent=2)
)
print(f"Merged scaling table: {len(merged)} states")
for s in STATES:
    print(f"  {s}: residual_p95={merged[s]['residual_p95']:.4e} {merged[s]['residual_unit']}, "
          f"norm_std={merged[s]['normalization_std']:.4e}")
print("\n=== 3.3.A PASS ===")
