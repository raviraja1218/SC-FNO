"""Step 5.C — Compare causal vs non-causal boundary ratios."""
import json
from pathlib import Path
import numpy as np

# Non-causal baseline from Phase 5.B seed 42
phase5b = json.load(open("results/phase_5/step_5_B/seed_42/summary.json"))
# Note: Phase 5.B didn't record boundary ratios — we compute them from
# the same training script's boundary_check function if we had it.
# For now, use the numbers from Step 5.B's decoder health check, and
# recompute boundary ratios from seed_42_best.pt in a separate script.

# Causal seed 42
causal_path = Path("results/phase_5/step_5_C/seeds/seed_42_causal_summary.json")
if causal_path.exists():
    causal = json.load(open(causal_path))
    print("=" * 70)
    print("Step 5.C — Causal vs Non-causal comparison")
    print("=" * 70)
    print(f"\nCausal mask (seed 42):")
    print(f"  best_val = {causal['best_val_data_loss']:.6f}")
    print(f"  decoder health: du={causal['post_training_decoder_health']['du_dt_std']:.3f}, "
          f"uvar={causal['post_training_decoder_health']['u_hat_variance']:.3f}")
    b = causal["boundary_ratios"]
    for k in ["n", "Tc", "Tf"]:
        print(f"  B_{k} = {b[k]['start_interior']:.2f}x")

    # Comparison target
    print(f"\nPhase 4A.1 reported: B_n = 4.98x  (target < 4.0)")
    print(f"Phase 2 audit:       B_n = 8.05x")
    print(f"\nVerdict: B_n < 4.0 = {'PASS' if b['n']['start_interior'] < 4.0 else 'FAIL'}")
else:
    print(f"[MISSING] {causal_path}")
