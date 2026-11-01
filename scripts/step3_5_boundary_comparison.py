"""Missing Item 1 — Boundary artifact comparison on test set.

Reproduces Phase 2's methodology: for each state, compute
    boundary_ratio = start_MAE / interior_MAE
on the full 200-scenario test set for all 4 models.

Phase 2 measured: n = 8.05×, Tc = 8.66× (start/interior).
Phase 3 will report the same for P2, P-FNO, PI-FNO, P-PI-FNO.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")
from step3_5_eval_utils import load_test_data, STATES, mae

base = Path("results/phase_3/evaluation")
test_data = load_test_data()

# Window for "start" boundary — match Phase 2 methodology (first 10 points = 0.5 s)
START_WINDOW = 10  # 10 timesteps × 0.050251s ≈ 0.50 s
# Interior: skip first and last START_WINDOW points
INTERIOR_START = START_WINDOW
INTERIOR_END = -START_WINDOW

def compute_boundary_ratios(predictions_dict):
    """For each state, compute start_MAE / interior_MAE, averaged over scenarios."""
    per_state = {}
    for state in STATES:
        per_state[state] = {
            "start_maes": [],
            "interior_maes": [],
        }

    for sid, pred in predictions_dict.items():
        pred_arr = np.array(pred)  # [9, 200] physical units
        gt = test_data[sid]["trajectory"].T  # [9, 200] physical units

        for j, state in enumerate(STATES):
            p = pred_arr[j]
            g = gt[j]
            start_mae = float(np.mean(np.abs(p[:START_WINDOW] - g[:START_WINDOW])))
            interior_mae = float(np.mean(np.abs(p[INTERIOR_START:INTERIOR_END] - g[INTERIOR_START:INTERIOR_END])))
            per_state[state]["start_maes"].append(start_mae)
            per_state[state]["interior_maes"].append(interior_mae)

    # Compute mean and ratio
    result = {}
    for state in STATES:
        sm = float(np.mean(per_state[state]["start_maes"]))
        im = float(np.mean(per_state[state]["interior_maes"]))
        ratio = sm / (im + 1e-12)
        result[state] = {
            "start_mae": sm,
            "interior_mae": im,
            "boundary_ratio": ratio,
        }
    return result

# Load all 4 predictions
models = {}
for v, name in [("p2","P2-FNO"), ("p_fno","P-FNO"), ("pi_fno","PI-FNO"), ("p_pi_fno","P-PI-FNO")]:
    if v == "p2":
        preds = json.load(open(base / "p2_fno_predictions.json"))["predictions"]
    else:
        preds = json.load(open(base / f"{v}_predictions.json"))["predictions"]
    models[name] = compute_boundary_ratios(preds)
    print(f"\n=== {name} ===")
    for state in STATES:
        r = models[name][state]["boundary_ratio"]
        print(f"  {state:>4}: start_MAE = {models[name][state]['start_mae']:.4e}, "
              f"interior_MAE = {models[name][state]['interior_mae']:.4e}, ratio = {r:.3f}x")

# Save
out = {
    "start_window_timesteps": START_WINDOW,
    "start_window_seconds": START_WINDOW * 0.050251256,
    "methodology": (
        "Boundary ratio = MAE(first 10 timesteps) / MAE(interior 180 timesteps), "
        "averaged over 200 test scenarios. Matches Phase 2 supplemental audit "
        "methodology (Phase 2 reported n=8.05x, Tc=8.66x on the same metric)."
    ),
    "phase_2_reference": {
        "n": 8.05,
        "Tc": 8.66,
        "note": "From Phase 2 supplemental audit Item 2 (start/interior ratio on 200 test scenarios)"
    },
    "models": models,
    "honest_interpretation": {
        "if_ratio_decreased": "Padding or physics may have reduced the boundary artifact",
        "if_ratio_unchanged": "Phase 3 did not address the Phase 2 boundary artifact",
        "phase_3_actual": "See models above — report actual measured values without claiming improvement unless they exist"
    }
}
(base / "boundary_comparison.json").write_text(json.dumps(out, indent=2))
print("\n\n=== Saved boundary_comparison.json ===")

# Print summary comparison table
print("\n" + "="*90)
print("BOUNDARY RATIO COMPARISON (start MAE / interior MAE)")
print("="*90)
print(f"{'State':>6} {'Phase 2':>12} {'P2-FNO':>12} {'P-FNO':>12} {'PI-FNO':>12} {'P-PI-FNO':>12}")
print("-"*90)
for state in STATES:
    p2_ref = out["phase_2_reference"].get(state, "—")
    p2_ref_str = f"{p2_ref:.2f}x" if isinstance(p2_ref, (int, float)) else p2_ref
    vals = []
    for name in ["P2-FNO", "P-FNO", "PI-FNO", "P-PI-FNO"]:
        vals.append(f"{models[name][state]['boundary_ratio']:.2f}x")
    print(f"{state:>6} {p2_ref_str:>12} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

print("\nHonest verdict: If PI-FNO boundary ratio for n is << 8.05x, physics/padding helped.")
print("If ~8.05x, boundary artifact unchanged from Phase 2.")
