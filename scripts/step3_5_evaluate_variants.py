"""Step 3.5.C — Evaluate P-FNO, PI-FNO, P-PI-FNO on 200 test scenarios."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
from step3_5_eval_utils import (
    load_norm, load_test_ids, load_test_data, denorm_states,
    build_model_for_variant, predict_data_path, compute_scenario_metrics,
    compute_boundary_ratio, get_family, STATES,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
norm = load_norm()
test_ids = load_test_ids()
test_data = load_test_data()
test_ids_set = set(test_ids)


def evaluate_variant(variant):
    print(f"\n{'='*70}\nEvaluating {variant}\n{'='*70}")
    model, padding, _ = build_model_for_variant(variant, device=device)
    all_preds = {}
    per_scenario = {}
    for i, sid in enumerate(test_ids):
        # TEST SET INTEGRITY ASSERTION
        assert sid in test_ids_set, f"Non-test ID {sid} in evaluation"
        d = test_data[sid]
        rho_norm = (d["forcing"] - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        pred_norm = predict_data_path(model, padding, rho_norm[None, :], device)
        pred_phys = denorm_states(torch.tensor(pred_norm), norm).numpy()[0]
        gt_phys = d["trajectory"].T
        all_preds[sid] = pred_phys
        per_scenario[sid] = compute_scenario_metrics(pred_phys, gt_phys)
        per_scenario[sid]["family"] = get_family(sid)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/200 scenarios evaluated")
    return all_preds, per_scenario


# Evaluate each
results = {}
for v in ["p_fno", "pi_fno", "p_pi_fno"]:
    preds, per_sc = evaluate_variant(v)
    results[v] = {"per_scenario": per_sc}
    out = {"predictions": {sid: p.tolist() for sid, p in preds.items()},
           "per_scenario": per_sc}
    Path(f"results/phase_3/evaluation/{v}_predictions.json").write_text(json.dumps(out))
    print(f"  {v}: predictions saved")

# Save consolidated per-scenario table
for v in ["p_fno", "pi_fno", "p_pi_fno"]:
    rows = []
    for sid, m in results[v]["per_scenario"].items():
        row = {"scenario_id": sid, "family": m["family"],
               "aggregate_rel_l2": m["aggregate_rel_l2"]}
        for s in STATES:
            for k in ["mae", "rmse", "rel_l2", "max_abs"]:
                row[f"{s}_{k}"] = m["per_state"][s][k]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"results/phase_3/evaluation/{v}_scenario_metrics.csv", index=False)
    print(f"  {v}: scenario metrics CSV saved ({len(df)} rows)")

print("\n=== 3.5.C PASS — All 3 variants evaluated ===")
