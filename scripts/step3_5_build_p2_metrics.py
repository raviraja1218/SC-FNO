"""Build p2_scenario_metrics.csv from p2_fno_predictions.json, matching the
exact metric computation used for the SC-FNO variants in 3.5.C."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from step3_5_eval_utils import (
    load_test_data, compute_scenario_metrics, get_family, STATES,
)

base = Path("results/phase_3/evaluation")
test_data = load_test_data()

# Load p2 predictions
p2_preds = json.load(open(base / "p2_fno_predictions.json"))
preds = p2_preds["predictions"]

rows = []
for sid in sorted(preds.keys()):
    pred = np.array(preds[sid])          # [9, 200] physical units
    gt = test_data[sid]["trajectory"].T  # [9, 200] physical units
    m = compute_scenario_metrics(pred, gt)
    row = {
        "scenario_id": sid,
        "family": get_family(sid),
        "aggregate_rel_l2": m["aggregate_rel_l2"],
    }
    for s in STATES:
        for k in ["mae", "rmse", "rel_l2", "max_abs"]:
            row[f"{s}_{k}"] = m["per_state"][s][k]
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(base / "p2_scenario_metrics.csv", index=False)
print(f"Saved p2_scenario_metrics.csv: {len(df)} rows")
print(df.head(3).to_string())
