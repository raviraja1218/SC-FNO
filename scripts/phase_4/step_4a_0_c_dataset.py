"""Step 4A.0.C — Verify Phase 1 dataset, splits, normalization, and dt."""
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

EXPECTED = {
    "train": "16ad4670eee9555aec237c94d2d241e90b74b50ce8ea57eb07efe209fe852834",
    "val":   "19b781283503b1a1cdb313dc68ce153086f18c25043f1ff5123e3cb3e3205bb6",
    "test":  "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2",
    "norm":  "4e832aa5cdaa940cb2e07bf64c5741cc5124ad40242dd9532da231ff257f6727",
}

out = {"checks": {}, "all_pass": False, "data_stats": {}}
def check(name, ok, detail=""):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

train = pd.read_csv("results/phase_1/splitting/train_ids.csv")
val   = pd.read_csv("results/phase_1/splitting/validation_ids.csv")
test  = pd.read_csv("results/phase_1/splitting/test_ids.csv")

check("train_count", len(train) == 1600, str(len(train)))
check("val_count",   len(val)   == 200,  str(len(val)))
check("test_count",  len(test)  == 200,  str(len(test)))

tr, va, te = set(train["scenario_id"]), set(val["scenario_id"]), set(test["scenario_id"])
check("no_train_val_overlap", len(tr & va) == 0, f"{len(tr & va)} overlap")
check("no_train_test_overlap", len(tr & te) == 0, f"{len(tr & te)} overlap")
check("no_val_test_overlap",  len(va & te) == 0, f"{len(va & te)} overlap")
check("union_2000", len(tr | va | te) == 2000, f"{len(tr | va | te)}")

for name, path in [
    ("train_sha", "results/phase_1/splitting/train_ids.csv"),
    ("val_sha",   "results/phase_1/splitting/validation_ids.csv"),
    ("test_sha",  "results/phase_1/splitting/test_ids.csv"),
    ("norm_sha",  "results/phase_1/splitting/normalization_statistics.json"),
]:
    key = name.replace("_sha", "")
    actual = sha256_file(path)
    check(name, actual == EXPECTED[key], actual[:16])

RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
n_scenarios = len(list(RAW.glob("scenario_*")))
check("scenario_count_2000", n_scenarios == 2000, f"{n_scenarios}")

import random
random.seed(42)
sample = random.sample(sorted(RAW.glob("scenario_*")), 10)
for s in sample[:3]:
    forcing    = np.load(s / "forcing.npy")
    trajectory = np.load(s / "trajectory.npy")
    time       = np.load(s / "time.npy")
    assert forcing.shape == (200,), f"{s.name} forcing {forcing.shape}"
    assert trajectory.shape == (200, 9), f"{s.name} traj {trajectory.shape}"
    assert time.shape == (200,), f"{s.name} time {time.shape}"

dt_vals = []
for s in sample:
    time = np.load(s / "time.npy")
    dt = np.diff(time)
    dt_vals.append(dt.mean())

dt_mean = float(np.mean(dt_vals))
dt_std  = float(np.std(dt_vals))
out["data_stats"]["dt_mean"] = dt_mean
out["data_stats"]["dt_std"] = dt_std
out["data_stats"]["t_start"] = float(np.load(sample[0] / "time.npy")[0])
out["data_stats"]["t_end"]   = float(np.load(sample[0] / "time.npy")[-1])

check("dt_uniform", dt_std < 1e-6, f"mean={dt_mean:.9f}, std={dt_std:.3e}")
check("t_start_zero", abs(out["data_stats"]["t_start"]) < 1e-9,
      str(out["data_stats"]["t_start"]))
check("t_end_10", abs(out["data_stats"]["t_end"] - 10.0) < 1e-6,
      str(out["data_stats"]["t_end"]))
check("dt_matches_050251256", abs(dt_mean - 0.050251256281407) < 1e-6,
      f"{dt_mean:.12f}")

norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))
check("norm_training_only", "training only" in norm["source"].lower(),
      norm["source"][:60])

expected_vars = ["rho_ext","n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
missing = [v for v in expected_vars if v not in norm["global"]]
check("norm_10_vars", len(missing) == 0, f"missing: {missing}")

near_zero = [v for v in expected_vars
             if v in norm["global"] and abs(norm["global"][v]["std"]) < 1e-12]
check("norm_no_near_zero_std", len(near_zero) == 0, f"near-zero: {near_zero}")

out["all_pass"] = all(c["pass"] for c in out["checks"].values())

with open("results/phase_4/phase_4a/preflight/dataset_verification.json", "w") as f:
    json.dump(out, f, indent=2)

with open("results/phase_4/phase_4a/preflight/normalization_verification.json", "w") as f:
    json.dump({
        "source": norm["source"],
        "variables": expected_vars,
        "global_stats": {v: norm["global"][v] for v in expected_vars if v in norm["global"]},
        "all_pass": len(missing) == 0 and len(near_zero) == 0,
    }, f, indent=2)

print(f"\n=== 4A.0.C Dataset verification: {'PASS' if out['all_pass'] else 'FAIL'} ===")
