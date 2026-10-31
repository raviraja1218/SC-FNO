"""Step 3.0.G — Verify Phase 1/Phase 2 I/O conventions unchanged."""
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path

def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

out = {"checks": {}, "all_pass": False}
def check(name, ok, detail=""):
    out["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# 1. State ordering (9-state)
STATE_ORDER = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
check("state_ordering_length", len(STATE_ORDER) == 9, f"{len(STATE_ORDER)} states")

# 2. Scenario I/O shapes
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
sid = "scenario_0001"
forcing    = np.load(RAW / sid / "forcing.npy")
trajectory = np.load(RAW / sid / "trajectory.npy")
time       = np.load(RAW / sid / "time.npy")

check("forcing_shape",  forcing.shape == (200,),    str(forcing.shape))
check("trajectory_shape", trajectory.shape == (200, 9), str(trajectory.shape))
check("time_shape",     time.shape == (200,),       str(time.shape))
check("time_starts_at_0",  abs(float(time[0])) < 1e-6, f"{time[0]}")
check("time_ends_at_10",   abs(float(time[-1]) - 10.0) < 1e-3, f"{time[-1]}")
check("time_uniform_dt",   bool(np.allclose(np.diff(time), 0.05, atol=1e-6)),
      f"dt = {np.diff(time)[0]:.6f}")

# 3. Splits
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
check("split_union_2000",     len(tr | va | te) == 2000, f"{len(tr | va | te)} total")

# 4. Normalization
norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))
check("normalization_training_only", "training only" in norm["source"].lower(),
      norm["source"][:60])
expected_vars = ["rho_ext","n","C1","C2","C3","C4","C5","C6","Tf","Tc"]
missing_vars = [v for v in expected_vars if v not in norm["global"]]
check("normalization_10_vars", len(missing_vars) == 0, f"missing: {missing_vars}")

# 5. Splits SHA
for name, path, expected in [
    ("train_sha", "results/phase_1/splitting/train_ids.csv",
     "16ad4670eee9555aec237c94d2d241e90b74b50ce8ea57eb07efe209fe852834"),
    ("val_sha",   "results/phase_1/splitting/validation_ids.csv",
     "19b781283503b1a1cdb313dc68ce153086f18c25043f1ff5123e3cb3e3205bb6"),
    ("test_sha",  "results/phase_1/splitting/test_ids.csv",
     "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2"),
    ("norm_sha",  "results/phase_1/splitting/normalization_statistics.json",
     "4e832aa5cdaa940cb2e07bf64c5741cc5124ad40242dd9532da231ff257f6727"),
]:
    actual = sha256_file(path)
    check(name, actual == expected, actual[:16])

out["all_pass"] = all(c["pass"] for c in out["checks"].values())
with open("results/phase_3/preflight/io_verification.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n=== I/O verification:", "PASS" if out["all_pass"] else "FAIL", "===")
