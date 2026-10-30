"""Step 1.4 Block B — deterministic stratified scenario-level split.

Strategy (per approved plan):
  - For each family, sort its 500 scenarios by the canonical parameter
    (amplitude for step/ramp, omega for sinusoidal, length_scale for GRF)
  - Assign index 0, 10, 20, ..., 490 to TEST (50 scenarios)
  - Assign index 5, 15, 25, ..., 495 to VALIDATION (50 scenarios)
  - Remaining 400 to TRAIN
  - Result: 400/50/50 per family → 1600/200/200 total

Split seed: split_v1_2026 (recorded in manifest, deterministic)
"""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data.config_loader import load_config
from data.forcing_profiles import build_production_registry
from physics.kinetics import KineticsParameters

R = Path('results/phase_1/splitting')
R.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = "split_v1_2026"
PER_FAMILY = 500
PER_FAMILY_VAL = 50
PER_FAMILY_TEST = 50
PER_FAMILY_TRAIN = PER_FAMILY - PER_FAMILY_VAL - PER_FAMILY_TEST  # = 400


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_key(spec):
    """Return a sortable canonical parameter per family."""
    p = spec.params
    if spec.family == "step":
        return p["amplitude"]
    elif spec.family == "ramp":
        return p["amplitude"]
    elif spec.family == "sinusoidal":
        return p["omega"]
    elif spec.family == "grf":
        return p["length_scale"]
    else:
        raise ValueError(spec.family)


def main():
    kin = KineticsParameters.from_yaml(load_config('reactor_parameters'))
    t_grid = np.linspace(0.0, 10.0, 200)
    specs = build_production_registry(kin.beta_eff, t_grid)
    assert len(specs) == 2000

    # Group by family
    by_fam = defaultdict(list)
    for s in specs:
        by_fam[s.family].append(s)

    # Sort each family by canonical parameter
    for fam in by_fam:
        by_fam[fam].sort(key=canonical_key)

    # Assign splits deterministically
    train_ids, val_ids, test_ids = [], [], []
    family_split = {}
    param_coverage = {}

    for fam in ["step", "ramp", "sinusoidal", "grf"]:
        group = by_fam[fam]
        assert len(group) == PER_FAMILY, f"{fam}: {len(group)} != 500"

        # Test: every 10th starting at 0 → indices 0,10,...,490
        test_idx = set(range(0, PER_FAMILY, 10))
        # Validation: every 10th starting at 5 → indices 5,15,...,495
        val_idx = set(range(5, PER_FAMILY, 10))
        assert len(test_idx) == PER_FAMILY_TEST
        assert len(val_idx) == PER_FAMILY_VAL
        assert test_idx.isdisjoint(val_idx)

        train_f, val_f, test_f = [], [], []
        for i, s in enumerate(group):
            if i in test_idx:
                test_f.append(s.scenario_id)
            elif i in val_idx:
                val_f.append(s.scenario_id)
            else:
                train_f.append(s.scenario_id)

        assert len(train_f) == PER_FAMILY_TRAIN, f"{fam} train={len(train_f)}"
        assert len(val_f) == PER_FAMILY_VAL, f"{fam} val={len(val_f)}"
        assert len(test_f) == PER_FAMILY_TEST, f"{fam} test={len(test_f)}"

        family_split[fam] = {
            "train": len(train_f), "val": len(val_f), "test": len(test_f),
        }
        train_ids.extend(train_f)
        val_ids.extend(val_f)
        test_ids.extend(test_f)

        # Parameter coverage
        def param_range(idx_list):
            vals = [canonical_key(group[i]) for i in idx_list]
            return {"min": float(min(vals)), "max": float(max(vals)),
                    "mean": float(np.mean(vals))}

        param_coverage[fam] = {
            "train": param_range([i for i in range(PER_FAMILY) if i not in test_idx and i not in val_idx]),
            "val":   param_range(sorted(val_idx)),
            "test":  param_range(sorted(test_idx)),
        }

    # Total counts
    assert len(train_ids) == 1600, len(train_ids)
    assert len(val_ids) == 200, len(val_ids)
    assert len(test_ids) == 200, len(test_ids)

    # Global uniqueness
    all_ids = train_ids + val_ids + test_ids
    assert len(set(all_ids)) == 2000, "duplicates across splits"
    assert set(all_ids) == {s.scenario_id for s in specs}, "missing IDs"

    # Write CSVs
    def write_ids(path, ids):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario_id"])
            for sid in ids:
                w.writerow([sid])

    write_ids(R / "train_ids.csv", train_ids)
    write_ids(R / "validation_ids.csv", val_ids)
    write_ids(R / "test_ids.csv", test_ids)

    # Compute SHA-256 for each split file
    train_sha = sha256_file(R / "train_ids.csv")
    val_sha = sha256_file(R / "validation_ids.csv")
    test_sha = sha256_file(R / "test_ids.csv")

    # Split manifest
    manifest = {
        "block": "B (Step 1.4)",
        "split_seed": SPLIT_SEED,
        "split_strategy": "deterministic stratified by canonical parameter per family",
        "split_assignment": {
            "per_family_total": PER_FAMILY,
            "per_family_train": PER_FAMILY_TRAIN,
            "per_family_val": PER_FAMILY_VAL,
            "per_family_test": PER_FAMILY_TEST,
            "test_indices": "0,10,20,...,490",
            "val_indices": "5,15,25,...,495",
            "train_indices": "remaining",
        },
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": len(all_ids),
        },
        "family_split": family_split,
        "parameter_coverage": param_coverage,
        "split_file_sha256": {
            "train_ids.csv": train_sha,
            "validation_ids.csv": val_sha,
            "test_ids.csv": test_sha,
        },
    }
    (R / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Split complete")
    print(json.dumps({
        "counts": manifest["counts"],
        "family_split": family_split,
        "split_sha256": manifest["split_file_sha256"],
    }, indent=2))
    print(f"\nParameter coverage (per family):")
    for fam, cov in param_coverage.items():
        print(f"  {fam}:")
        for split in ["train", "val", "test"]:
            c = cov[split]
            print(f"    {split}: [{c['min']:.4f}, {c['max']:.4f}] mean={c['mean']:.4f}")


if __name__ == "__main__":
    main()
