"""Step 1.4 Block A — full integrity verification of 2,000 scenarios.

Read-only against ~/Research/Datasets/SC-FNO/raw/scenarios_2000/.
"""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from data.config_loader import load_config
from physics.kinetics import KineticsParameters, equilibrium_precursors

RAW_DIR = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"
R = Path('results/phase_1/splitting')
R.mkdir(parents=True, exist_ok=True)

# Expected initial state
INIT_N_TOL = 1e-12
INIT_TF_TOL = 1e-10
INIT_TC_TOL = 1e-10


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    kin = KineticsParameters.from_yaml(load_config('reactor_parameters'))

    # Load Step 1.3 manifest for cross-reference
    manifest_path = Path('results/phase_1/generation/scenario_manifest_2000.csv')
    with manifest_path.open() as f:
        manifest_rows = {r["scenario_id"]: r for r in csv.DictReader(f)}

    scen_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Validating {len(scen_dirs)} scenarios")
    print(f"  RAW_DIR = {RAW_DIR}")
    print(f"  Manifest scenarios: {len(manifest_rows)}")
    print()

    failures = []
    trajectory_hashes = []
    forcing_hashes = []
    t0 = time.perf_counter()

    for i, d in enumerate(scen_dirs, 1):
        sid = d.name
        failure_reasons = []

        # 1. Required files
        required = ["forcing.npy", "trajectory.npy", "time.npy",
                    "metadata.json", "solver_diagnostics.json"]
        for f in required:
            if not (d / f).exists():
                failure_reasons.append(f"missing {f}")

        if failure_reasons:
            failures.append({"scenario_id": sid, "family": "?",
                             "reasons": "; ".join(failure_reasons)})
            continue

        # 2. Load arrays
        try:
            forcing = np.load(d / "forcing.npy")
            traj = np.load(d / "trajectory.npy")
            t = np.load(d / "time.npy")
        except Exception as e:
            failures.append({"scenario_id": sid, "family": "?",
                             "reasons": f"load error: {e}"})
            continue

        # 3. Shapes
        if forcing.shape != (200,):
            failure_reasons.append(f"forcing shape {forcing.shape} != (200,)")
        if traj.shape != (200, 9):
            failure_reasons.append(f"traj shape {traj.shape} != (200,9)")
        if t.shape != (200,):
            failure_reasons.append(f"time shape {t.shape} != (200,)")

        # 4. Time grid
        if len(t) == 200:
            if not np.all(np.diff(t) > 0):
                failure_reasons.append("time not strictly increasing")
            if abs(t[0] - 0.0) > 1e-12:
                failure_reasons.append(f"t[0]={t[0]}")
            if abs(t[-1] - 10.0) > 1e-10:
                failure_reasons.append(f"t[-1]={t[-1]}")

        # 5. Finiteness
        if not np.all(np.isfinite(forcing)):
            failure_reasons.append("forcing has non-finite")
        if not np.all(np.isfinite(traj)):
            failure_reasons.append("traj has non-finite")
        if not np.all(np.isfinite(t)):
            failure_reasons.append("time has non-finite")

        # 6. Initial conditions
        if traj.shape == (200, 9):
            if abs(traj[0, 0] - 1.0) > INIT_N_TOL:
                failure_reasons.append(f"n0={traj[0,0]:.15g}")
            if abs(traj[0, 7] - 900.0) > INIT_TF_TOL:
                failure_reasons.append(f"Tf0={traj[0,7]:.15g}")
            if abs(traj[0, 8] - 583.0) > INIT_TC_TOL:
                failure_reasons.append(f"Tc0={traj[0,8]:.15g}")
            # Precursor equilibrium
            C_eq = equilibrium_precursors(1.0, kin)
            if not np.allclose(traj[0, 1:7], C_eq, rtol=1e-10, atol=1e-10):
                failure_reasons.append("C_i0 != equilibrium")

        # 7. Metadata
        try:
            meta = json.load(open(d / "metadata.json"))
            if meta["scenario_id"] != sid:
                failure_reasons.append(f"meta sid mismatch: {meta['scenario_id']}")
            fam = meta.get("family", "?")
        except Exception as e:
            failure_reasons.append(f"meta load: {e}")
            fam = "?"

        # 8. Solver diagnostics
        try:
            diag = json.load(open(d / "solver_diagnostics.json"))
            if not diag.get("success", False):
                failure_reasons.append(f"solver success=False: {diag.get('message')}")
        except Exception as e:
            failure_reasons.append(f"diag load: {e}")

        # 9. Manifest cross-reference
        if sid not in manifest_rows:
            failure_reasons.append("not in Step 1.3 manifest")
        else:
            mrow = manifest_rows[sid]
            if mrow["family"] != fam:
                failure_reasons.append(f"family mismatch: manifest={mrow['family']} meta={fam}")

        # 10. Compute hashes for later
        trajectory_hashes.append({"scenario_id": sid, "family": fam,
                                  "sha256": sha256_file(d / "trajectory.npy")})
        forcing_hashes.append({"scenario_id": sid, "family": fam,
                               "sha256": sha256_file(d / "forcing.npy")})

        if failure_reasons:
            failures.append({"scenario_id": sid, "family": fam,
                             "reasons": "; ".join(failure_reasons)})

        if i % 500 == 0 or i == len(scen_dirs):
            print(f"  [{i:4d}/{len(scen_dirs)}] failures so far: {len(failures)}")

    val_wall = time.perf_counter() - t0

    # Write failures
    with (R / "integrity_failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "family", "reasons"])
        w.writeheader()
        w.writerows(failures)

    # Write hashes
    with (R / "scenario_hashes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "family", "trajectory_sha256", "forcing_sha256"])
        traj_map = {r["scenario_id"]: r["sha256"] for r in trajectory_hashes}
        for r in forcing_hashes:
            w.writerow([r["scenario_id"], r["family"],
                        traj_map.get(r["scenario_id"], ""), r["sha256"]])

    # Summary
    from collections import Counter
    fam_counts = Counter(r["family"] for r in trajectory_hashes)
    summary = {
        "block": "A (Step 1.4)",
        "n_scenarios_checked": len(scen_dirs),
        "n_passed": len(scen_dirs) - len(failures),
        "n_failures": len(failures),
        "all_passed": len(failures) == 0,
        "wall_time_s": val_wall,
        "family_counts": dict(fam_counts),
        "hashes_computed": len(trajectory_hashes),
    }
    (R / "block_A_summary.json").write_text(json.dumps(summary, indent=2))

    # Integrity report
    report = f"""# Step 1.4 Block A — Integrity Verification Report

## Status
- Scenarios checked: {summary['n_scenarios_checked']}
- Passed: {summary['n_passed']}
- Failed: {summary['n_failures']}
- Wall time: {val_wall:.1f} s

## Checks performed per scenario
1. All 5 required files present
2. Arrays load cleanly
3. forcing shape == (200,)
4. trajectory shape == (200, 9)
5. time shape == (200,)
6. time strictly increasing, [0, 10]
7. No NaN or Inf in any array
8. Initial state: n0=1, Tf0=900 K, Tc0=583 K, C_i0=equilibrium
9. metadata.json present and consistent (scenario_id matches dir name)
10. solver_diagnostics.json shows success=true
11. Family consistent with Step 1.3 manifest
12. Scenario ID present in Step 1.3 manifest

## Family distribution (validated scenarios)
"""
    for fam in ["step", "ramp", "sinusoidal", "grf"]:
        report += f"- {fam}: {fam_counts.get(fam, 0)}\n"
    report += f"\n## Failures\n"
    if failures:
        report += f"{len(failures)} scenarios failed integrity checks. See `integrity_failures.csv`.\n"
    else:
        report += "None — all 2,000 scenarios pass all integrity checks.\n"

    (R / "integrity_report.md").write_text(report)

    print()
    print(json.dumps(summary, indent=2))
    if failures:
        print(f"\nFirst 5 failures:")
        for f_ in failures[:5]:
            print(f"  {f_['scenario_id']}: {f_['reasons']}")


if __name__ == "__main__":
    main()
