"""Step 1.3 Block C — Radau cross-check (200) + tolerance sweep (40).

Radau cross-check:
  - 200 scenarios selected evenly (every 10th scenario from 2000)
  - Rerun with Radau at rtol=1e-8, atol=1e-10
  - Compare all 9 states to BDF reference using v2 thresholds
  - Save trajectory_radau.npy per scenario + comparison.json
  - Save radau_comparison.csv (200×9 = 1800 rows)

Tolerance sweep:
  - 40 scenarios: 10 evenly-spaced per family
  - Rerun with BDF at rtol=1e-10, atol=1e-12
  - Compare to reference (rtol=1e-8, atol=1e-10)
  - Acceptance: relative L2 < 1e-4 per state
  - Save tolerance_sensitivity.csv (40×9 = 360 rows)

Thresholds (v2 methodology):
  Kinetics (n, C_i): max_abs_diff <= 1e-6 * max(|BDF value|)
  Thermal (Tf, Tc):  max_abs_diff <= 1e-2 K
  All:               rel_l2 <= 1e-5 (comparison) / 1e-4 (tolerance)
"""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from data.config_loader import load_config
from data.forcing_profiles import build_production_registry
from data.solver import coupled_rhs, initial_state
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters

RAW_DIR = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"
R = Path('results/phase_1/generation')
R.mkdir(parents=True, exist_ok=True)

STATE_NAMES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
STATE_KIND = {n: "kinetics" for n in STATE_NAMES[:7]}
STATE_KIND.update({"Tf": "thermal", "Tc": "thermal"})

KINETICS_REL_MAX = 1e-6
THERMAL_ABS_MAX = 1e-2
REL_L2_CMP_MAX = 1e-5
REL_L2_TOL_MAX = 1e-4


def pick_radau_sample(specs):
    """200 evenly-spaced: indices 0, 10, 20, ..., 1990."""
    assert len(specs) == 2000
    return [specs[i] for i in range(0, 2000, 10)]


def pick_tolerance_sample(specs):
    """40 scenarios: 10 evenly spaced per family."""
    per_fam = {"step": [], "ramp": [], "sinusoidal": [], "grf": []}
    for s in specs:
        per_fam[s.family].append(s)
    picks = []
    for fam in ("step", "ramp", "sinusoidal", "grf"):
        group = per_fam[fam]  # 500 each
        assert len(group) == 500
        # 10 evenly spaced: 0, 50, 100, ..., 450
        picks.extend(group[i] for i in range(0, 500, 50))
    assert len(picks) == 40
    return picks


def solve_bdf(t_grid, forcing, kin, th, rtol, atol, method):
    def rho_fn(t, _t=t_grid, _r=forcing):
        return float(np.interp(t, _t, _r))

    def q_fn(t):
        return 0.0

    def rhs(t, y):
        return coupled_rhs(t, y, kin, th, rho_fn, q_fn, True)

    y0 = initial_state(kin, th)
    return solve_ivp(rhs, [float(t_grid[0]), float(t_grid[-1])], y0,
                     method=method, t_eval=t_grid, rtol=rtol, atol=atol)


def compare_state(bdf, other, name):
    """Return per-state comparison dict."""
    diff = np.abs(bdf - other)
    max_abs = float(np.max(diff))
    scale = float(np.max(np.abs(bdf))) + 1e-30
    rel_l2 = float(np.linalg.norm(bdf - other) / scale)
    if STATE_KIND[name] == "kinetics":
        thr = KINETICS_REL_MAX * scale
        passed = max_abs <= thr and rel_l2 <= REL_L2_CMP_MAX
    else:
        thr = THERMAL_ABS_MAX
        passed = max_abs <= thr and rel_l2 <= REL_L2_CMP_MAX
    return {
        "max_abs_diff": max_abs,
        "scale": scale,
        "rel_l2": rel_l2,
        "threshold_abs": thr,
        "passed": passed,
    }


def main():
    kin = KineticsParameters.from_yaml(load_config("reactor_parameters"))
    th = THParameters.from_yaml(load_config("thermal_hydraulics"))
    t_grid = np.linspace(0.0, 10.0, 200)
    specs = build_production_registry(kin.beta_eff, t_grid)

    print("=" * 60)
    print("RADAU CROSS-CHECK (200 scenarios)")
    print("=" * 60)
    radau_picks = pick_radau_sample(specs)
    radau_rows = []
    radau_failures = []
    t0 = time.perf_counter()

    for i, spec in enumerate(radau_picks, 1):
        d = RAW_DIR / spec.scenario_id
        t = np.load(d / "time.npy")
        forcing = np.load(d / "forcing.npy")
        bdf = np.load(d / "trajectory.npy")

        sol = solve_bdf(t, forcing, kin, th, 1e-8, 1e-10, "Radau")
        if not sol.success or sol.y.shape[1] != len(t):
            radau_failures.append({"scenario_id": spec.scenario_id,
                                   "reason": sol.message})
            continue
        radau = sol.y.T
        np.save(d / "trajectory_radau.npy", radau)

        per_state = {}
        all_pass = True
        for j, name in enumerate(STATE_NAMES):
            r = compare_state(bdf[:, j], radau[:, j], name)
            per_state[name] = r
            radau_rows.append({
                "scenario_id": spec.scenario_id,
                "family": spec.family,
                "state": name,
                "max_abs_diff": r["max_abs_diff"],
                "scale": r["scale"],
                "rel_l2": r["rel_l2"],
                "threshold_abs": r["threshold_abs"],
                "passed": r["passed"],
            })
            if not r["passed"]:
                all_pass = False

        (d / "comparison.json").write_text(json.dumps(
            {"scenario_id": spec.scenario_id, "states": per_state,
             "passed": all_pass}, indent=2))

        if i % 50 == 0 or i == len(radau_picks):
            print(f"  [{i:3d}/{len(radau_picks)}] {spec.scenario_id} "
                  f"cmp_passed={all_pass}")

    radau_wall = time.perf_counter() - t0

    # Write radau comparison CSV
    with (R / "radau_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "family", "state",
                                          "max_abs_diff", "scale", "rel_l2",
                                          "threshold_abs", "passed"])
        w.writeheader()
        w.writerows(radau_rows)

    with (R / "radau_failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "reason"])
        w.writeheader()
        w.writerows(radau_failures)

    # Per-state stats
    radau_per_state = {}
    for name in STATE_NAMES:
        rs = [r for r in radau_rows if r["state"] == name]
        radau_per_state[name] = {
            "n": len(rs),
            "n_passed": sum(1 for r in rs if r["passed"]),
            "max_abs_diff": float(max(r["max_abs_diff"] for r in rs)) if rs else 0.0,
            "max_rel_l2": float(max(r["rel_l2"] for r in rs)) if rs else 0.0,
        }

    print(f"\nRadau wall time: {radau_wall:.1f}s")

    # --- Tolerance sweep
    print("\n" + "=" * 60)
    print("TOLERANCE SENSITIVITY SWEEP (40 scenarios)")
    print("=" * 60)
    tol_picks = pick_tolerance_sample(specs)
    tol_rows = []
    tol_failures = []
    t0 = time.perf_counter()

    for i, spec in enumerate(tol_picks, 1):
        d = RAW_DIR / spec.scenario_id
        t = np.load(d / "time.npy")
        forcing = np.load(d / "forcing.npy")
        ref = np.load(d / "trajectory.npy")

        sol = solve_bdf(t, forcing, kin, th, 1e-10, 1e-12, "BDF")
        if not sol.success or sol.y.shape[1] != len(t):
            tol_failures.append({"scenario_id": spec.scenario_id,
                                 "reason": sol.message})
            continue
        tight = sol.y.T
        np.save(d / "trajectory_tight.npy", tight)

        for j, name in enumerate(STATE_NAMES):
            ref_arr = ref[:, j]
            diff = np.abs(ref_arr - tight[:, j])
            scale = float(np.max(np.abs(ref_arr))) + 1e-30
            rel_l2 = float(np.linalg.norm(ref_arr - tight[:, j]) / scale)
            tol_rows.append({
                "scenario_id": spec.scenario_id,
                "family": spec.family,
                "state": name,
                "max_abs_diff": float(np.max(diff)),
                "rel_l2": rel_l2,
                "scale": scale,
                "passed": rel_l2 < REL_L2_TOL_MAX,
            })

        if i % 10 == 0 or i == len(tol_picks):
            print(f"  [{i:2d}/{len(tol_picks)}] {spec.scenario_id} ({spec.family})")

    tol_wall = time.perf_counter() - t0

    with (R / "tolerance_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "family", "state",
                                          "max_abs_diff", "rel_l2", "scale",
                                          "passed"])
        w.writeheader()
        w.writerows(tol_rows)

    with (R / "tolerance_failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "reason"])
        w.writeheader()
        w.writerows(tol_failures)

    tol_per_state = {}
    for name in STATE_NAMES:
        rs = [r for r in tol_rows if r["state"] == name]
        tol_per_state[name] = {
            "n": len(rs),
            "n_passed": sum(1 for r in rs if r["passed"]),
            "max_rel_l2": float(max(r["rel_l2"] for r in rs)) if rs else 0.0,
        }

    print(f"\nTolerance sweep wall time: {tol_wall:.1f}s")

    # Summary
    summary = {
        "block": "C (Step 1.3)",
        "radau": {
            "n_scenarios": len(radau_picks),
            "n_rows": len(radau_rows),
            "n_failures": len(radau_failures),
            "all_passed": len(radau_failures) == 0 and all(r["passed"] for r in radau_rows),
            "wall_time_s": radau_wall,
            "per_state": radau_per_state,
        },
        "tolerance": {
            "n_scenarios": len(tol_picks),
            "n_rows": len(tol_rows),
            "n_failures": len(tol_failures),
            "all_passed": len(tol_failures) == 0 and all(r["passed"] for r in tol_rows),
            "wall_time_s": tol_wall,
            "per_state": tol_per_state,
        },
        "thresholds": {
            "kinetics_rel_max": KINETICS_REL_MAX,
            "thermal_abs_max_K": THERMAL_ABS_MAX,
            "rel_l2_comparison_max": REL_L2_CMP_MAX,
            "rel_l2_tolerance_max": REL_L2_TOL_MAX,
        },
    }
    (R / "block_C_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
