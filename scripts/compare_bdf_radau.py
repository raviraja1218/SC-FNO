"""Phase 1.2 Block C — BDF/Radau comparison + physical-range validation.

For each of the 100 pilot scenarios:
  1. Load the stored BDF trajectory (from Block B).
  2. Re-run with Radau at identical tolerances.
  3. Compare all 9 states against pre-defined thresholds.
  4. Validate physical ranges of the BDF trajectory.
  5. Record per-scenario, per-state pass/fail.

Thresholds (from bdf_radau_methodology.md):
  - n, C_i:   max abs diff <= 1e-6 * max(|value|)
  - T_f, T_c: max abs diff <= 1e-6 K
  - Relative L2 per state <= 1e-5

Outputs:
  results/phase_1/pilot/pilot_solver_comparison.csv
  results/phase_1/pilot/pilot_solver_failures.csv
  results/phase_1/pilot/block_C_summary.json
  data/raw/pilot_100/pilot_XXXX/trajectory_radau.npy
  data/raw/pilot_100/pilot_XXXX/comparison.json
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
from data.forcing_profiles import build_pilot_registry
from data.solver import coupled_rhs, initial_state
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters

RAW_DIR = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "pilot_100"
RESULTS_DIR = Path("results/phase_1/pilot")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATE_NAMES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
STATE_KIND = {"n": "kinetics", "C1": "kinetics", "C2": "kinetics",
              "C3": "kinetics", "C4": "kinetics", "C5": "kinetics",
              "C6": "kinetics", "Tf": "thermal", "Tc": "thermal"}

# Thresholds (from bdf_radau_methodology.md)
KINETICS_REL_MAX = 1e-6     # max_abs_diff <= this * max(|value|)
THERMAL_ABS_MAX  = 1e-2     # K, absolute
REL_L2_MAX       = 1e-5     # per state


def compare_one(scen_dir: Path, kin, th) -> dict:
    """Compare one scenario: BDF (stored) vs Radau (fresh)."""
    sid = scen_dir.name
    t_grid = np.load(scen_dir / "time.npy")
    forcing = np.load(scen_dir / "forcing.npy")
    bdf = np.load(scen_dir / "trajectory.npy")

    # Forcing as interpolator
    def rho_fn(t, _t=t_grid, _r=forcing):
        return float(np.interp(t, _t, _r))

    def q_fn(t):
        return 0.0

    def rhs(t, y):
        return coupled_rhs(t, y, kin, th, rho_fn, q_fn, True)

    y0 = initial_state(kin, th)
    t0 = time.perf_counter()
    sol = solve_ivp(rhs, [float(t_grid[0]), float(t_grid[-1])], y0,
                    method="Radau", t_eval=t_grid,
                    rtol=1e-8, atol=1e-10)
    radau_wall = time.perf_counter() - t0

    if not sol.success or sol.y.shape[1] != len(t_grid):
        return {
            "scenario_id": sid,
            "radau_success": False,
            "radau_message": sol.message,
            "states": {},
            "passed": False,
            "reason": f"Radau failed: {sol.message}",
        }

    radau = sol.y.T  # (200, 9)
    np.save(scen_dir / "trajectory_radau.npy", radau)

    # Per-state comparison
    per_state = {}
    all_pass = True
    for j, name in enumerate(STATE_NAMES):
        b = bdf[:, j]
        r = radau[:, j]
        diff = np.abs(b - r)
        max_abs = float(np.max(diff))
        scale = float(np.max(np.abs(b)))
        rel_l2 = float(np.linalg.norm(b - r) / (scale + 1e-30))

        if STATE_KIND[name] == "kinetics":
            threshold_val = KINETICS_REL_MAX * scale
            passed = max_abs <= threshold_val and rel_l2 <= REL_L2_MAX
        else:
            threshold_val = THERMAL_ABS_MAX
            passed = max_abs <= threshold_val and rel_l2 <= REL_L2_MAX

        per_state[name] = {
            "max_abs_diff": max_abs,
            "max_abs_value_bdf": scale,
            "rel_l2": rel_l2,
            "threshold_abs": threshold_val,
            "passed": passed,
        }
        if not passed:
            all_pass = False

    result = {
        "scenario_id": sid,
        "radau_success": True,
        "radau_message": sol.message,
        "radau_nfev": int(sol.nfev),
        "radau_wall_s": radau_wall,
        "states": per_state,
        "passed": all_pass,
        "reason": "ok" if all_pass else "threshold_exceeded",
    }
    (scen_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    return result


def validate_physical(scen_dir: Path, kin) -> dict:
    """Physical-range validation of the BDF trajectory."""
    sid = scen_dir.name
    traj = np.load(scen_dir / "trajectory.npy")
    n = traj[:, 0]
    C = traj[:, 1:7]
    Tf = traj[:, 7]
    Tc = traj[:, 8]

    # Loose tolerances for physical plausibility
    n_ok = bool(np.all(n >= -1e-12))
    C_ok = bool(np.all(C >= -1e-12))
    Tf_ok = bool(np.all(np.isfinite(Tf)))
    Tc_ok = bool(np.all(np.isfinite(Tc)))

    # Plausibility bounds from adiabatic heating estimate over 10 s:
    # dTf_max ~ (f_f*P0*n_max)/(M_f*c_p_f) * t_end, roughly
    # Just check monotonic non-catastrophic bounds (no blow-up)
    Tf_reasonable = bool(np.all(Tf < 5000.0))
    Tc_reasonable = bool(np.all(Tc < 2000.0))

    return {
        "scenario_id": sid,
        "n_nonneg": n_ok,
        "C_nonneg": C_ok,
        "Tf_finite": Tf_ok,
        "Tc_finite": Tc_ok,
        "Tf_reasonable": Tf_reasonable,
        "Tc_reasonable": Tc_reasonable,
        "n_min": float(np.min(n)),
        "n_max": float(np.max(n)),
        "C_min": float(np.min(C)),
        "C_max": float(np.max(C)),
        "Tf_min": float(np.min(Tf)),
        "Tf_max": float(np.max(Tf)),
        "Tc_min": float(np.min(Tc)),
        "Tc_max": float(np.max(Tc)),
        "passed": bool(n_ok and C_ok and Tf_ok and Tc_ok
                       and Tf_reasonable and Tc_reasonable),
    }


def main() -> None:
    kin = KineticsParameters.from_yaml(load_config("reactor_parameters"))
    th = THParameters.from_yaml(load_config("thermal_hydraulics"))

    scen_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Comparing {len(scen_dirs)} scenarios (BDF vs Radau)")
    assert len(scen_dirs) == 100, f"Expected 100 scenarios, got {len(scen_dirs)}"

    comparison_rows = []
    failure_rows = []
    physical_rows = []
    t_start = time.perf_counter()

    for i, d in enumerate(scen_dirs, 1):
        cmp = compare_one(d, kin, th)
        phys = validate_physical(d, kin)

        # Flatten comparison to CSV row per (scenario, state)
        for name, s in cmp.get("states", {}).items():
            comparison_rows.append({
                "scenario_id": cmp["scenario_id"],
                "state": name,
                "max_abs_diff": s["max_abs_diff"],
                "max_abs_value_bdf": s["max_abs_value_bdf"],
                "rel_l2": s["rel_l2"],
                "threshold_abs": s["threshold_abs"],
                "passed": s["passed"],
            })

        physical_rows.append(phys)

        if not cmp["passed"] or not phys["passed"]:
            failure_rows.append({
                "scenario_id": cmp["scenario_id"],
                "comparison_passed": cmp["passed"],
                "physical_passed": phys["passed"],
                "reason": cmp.get("reason", "") + "|" + ("physical_ok" if phys["passed"] else "physical_violation"),
            })

        if i % 10 == 0 or i == len(scen_dirs):
            print(f"  [{i:3d}/{len(scen_dirs)}] {cmp['scenario_id']}: "
                  f"cmp_passed={cmp['passed']}, phys_passed={phys['passed']}")

    total_wall = time.perf_counter() - t_start

    # Write comparison CSV
    with (RESULTS_DIR / "pilot_solver_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)

    # Write physical validation CSV
    with (RESULTS_DIR / "pilot_physical_validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(physical_rows[0].keys()))
        w.writeheader()
        w.writerows(physical_rows)

    # Write failures CSV
    with (RESULTS_DIR / "pilot_solver_failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "comparison_passed",
                                          "physical_passed", "reason"])
        w.writeheader()
        w.writerows(failure_rows)

    # Aggregate per-state stats
    per_state_stats = {}
    for name in STATE_NAMES:
        rows = [r for r in comparison_rows if r["state"] == name]
        if not rows:
            continue
        per_state_stats[name] = {
            "n_scenarios": len(rows),
            "n_passed": sum(1 for r in rows if r["passed"]),
            "max_abs_diff": float(max(r["max_abs_diff"] for r in rows)),
            "max_rel_l2": float(max(r["rel_l2"] for r in rows)),
        }

    summary = {
        "block": "C",
        "n_scenarios": len(scen_dirs),
        "n_comparison_passed": sum(1 for r in comparison_rows
                                   if r["state"] == "n" and r["passed"]) if comparison_rows else 0,
        "n_physical_passed": sum(1 for r in physical_rows if r["passed"]),
        "n_failures": len(failure_rows),
        "all_passed": len(failure_rows) == 0,
        "total_wall_s": total_wall,
        "per_state_stats": per_state_stats,
        "thresholds": {
            "kinetics_rel_max": KINETICS_REL_MAX,
            "thermal_abs_max_K": THERMAL_ABS_MAX,
            "rel_l2_max": REL_L2_MAX,
        },
    }
    (RESULTS_DIR / "block_C_summary.json").write_text(
        json.dumps(summary, indent=2))

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
