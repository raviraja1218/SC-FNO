"""Phase 1.2 Block D — tolerance-sensitivity sweep.

For 20 representative pilot scenarios (5 per forcing family):
  - Load reference trajectory (BDF, rtol=1e-8, atol=1e-10) from Block B
  - Rerun BDF with tighter tolerances (rtol=1e-10, atol=1e-12)
  - Compare all 9 states

Acceptance: relative L2 difference < 1e-4 per state per scenario.

Outputs:
  results/phase_1/pilot/pilot_tolerance_sensitivity.csv
  results/phase_1/pilot/block_D_summary.json
  results/phase_1/pilot/block_D_failures.csv
  data/raw/pilot_100/pilot_XXXX/trajectory_tight.npy
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
REL_L2_MAX = 1e-4          # tightened-run vs reference acceptance
RTOL_REF, ATOL_REF = 1e-8, 1e-10
RTOL_TIGHT, ATOL_TIGHT = 1e-10, 1e-12


def pick_20(specs: list) -> list:
    """Pick 5 scenarios per family, spread across parameter ranges."""
    per_fam = {"step": [], "ramp": [], "sinusoidal": [], "grf": []}
    for s in specs:
        per_fam[s.family].append(s)
    picks = []
    for fam in ("step", "ramp", "sinusoidal", "grf"):
        group = per_fam[fam]
        assert len(group) == 25
        # 5 spread across the 25 (indices 0, 6, 12, 18, 24)
        idxs = [0, 6, 12, 18, 24]
        picks.extend(group[i] for i in idxs)
    assert len(picks) == 20
    return picks


def solve_with_tolerances(spec, t_grid, kin, th, forcing,
                          rtol, atol, method="BDF"):
    def rho_fn(t, _t=t_grid, _r=forcing):
        return float(np.interp(t, _t, _r))

    def q_fn(t):
        return 0.0

    def rhs(t, y):
        return coupled_rhs(t, y, kin, th, rho_fn, q_fn, True)

    y0 = initial_state(kin, th)
    sol = solve_ivp(rhs, [float(t_grid[0]), float(t_grid[-1])], y0,
                    method=method, t_eval=t_grid,
                    rtol=rtol, atol=atol)
    return sol


def main() -> None:
    kin = KineticsParameters.from_yaml(load_config("reactor_parameters"))
    th  = THParameters.from_yaml(load_config("thermal_hydraulics"))
    t_grid = np.linspace(0.0, 10.0, 200)
    specs = build_pilot_registry(kin.beta_eff, t_grid)
    picks = pick_20(specs)

    print(f"Tolerance sweep: {len(picks)} scenarios (5 per family)")
    print(f"  Reference:   rtol={RTOL_REF:g}, atol={ATOL_REF:g}")
    print(f"  Tightened:   rtol={RTOL_TIGHT:g}, atol={ATOL_TIGHT:g}")

    rows = []
    failures = []
    t_start = time.perf_counter()

    for i, spec in enumerate(picks, 1):
        scen_dir = RAW_DIR / spec.scenario_id
        t_loaded = np.load(scen_dir / "time.npy")
        forcing  = np.load(scen_dir / "forcing.npy")
        ref = np.load(scen_dir / "trajectory.npy")  # from Block B

        sol = solve_with_tolerances(spec, t_loaded, kin, th, forcing,
                                    RTOL_TIGHT, ATOL_TIGHT, "BDF")
        if not sol.success or sol.y.shape[1] != len(t_loaded):
            failures.append({
                "scenario_id": spec.scenario_id,
                "reason": f"tight solve failed: {sol.message}",
            })
            continue

        tight = sol.y.T
        np.save(scen_dir / "trajectory_tight.npy", tight)

        for j, name in enumerate(STATE_NAMES):
            r = ref[:, j]
            t_new = tight[:, j]
            diff = np.abs(r - t_new)
            scale = float(np.max(np.abs(r))) + 1e-30
            rel_l2 = float(np.linalg.norm(r - t_new) / scale)
            rows.append({
                "scenario_id": spec.scenario_id,
                "family": spec.family,
                "state": name,
                "max_abs_diff": float(np.max(diff)),
                "rel_l2": rel_l2,
                "scale": scale,
                "passed": rel_l2 < REL_L2_MAX,
            })

        if i % 5 == 0:
            print(f"  [{i:2d}/{len(picks)}] {spec.scenario_id} ({spec.family}) done")

    total_wall = time.perf_counter() - t_start

    with (RESULTS_DIR / "pilot_tolerance_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (RESULTS_DIR / "block_D_failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "reason"])
        w.writeheader()
        w.writerows(failures)

    per_state = {}
    for name in STATE_NAMES:
        rs = [r for r in rows if r["state"] == name]
        per_state[name] = {
            "n_scenarios": len(rs),
            "n_passed": sum(1 for r in rs if r["passed"]),
            "max_rel_l2": float(max(r["rel_l2"] for r in rs)) if rs else 0.0,
            "max_abs_diff": float(max(r["max_abs_diff"] for r in rs)) if rs else 0.0,
        }

    summary = {
        "block": "D",
        "n_scenarios": len(picks),
        "n_rows": len(rows),
        "n_failures": len(failures),
        "all_passed": len(failures) == 0 and all(r["passed"] for r in rows),
        "total_wall_s": total_wall,
        "tolerances": {
            "reference": {"rtol": RTOL_REF, "atol": ATOL_REF},
            "tightened": {"rtol": RTOL_TIGHT, "atol": ATOL_TIGHT},
        },
        "acceptance_rel_l2_max": REL_L2_MAX,
        "per_state_stats": per_state,
    }
    (RESULTS_DIR / "block_D_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
