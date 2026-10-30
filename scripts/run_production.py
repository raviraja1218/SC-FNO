"""Step 1.3 Block B — run all 2,000 production scenarios through BDF.

Per-scenario output under ~/Research/Datasets/SC-FNO/raw/scenarios_2000/scenario_XXXX/:
    forcing.npy          (200,)
    trajectory.npy       (200, 9)
    time.npy             (200,)
    metadata.json
    solver_diagnostics.json

Aggregate outputs under results/phase_1/generation/:
    block_B_summary.json
    block_B_runtime.csv
    block_B_failures.csv
    block_B_run_log.txt
"""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')

import csv
import json
import time
from pathlib import Path

import numpy as np

from data.config_loader import load_config
from data.forcing_profiles import build_production_registry, generate_forcing
from data.solver import solve_one_scenario
from physics.kinetics import KineticsParameters
from physics.thermal_hydraulics import THParameters

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"
RESULTS_DIR = ROOT / "results" / "phase_1" / "generation"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    kin = KineticsParameters.from_yaml(load_config("reactor_parameters"))
    th  = THParameters.from_yaml(load_config("thermal_hydraulics"))
    t_grid = np.linspace(0.0, 10.0, 200)
    specs = build_production_registry(kin.beta_eff, t_grid)

    print(f"Running {len(specs)} production scenarios")
    print(f"  RAW_DIR     = {RAW_DIR}")
    print(f"  RESULTS_DIR = {RESULTS_DIR}")
    print(f"  BDF, rtol=1e-8, atol=1e-10, {len(t_grid)} output points, t_end={t_grid[-1]} s")
    print()

    runtime_rows = []
    failure_rows = []
    n_success = 0
    total_solver_wall = 0.0
    total_wall_start = time.perf_counter()

    for i, spec in enumerate(specs, 1):
        scen_dir = RAW_DIR / spec.scenario_id
        scen_dir.mkdir(parents=True, exist_ok=True)

        rho_arr = generate_forcing(spec, t_grid)

        def rho_fn(t, _t=t_grid, _r=rho_arr):
            return float(np.interp(t, _t, _r))

        res = solve_one_scenario(
            spec, t_grid, kin, th, rho_fn,
            solver_method="BDF", rtol=1e-8, atol=1e-10,
            q_t_default=0.0, include_q=True,
            config_versions={"reactor_parameters": "v3",
                             "thermal_hydraulics": "v2"},
        )
        total_solver_wall += res.wall_time_s

        # Save forcing and diagnostics (always)
        np.save(scen_dir / "forcing.npy", rho_arr)
        np.save(scen_dir / "time.npy", t_grid)
        meta = {
            "scenario_id": spec.scenario_id,
            "family": spec.family,
            "seed": int(spec.seed),
            "params": spec.params,
            "t_end": float(t_grid[-1]),
            "n_points": int(len(t_grid)),
            "beta_eff": float(kin.beta_eff),
            "config_versions": {"reactor_parameters": "v3",
                                "thermal_hydraulics": "v2"},
            "q_t_default": 0.0,
            "include_q": True,
        }
        (scen_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        diag = {
            "scenario_id": spec.scenario_id,
            "success": bool(res.success),
            "message": res.message,
            "method": res.method,
            "rtol": res.rtol,
            "atol": res.atol,
            "nfev": res.nfev,
            "njev": res.njev,
            "nlu": res.nlu,
            "wall_time_s": res.wall_time_s,
        }
        (scen_dir / "solver_diagnostics.json").write_text(
            json.dumps(diag, indent=2))

        if res.success:
            np.save(scen_dir / "trajectory.npy", res.trajectory)
            n_success += 1
        else:
            failure_rows.append({
                "scenario_id": spec.scenario_id,
                "family": spec.family,
                "message": res.message,
            })

        runtime_rows.append({
            "scenario_id": spec.scenario_id,
            "family": spec.family,
            "success": res.success,
            "nfev": res.nfev,
            "njev": res.njev,
            "nlu": res.nlu,
            "wall_time_s": res.wall_time_s,
        })

        if i % 100 == 0 or i == len(specs):
            elapsed = time.perf_counter() - total_wall_start
            eta = elapsed / i * (len(specs) - i)
            print(f"  [{i:4d}/{len(specs)}] elapsed={elapsed:.1f}s "
                  f"eta={eta:.1f}s success={n_success}")

    total_wall = time.perf_counter() - total_wall_start
    wall_arr = np.array([r["wall_time_s"] for r in runtime_rows if r["success"]])

    # Per-family stats
    fam_stats = {}
    for fam in ["step", "ramp", "sinusoidal", "grf"]:
        fam_walls = [r["wall_time_s"] for r in runtime_rows
                     if r["family"] == fam and r["success"]]
        if fam_walls:
            fam_stats[fam] = {
                "n": len(fam_walls),
                "mean_s": float(np.mean(fam_walls)),
                "median_s": float(np.median(fam_walls)),
                "total_s": float(np.sum(fam_walls)),
            }

    summary = {
        "block": "B (Step 1.3)",
        "n_scenarios": len(specs),
        "n_success": n_success,
        "n_failures": len(failure_rows),
        "all_passed": n_success == len(specs),
        "total_wall_time_s": total_wall,
        "sum_solver_wall_time_s": total_solver_wall,
        "solver_wall_stats": ({
            "mean_s": float(np.mean(wall_arr)),
            "median_s": float(np.median(wall_arr)),
            "std_s": float(np.std(wall_arr)),
            "min_s": float(np.min(wall_arr)),
            "max_s": float(np.max(wall_arr)),
            "p90_s": float(np.percentile(wall_arr, 90)),
            "p95_s": float(np.percentile(wall_arr, 95)),
        } if len(wall_arr) else {}),
        "per_family": fam_stats,
        "t_grid": {"n_points": int(len(t_grid)),
                   "t_start": float(t_grid[0]),
                   "t_end": float(t_grid[-1])},
        "state_vector_dim": 9,
        "scenario_dirs": str(RAW_DIR),
    }
    (RESULTS_DIR / "block_B_summary.json").write_text(
        json.dumps(summary, indent=2))

    with (RESULTS_DIR / "block_B_runtime.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runtime_rows[0].keys()))
        w.writeheader()
        w.writerows(runtime_rows)

    with (RESULTS_DIR / "block_B_failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario_id", "family", "message"])
        w.writeheader()
        w.writerows(failure_rows)

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
