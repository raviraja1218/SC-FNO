"""Step 1.3 Block E — figures + dataset summary from 2,000 production scenarios.

Reads from ~/Research/Datasets/SC-FNO/raw/scenarios_2000/ and
results/phase_1/generation/.
"""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.config_loader import load_config
from data.forcing_profiles import build_production_registry
from physics.kinetics import KineticsParameters

RAW_DIR = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"
R = Path('results/phase_1/generation')
FIGS = R / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

STATE_NAMES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]

# Representative scenario per family (median-index within family for reproducibility)
PICKS = {
    "step":       "scenario_0250",   # step family: indices 1..500
    "ramp":       "scenario_0750",   # ramp family: indices 501..1000
    "sinusoidal": "scenario_1250",   # sinusoidal family: indices 1001..1500
    "grf":        "scenario_1750",   # grf family: indices 1501..2000
}

COLORS = {"step": "C0", "ramp": "C1", "sinusoidal": "C2", "grf": "C3"}


def load_rep():
    out = {}
    for fam, sid in PICKS.items():
        d = RAW_DIR / sid
        out[fam] = {
            "sid": sid,
            "t": np.load(d / "time.npy"),
            "forcing": np.load(d / "forcing.npy"),
            "traj": np.load(d / "trajectory.npy"),
        }
    return out


def fig_composite_inputs(data, kin):
    """Manuscript Figure 2 — input reactivity profiles."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for fam, d in data.items():
        ax.plot(d["t"], d["forcing"] / kin.beta_eff, color=COLORS[fam],
                lw=1.5, label=f"{fam} ({d['sid']})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\rho_{ext}(t) / \beta_{eff}$")
    ax.set_title("Input: representative external reactivity (one per family)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "production_input_reactivity.png", dpi=140)
    plt.close()
    print("Saved: production_input_reactivity.png")


def fig_composite_responses(data):
    """Manuscript Figure 2 — 9-state responses composite (4 panels × 4 families)."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for row, (fam, d) in enumerate(data.items()):
        t = d["t"]
        traj = d["traj"]

        # neutron density
        ax = axes[row, 0]
        ax.plot(t, traj[:, 0], color=COLORS[fam], lw=1.5)
        ax.set_ylabel("n(t) / n(0)")
        ax.set_title(f"{fam} ({d['sid']}): neutron density")
        ax.grid(alpha=0.3)

        # total precursors
        ax = axes[row, 1]
        ax.plot(t, traj[:, 1:7].sum(axis=1), color=COLORS[fam], lw=1.5)
        ax.set_ylabel("sum C_i(t)")
        ax.set_title(f"{fam}: total precursors")
        ax.grid(alpha=0.3)

        # fuel + coolant temperatures
        ax = axes[row, 2]
        ax.plot(t, traj[:, 7], color=COLORS[fam], lw=1.5, label="T_f")
        ax.plot(t, traj[:, 8], color=COLORS[fam], lw=1.0, ls="--", label="T_c")
        ax.set_ylabel("T [K]")
        ax.set_xlabel("t [s]")
        ax.set_title(f"{fam}: fuel/coolant temperatures")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS / "production_responses_composite.png", dpi=140)
    plt.close()
    print("Saved: production_responses_composite.png")


def fig_neutron_density(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    for fam, d in data.items():
        ax.plot(d["t"], d["traj"][:, 0], color=COLORS[fam], lw=1.5,
                label=f"{fam} ({d['sid']})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("n(t) / n(0)")
    ax.set_title("Neutron density — one representative scenario per family")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "production_neutron_density.png", dpi=140)
    plt.close()
    print("Saved: production_neutron_density.png")


def fig_precursors(data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (fam, d) in zip(axes.flat, data.items()):
        t = d["t"]
        for i in range(6):
            ax.plot(t, d["traj"][:, 1 + i], lw=1.3, label=f"C{i+1}")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("C_i(t)")
        ax.set_title(f"{fam} ({d['sid']}): 6 delayed precursors")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "production_precursor_evolution.png", dpi=140)
    plt.close()
    print("Saved: production_precursor_evolution.png")


def fig_temperatures(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fam, d in data.items():
        axes[0].plot(d["t"], d["traj"][:, 7], color=COLORS[fam], lw=1.5,
                     label=f"{fam}")
        axes[1].plot(d["t"], d["traj"][:, 8], color=COLORS[fam], lw=1.5,
                     label=f"{fam}")
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("T_f [K]")
    axes[0].set_title("Fuel temperature — one per family")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("T_c [K]")
    axes[1].set_title("Coolant temperature — one per family")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "production_temperatures.png", dpi=140)
    plt.close()
    print("Saved: production_temperatures.png")


def fig_runtime_distribution(bench_rows):
    """Histogram + per-family boxplot of per-scenario wall time."""
    all_walls = [r["wall_time_s"] for r in bench_rows]
    by_fam = defaultdict(list)
    for r in bench_rows:
        by_fam[r["family"]].append(r["wall_time_s"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram (log x-axis)
    axes[0].hist(all_walls, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("wall time [s]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Runtime distribution (n={len(all_walls)})")
    axes[0].axvline(np.median(all_walls), color='r', ls='--',
                    label=f"median = {np.median(all_walls)*1000:.1f} ms")
    axes[0].axvline(np.mean(all_walls), color='g', ls='--',
                    label=f"mean = {np.mean(all_walls)*1000:.1f} ms")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Per-family boxplot
    fams = ["step", "ramp", "sinusoidal", "grf"]
    data = [by_fam[f] for f in fams]
    bp = axes[1].boxplot(data, labels=fams, patch_artist=True, showfliers=True)
    for patch, fam in zip(bp['boxes'], fams):
        patch.set_facecolor(COLORS[fam])
        patch.set_alpha(0.6)
    axes[1].set_ylabel("wall time [s]")
    axes[1].set_title("Per-family runtime distribution")
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGS / "production_runtime_distribution.png", dpi=140)
    plt.close()
    print("Saved: production_runtime_distribution.png")


def fig_physical_bounds(phys_rows):
    """Per-family state bounds — 4 subplots (n, C_total, Tf, Tc)."""
    fams = ["step", "ramp", "sinusoidal", "grf"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def get_bounds(fam, col_min, col_max):
        rs = [r for r in phys_rows if r["family"] == fam]
        return min(float(r[col_min]) for r in rs), max(float(r[col_max]) for r in rs)

    metrics = [
        ("n", "n_min", "n_max", "Neutron density"),
        ("C_total", "C_min", "C_max", "Total precursors"),
        ("Tf", "Tf_min", "Tf_max", "Fuel temperature [K]"),
        ("Tc", "Tc_min", "Tc_max", "Coolant temperature [K]"),
    ]
    for ax, (label, cmin, cmax, title) in zip(axes.flat, metrics):
        mins = []
        maxs = []
        for fam in fams:
            lo, hi = get_bounds(fam, cmin, cmax)
            mins.append(lo)
            maxs.append(hi)
        x = np.arange(len(fams))
        ax.bar(x, [m - mn for m, mn in zip(maxs, mins)], bottom=mins,
               color=[COLORS[f] for f in fams], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(fams)
        ax.set_ylabel(title)
        ax.set_title(f"{title} — per-family range")
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGS / "production_physical_bounds.png", dpi=140)
    plt.close()
    print("Saved: production_physical_bounds.png")


def main():
    kin = KineticsParameters.from_yaml(load_config("reactor_parameters"))
    data = load_rep()

    # Load data from prior blocks
    with (R / "physical_validation_full.csv").open() as f:
        phys_rows = list(csv.DictReader(f))
    with (R / "bdf_runtime_benchmark_full.csv").open() as f:
        bench_rows = list(csv.DictReader(f))
        bench_rows = [{**r, "wall_time_s": float(r["wall_time_s"])}
                      for r in bench_rows]

    # Generate all figures
    print("Generating figures...")
    fig_composite_inputs(data, kin)
    fig_composite_responses(data)
    fig_neutron_density(data)
    fig_precursors(data)
    fig_temperatures(data)
    fig_runtime_distribution(bench_rows)
    fig_physical_bounds(phys_rows)

    # --- Dataset summary JSON
    fam_counts = defaultdict(int)
    for r in phys_rows:
        fam_counts[r["family"]] += 1

    fam_bounds = {}
    for fam in ["step", "ramp", "sinusoidal", "grf"]:
        rs = [r for r in phys_rows if r["family"] == fam]
        fam_bounds[fam] = {
            "n": {"min": min(float(r["n_min"]) for r in rs),
                  "max": max(float(r["n_max"]) for r in rs)},
            "C": {"min": min(float(r["C_min"]) for r in rs),
                  "max": max(float(r["C_max"]) for r in rs)},
            "Tf": {"min": min(float(r["Tf_min"]) for r in rs),
                   "max": max(float(r["Tf_max"]) for r in rs)},
            "Tc": {"min": min(float(r["Tc_min"]) for r in rs),
                   "max": max(float(r["Tc_max"]) for r in rs)},
        }

    fam_runtime = {}
    for fam in ["step", "ramp", "sinusoidal", "grf"]:
        walls = [r["wall_time_s"] for r in bench_rows if r["family"] == fam]
        fam_runtime[fam] = {
            "n": len(walls),
            "mean_s": float(np.mean(walls)),
            "median_s": float(np.median(walls)),
            "total_s": float(np.sum(walls)),
        }

    all_walls = [r["wall_time_s"] for r in bench_rows]
    dataset_summary = {
        "dataset_version": "step_1_3_production_2000",
        "phase": "1.3",
        "timestamp_utc": "2026-10-30",
        "n_scenarios": 2000,
        "family_counts": dict(fam_counts),
        "state_vector_dim": 9,
        "state_vector_order": STATE_NAMES,
        "t_grid_points": 200,
        "t_end_s": 10.0,
        "config_versions": {"reactor_parameters": "v3",
                            "thermal_hydraulics": "v2"},
        "solver": {
            "method": "BDF",
            "rtol": 1e-8,
            "atol": 1e-10,
        },
        "family_bounds": fam_bounds,
        "family_runtime": fam_runtime,
        "global_runtime": {
            "total_wall_s": float(np.sum(all_walls)),
            "mean_s": float(np.mean(all_walls)),
            "median_s": float(np.median(all_walls)),
            "min_s": float(np.min(all_walls)),
            "max_s": float(np.max(all_walls)),
            "p90_s": float(np.percentile(all_walls, 90)),
            "p95_s": float(np.percentile(all_walls, 95)),
        },
        "physical_validation": {
            "n_validated": len(phys_rows),
            "n_passed": sum(1 for r in phys_rows if r["passed"] == "True"),
            "all_passed": all(r["passed"] == "True" for r in phys_rows),
        },
        "cross_validation": {
            "radau_sample_size": 200,
            "radau_all_passed": True,
            "tolerance_sample_size": 40,
            "tolerance_all_passed": True,
        },
    }
    (R / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))
    print("Saved: dataset_summary.json")

    # --- Manuscript Table 3 data
    table3_rows = []
    for fam in ["step", "ramp", "sinusoidal", "grf"]:
        b = fam_bounds[fam]
        rt = fam_runtime[fam]
        table3_rows.append({
            "family": fam,
            "n_scenarios": fam_counts[fam],
            "n_min": b["n"]["min"],
            "n_max": b["n"]["max"],
            "Tf_min_K": b["Tf"]["min"],
            "Tf_max_K": b["Tf"]["max"],
            "Tc_min_K": b["Tc"]["min"],
            "Tc_max_K": b["Tc"]["max"],
            "runtime_mean_s": rt["mean_s"],
            "runtime_median_s": rt["median_s"],
            "runtime_total_s": rt["total_s"],
        })
    table3_rows.append({
        "family": "ALL",
        "n_scenarios": sum(fam_counts.values()),
        "n_min": min(fam_bounds[f]["n"]["min"] for f in fam_bounds),
        "n_max": max(fam_bounds[f]["n"]["max"] for f in fam_bounds),
        "Tf_min_K": min(fam_bounds[f]["Tf"]["min"] for f in fam_bounds),
        "Tf_max_K": max(fam_bounds[f]["Tf"]["max"] for f in fam_bounds),
        "Tc_min_K": min(fam_bounds[f]["Tc"]["min"] for f in fam_bounds),
        "Tc_max_K": max(fam_bounds[f]["Tc"]["max"] for f in fam_bounds),
        "runtime_mean_s": float(np.mean(all_walls)),
        "runtime_median_s": float(np.median(all_walls)),
        "runtime_total_s": float(np.sum(all_walls)),
    })
    with (R / "manuscript_table_3_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table3_rows[0].keys()))
        w.writeheader()
        w.writerows(table3_rows)
    print("Saved: manuscript_table_3_data.csv")

    # Block E summary
    figs = sorted(FIGS.glob("production_*.png"))
    summary = {
        "block": "E (Step 1.3)",
        "n_figures": len(figs),
        "figures": [f.name for f in figs],
        "dataset_summary_written": True,
        "manuscript_table_3_written": True,
    }
    (R / "block_E_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
