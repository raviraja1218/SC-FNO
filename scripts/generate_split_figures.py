"""Step 1.4 Block E — 7 manuscript figures (Figures 3–9).

All figures generated from actual split data.
Output: reports/figures/phase_1/
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

R = Path('results/phase_1/splitting')
RAW = Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"
FIGS = Path('reports/figures/phase_1')
FIGS.mkdir(parents=True, exist_ok=True)

STATE_NAMES = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]
FAMILIES = ["step", "ramp", "sinusoidal", "grf"]
COLORS = {"step": "C0", "ramp": "C1", "sinusoidal": "C2", "grf": "C3"}


def load_split_ids():
    def _load(p):
        with p.open() as f:
            return [r["scenario_id"] for r in csv.DictReader(f)]
    return {
        "train": _load(R / "train_ids.csv"),
        "val": _load(R / "validation_ids.csv"),
        "test": _load(R / "test_ids.csv"),
    }


def load_manifest_family():
    with (R.parent / "generation" / "scenario_manifest_2000.csv").open() as f:
        return {r["scenario_id"]: r["family"] for r in csv.DictReader(f)}


def load_scenario(sid):
    d = RAW / sid
    return {
        "t": np.load(d / "time.npy"),
        "forcing": np.load(d / "forcing.npy"),
        "traj": np.load(d / "trajectory.npy"),
    }


def pick_representatives(split_ids, family_map):
    """One representative per family from train split (deterministic: first occurrence in sorted order)."""
    by_fam = defaultdict(list)
    for sid in split_ids["train"]:
        by_fam[family_map[sid]].append(sid)
    reps = {}
    for fam in FAMILIES:
        ids = sorted(by_fam[fam])
        reps[fam] = ids[len(ids) // 2]  # median index for reproducibility
    return reps


# ---- Figure 3: five reactivity profiles (4 families + 1 extra)
def figure_03(reps, family_map):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for fam in FAMILIES:
        sid = reps[fam]
        s = load_scenario(sid)
        ax.plot(s["t"], s["forcing"], color=COLORS[fam], lw=1.5,
                label=f"{fam} ({sid})")
    # 5th: another GRF with different length scale
    extra_ids = sorted([sid for sid in family_map if family_map[sid] == "grf"])
    extra_sid = extra_ids[-1]  # last GRF
    s = load_scenario(extra_sid)
    ax.plot(s["t"], s["forcing"], color="C4", lw=1.2, ls="--",
            label=f"grf extra ({extra_sid})", alpha=0.7)

    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\rho_{ext}(t)$")
    ax.set_title("Figure 3: Five representative external reactivity profiles")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "figure_03_reactivity_profiles.png", dpi=140)
    plt.close()
    print("Saved: figure_03_reactivity_profiles.png")


# ---- Figure 4: neutron density (trajectories + split distribution)
def figure_04(reps, split_ids, family_map):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: representative trajectories
    ax = axes[0]
    for fam in FAMILIES:
        sid = reps[fam]
        s = load_scenario(sid)
        ax.plot(s["t"], s["traj"][:, 0], color=COLORS[fam], lw=1.5,
                label=f"{fam} ({sid})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("n(t) / n(0)")
    ax.set_title("Neutron density — one per family (train)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: split distribution (n at t_end)
    ax = axes[1]
    split_colors = {"train": "C0", "val": "C7", "test": "C3"}
    data_by_split = {}
    for split_name, ids in split_ids.items():
        n_end_vals = []
        for sid in ids:
            s = load_scenario(sid)
            n_end_vals.append(float(s["traj"][-1, 0]))
        data_by_split[split_name] = n_end_vals
    bp = ax.boxplot([data_by_split["train"], data_by_split["val"],
                     data_by_split["test"]],
                    labels=["train", "val", "test"],
                    patch_artist=True, showfliers=True)
    for patch, split_name in zip(bp['boxes'], ["train", "val", "test"]):
        patch.set_facecolor(split_colors[split_name])
        patch.set_alpha(0.6)
    ax.set_ylabel("n(t=10s) / n(0)")
    ax.set_title("Neutron density distribution by split")
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGS / "figure_04_neutron_density.png", dpi=140)
    plt.close()
    print("Saved: figure_04_neutron_density.png")


# ---- Figure 5: six-precursor evolution (representative scenario)
def figure_05(reps):
    # Pick sinusoidal representative — richest precursor dynamics
    sid = reps["sinusoidal"]
    s = load_scenario(sid)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i in range(6):
        ax.plot(s["t"], s["traj"][:, 1 + i], lw=1.4, label=f"C{i+1}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$C_i(t)$")
    ax.set_title(f"Figure 5: Six-group precursor evolution ({sid})")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "figure_05_precursor_evolution.png", dpi=140)
    plt.close()
    print("Saved: figure_05_precursor_evolution.png")


# ---- Figure 6: fuel temperature (trajectories + split coverage)
def figure_06(reps, split_ids):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: representative trajectories
    ax = axes[0]
    for fam in FAMILIES:
        sid = reps[fam]
        s = load_scenario(sid)
        ax.plot(s["t"], s["traj"][:, 7], color=COLORS[fam], lw=1.5,
                label=f"{fam} ({sid})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T_f [K]")
    ax.set_title("Fuel temperature — one per family (train)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: split distribution
    ax = axes[1]
    split_colors = {"train": "C0", "val": "C7", "test": "C3"}
    data_by_split = {}
    for split_name, ids in split_ids.items():
        tf_end = []
        for sid in ids:
            s = load_scenario(sid)
            tf_end.append(float(s["traj"][-1, 7]))
        data_by_split[split_name] = tf_end
    bp = ax.boxplot([data_by_split["train"], data_by_split["val"],
                     data_by_split["test"]],
                    labels=["train", "val", "test"],
                    patch_artist=True, showfliers=True)
    for patch, split_name in zip(bp['boxes'], ["train", "val", "test"]):
        patch.set_facecolor(split_colors[split_name])
        patch.set_alpha(0.6)
    ax.set_ylabel("T_f(t=10s) [K]")
    ax.set_title("Fuel temperature distribution by split")
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGS / "figure_06_fuel_temperature.png", dpi=140)
    plt.close()
    print("Saved: figure_06_fuel_temperature.png")


# ---- Figure 7: coolant temperature (trajectories + split coverage)
def figure_07(reps, split_ids):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: representative trajectories
    ax = axes[0]
    for fam in FAMILIES:
        sid = reps[fam]
        s = load_scenario(sid)
        ax.plot(s["t"], s["traj"][:, 8], color=COLORS[fam], lw=1.5,
                label=f"{fam} ({sid})")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T_c [K]")
    ax.set_title("Coolant temperature — one per family (train)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: split distribution
    ax = axes[1]
    split_colors = {"train": "C0", "val": "C7", "test": "C3"}
    data_by_split = {}
    for split_name, ids in split_ids.items():
        tc_end = []
        for sid in ids:
            s = load_scenario(sid)
            tc_end.append(float(s["traj"][-1, 8]))
        data_by_split[split_name] = tc_end
    bp = ax.boxplot([data_by_split["train"], data_by_split["val"],
                     data_by_split["test"]],
                    labels=["train", "val", "test"],
                    patch_artist=True, showfliers=True)
    for patch, split_name in zip(bp['boxes'], ["train", "val", "test"]):
        patch.set_facecolor(split_colors[split_name])
        patch.set_alpha(0.6)
    ax.set_ylabel("T_c(t=10s) [K]")
    ax.set_title("Coolant temperature distribution by split")
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGS / "figure_07_coolant_temperature.png", dpi=140)
    plt.close()
    print("Saved: figure_07_coolant_temperature.png")


# ---- Figure 8: state variable distributions (2x5 panel)
def figure_08(reps):
    # Use the 4 reps to show distributions of 9 states (plus forcing)
    # We show box plots across all 2000 scenarios per state
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    flat_axes = axes.ravel()

    # Load extrema for state distributions
    with (R / "scenario_extrema.csv").open() as f:
        extrema = list(csv.DictReader(f))

    for j, name in enumerate(STATE_NAMES):
        ax = flat_axes[j]
        # Distribution of per-scenario mean
        means = [float(r[f"{name}_mean"]) for r in extrema]
        ax.hist(means, bins=40, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_title(f"{name} (per-scenario mean)", fontsize=10)
        ax.grid(alpha=0.3)

    # 10th subplot: forcing amplitude distribution
    ax = flat_axes[9]
    forcing_amps = []
    for r in extrema:
        sid = r["scenario_id"]
        s = load_scenario(sid)
        forcing_amps.append(float(np.mean(s["forcing"])))
    ax.hist(forcing_amps, bins=40, color="orange", alpha=0.7, edgecolor="black")
    ax.set_title("rho_ext (per-scenario mean)", fontsize=10)
    ax.grid(alpha=0.3)

    for ax in flat_axes:
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.suptitle("Figure 8: State-variable distributions (per-scenario mean across all 2,000)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGS / "figure_08_state_distributions.png", dpi=140)
    plt.close()
    print("Saved: figure_08_state_distributions.png")


# ---- Figure 9: dataset split distribution
def figure_09(split_ids, family_map):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: split totals
    ax = axes[0]
    split_totals = {s: len(ids) for s, ids in split_ids.items()}
    x = np.arange(3)
    vals = [split_totals["train"], split_totals["val"], split_totals["test"]]
    bars = ax.bar(x, vals, color=["C0", "C7", "C3"])
    ax.set_xticks(x)
    ax.set_xticklabels(["train", "validation", "test"])
    ax.set_ylabel("scenario count")
    ax.set_title("Split totals")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(v), ha='center', fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(alpha=0.3, axis='y')

    # Right: per-family × split
    ax = axes[1]
    split_fam_counts = {s: defaultdict(int) for s in ["train", "val", "test"]}
    for split_name, ids in split_ids.items():
        for sid in ids:
            split_fam_counts[split_name][family_map[sid]] += 1

    x = np.arange(len(FAMILIES))
    width = 0.25
    for k, split_name in enumerate(["train", "val", "test"]):
        vals = [split_fam_counts[split_name][f] for f in FAMILIES]
        ax.bar(x + (k - 1) * width, vals, width, label=split_name,
               color=["C0", "C7", "C3"][k])
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES)
    ax.set_ylabel("scenario count")
    ax.set_title("Per-family × split")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle("Figure 9: Dataset split distribution", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGS / "figure_09_dataset_split.png", dpi=140)
    plt.close()
    print("Saved: figure_09_dataset_split.png")


def main():
    split_ids = load_split_ids()
    family_map = load_manifest_family()
    reps = pick_representatives(split_ids, family_map)
    print(f"Representatives: {reps}")

    figure_03(reps, family_map)
    figure_04(reps, split_ids, family_map)
    figure_05(reps)
    figure_06(reps, split_ids)
    figure_07(reps, split_ids)
    figure_08(reps)
    figure_09(split_ids, family_map)

    # Inventory
    figs = sorted(FIGS.glob("figure_*.png"))
    summary = {
        "block": "E (Step 1.4)",
        "n_figures": len(figs),
        "figures": [
            {"name": f.name, "size_bytes": f.stat().st_size}
            for f in figs
        ],
        "representatives": reps,
    }
    (R / "block_E_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"Total figures: {len(figs)}")
    for f in figs:
        print(f"  {f.name}: {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
