"""Step 1.5 Block C — authoritative Phase 1 dataset manifest.

Produces a single manifest consolidating all Phase 1 evidence, so that
downstream phases (2-5) reference one authoritative source of truth for
the frozen dataset, split, and normalization.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 60)
    print("Step 1.5 Block C — Create authoritative Phase 1 dataset manifest")
    print("=" * 60)

    # --- Load all prior evidence
    split_manifest = json.load(open('results/phase_1/splitting/split_manifest.json'))
    aggregate = json.load(open('results/phase_1/splitting/aggregate_checksums.json'))
    normalization = json.load(open('results/phase_1/splitting/normalization_statistics.json'))
    step_13_manifest = json.load(open('results/phase_1/generation/step_1_3_manifest.json'))

    # --- Compute checksums of key evidence files
    def sha(p):
        return sha256_file(p) if Path(p).exists() else None

    evidence = {
        "dataset_checksums": sha('results/phase_1/splitting/dataset_checksums.csv'),
        "aggregate_checksums": sha('results/phase_1/splitting/aggregate_checksums.json'),
        "split_manifest": sha('results/phase_1/splitting/split_manifest.json'),
        "train_ids": sha('results/phase_1/splitting/train_ids.csv'),
        "validation_ids": sha('results/phase_1/splitting/validation_ids.csv'),
        "test_ids": sha('results/phase_1/splitting/test_ids.csv'),
        "normalization_statistics": sha('results/phase_1/splitting/normalization_statistics.json'),
        "normalization_report": sha('results/phase_1/splitting/normalization_report.md'),
        "distribution_report": sha('results/phase_1/splitting/distribution_report.md'),
        "distribution_supplement": sha('results/phase_1/splitting/distribution_supplement.md'),
        "scenario_extrema": sha('results/phase_1/splitting/scenario_extrema.csv'),
        "per_family_fft": sha('results/phase_1/splitting/per_family_fft.json'),
        "integrity_report": sha('results/phase_1/splitting/integrity_report.md'),
        "step_1_3_manifest": sha('results/phase_1/generation/step_1_3_manifest.json'),
        "step_1_4_manifest": sha('results/phase_1/splitting/step_1_4_manifest.json'),
        "aggregate_reconciliation": sha('results/phase_1/reconciliation/aggregate_checksum_reconciliation.md'),
        "reconciliation_report": sha('results/phase_1/reconciliation/step_1_3_1_4_reconciliation.md'),
    }

    # --- Source code checksums (critical modules)
    source_checksums = {
        "src/data/solver.py": sha('src/data/solver.py'),
        "src/data/forcing_profiles.py": sha('src/data/forcing_profiles.py'),
        "src/data/config_loader.py": sha('src/data/config_loader.py'),
        "src/physics/kinetics.py": sha('src/physics/kinetics.py'),
        "src/physics/thermal_hydraulics.py": sha('src/physics/thermal_hydraulics.py'),
        "src/physics/reactivity.py": sha('src/physics/reactivity.py'),
        "configs/reactor_parameters.v3.yaml": sha('configs/reactor_parameters.v3.yaml'),
        "configs/thermal_hydraulics.v2.yaml": sha('configs/thermal_hydraulics.v2.yaml'),
    }

    # --- Figures
    figs_dir = Path('reports/figures/phase_1')
    figures = {}
    for p in sorted(figs_dir.glob('figure_*.png')):
        figures[p.name] = sha256_file(p)

    # --- Assemble manifest
    manifest = {
        "phase": "1",
        "status": "LOCKED",
        "timestamp": datetime.now().isoformat(),
        "purpose": (
            "Authoritative single reference for the frozen Phase 1 dataset. "
            "All downstream phases (2-5) must reference this manifest for "
            "dataset identity, split, normalization, and solver configuration."
        ),

        "dataset": {
            "n_scenarios": aggregate["n_scenarios"],
            "production_dir": str(Path.home() / "Research" / "Datasets" / "SC-FNO" / "raw" / "scenarios_2000"),
            "aggregate_sha256": aggregate["dataset_aggregate_sha256"],
            "aggregate_reconciliation_status": "INDEPENDENTLY_VERIFIED",
            "family_counts": aggregate["family_counts"],
            "state_vector": ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"],
            "state_vector_dim": 9,
            "t_grid_points": 200,
            "t_end_s": 10.0,
        },

        "split": {
            "seed": split_manifest["split_seed"],
            "strategy": split_manifest["split_strategy"],
            "counts": {
                "train": aggregate["n_train"],
                "val": aggregate["n_val"],
                "test": aggregate["n_test"],
                "total": aggregate["n_train"] + aggregate["n_val"] + aggregate["n_test"],
            },
            "per_family": {
                "train": 400, "val": 50, "test": 50,
            },
            "train_ids_sha256": evidence["train_ids"],
            "validation_ids_sha256": evidence["validation_ids"],
            "test_ids_sha256": evidence["test_ids"],
        },

        "normalization": {
            "source": normalization["source"],
            "method": normalization["method"]["global"],
            "n_variables": len(normalization["variables"]),
            "variables": normalization["variables"],
            "near_zero_variance": normalization["near_zero_variance_variables"],
            "statistics_sha256": evidence["normalization_statistics"],
            "n_training_scenarios": normalization["n_training_scenarios"],
        },

        "config_versions": {
            "reactor_parameters": "v3",
            "thermal_hydraulics": "v2",
        },

        "solver": {
            "method": "BDF",
            "rtol": 1e-8,
            "atol": 1e-10,
            "cross_check_method": "Radau",
            "cross_check_sample_size": 200,
        },

        "source_checksums": source_checksums,

        "steps": {
            "1.1": {
                "lock_commit": "e5b5628",
                "corrected_in": "959451b",
                "description": "Solver verification: 4 analytical tests (T1-T4) + T5 regression",
            },
            "1.2": {
                "lock_commit": "7049f3e",
                "description": "100-scenario pilot: BDF + Radau cross-check + tolerance sensitivity",
            },
            "1.3": {
                "lock_commit": "00d2819",
                "description": "2,000-scenario production dataset generated",
            },
            "1.4": {
                "lock_commit": "4306ce5",
                "description": "Dataset validation, split, normalization, distribution analysis",
            },
            "1.5": {
                "lock_commit": "(this commit)",
                "description": "Final dataset lock, Gate 1 verification, Phase 1 closure",
            },
        },

        "defects": [
            {
                "id": "SC-FNO-P1-DEFECT-002",
                "description": "Thermal absolute threshold calibration (1e-6 K → 1e-2 K)",
                "status": "RESOLVED",
                "resolution_commit": "959451b",
            },
            {
                "id": "SC-FNO-P0-DEFECT-003",
                "description": "coupled_rhs feedback bug (C5, C6 interpreted as Tf, Tc)",
                "status": "RESOLVED",
                "resolution_commit": "959451b",
                "regression_test": "tests/test_solver.py::test_T5_negative_step_decreases_power",
            },
            {
                "id": "SC-FNO-P1-DEFECT-004",
                "description": "Block D metric artifacts (cosine on proportional profiles, per-time-point z-score)",
                "status": "RESOLVED",
                "resolution_commit": "8580f83",
                "supplement_commit": "3bc3130",
            },
        ],

        "gate_1": "PASS",

        "evidence_checksums": evidence,

        "figures": figures,

        "manifest_self_reference_note": (
            "This manifest does not include its own SHA-256. The file "
            "phase_1_dataset_manifest.json is explicitly excluded from all "
            "checksum lists because writing its own checksum would change "
            "the file."
        ),
    }

    # --- Write manifest
    out_path = Path('results/phase_1/phase_1_dataset_manifest.json')
    out_path.write_text(json.dumps(manifest, indent=2))

    # --- Verify no self-reference
    manifest_text = out_path.read_text()
    assert 'phase_1_dataset_manifest.json' not in json.dumps(manifest.get("evidence_checksums", {})), \
        "self-reference in evidence_checksums!"
    assert 'phase_1_dataset_manifest.json' not in json.dumps(manifest.get("figures", {})), \
        "self-reference in figures!"
    print(f"\nManifest written: {out_path}")
    print(f"  Manifest SHA-256: {sha256_file(out_path)}")
    print(f"  Evidence files checksummed: {len([v for v in evidence.values() if v])}")
    print(f"  Source files checksummed: {len([v for v in source_checksums.values() if v])}")
    print(f"  Figures checksummed: {len(figures)}")
    print(f"  Steps recorded: {len(manifest['steps'])}")
    print(f"  Defects recorded: {len(manifest['defects'])}")
    print(f"  Gate 1: {manifest['gate_1']}")
    print(f"\n✓ No self-reference")


if __name__ == "__main__":
    main()
