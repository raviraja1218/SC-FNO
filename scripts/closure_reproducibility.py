"""Step 2.5.A — Reproducibility verification (read-only).

Re-inference determinism uses PER-STATE RELATIVE tolerance (1e-4), not
absolute. FP32 CUDA ops (matmul, FFT) are not bit-exact across process
runs; a fixed absolute tolerance is inappropriate when states span
different physical scales (Tf ~ 950 K, C2 ~ 470, n ~ 1.5).
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np
import torch
from neuralop.models import FNO

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/phase_2/closure"
EVAL = PROJECT / "results/phase_2/evaluation"
MODELS = PROJECT / "models/baseline"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"

sys.path.insert(0, str(PROJECT / "src"))
from data.dataset import NORM_PATH, STATE_ORDER

EXPECTED_BEST_SHA = "24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889"
EXPECTED_AGG = "9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6"
REL_TOL = 1e-4  # 0.01% relative — well above FP32 noise, well below any real discrepancy


def build_fno():
    return FNO(n_modes=(16,), hidden_channels=32, in_channels=1, out_channels=9,
               n_layers=4, positional_embedding=None, domain_padding=None)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = {}

    # 1. Architecture reconstructs
    model = build_fno()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checks["architecture_reconstructs"] = True
    checks["param_count_50409"] = (n_params == 50409)
    print(f"  Reconstructed param count: {n_params:,}")

    # 2. fno_best.pt loads + SHA-256
    ckpt_path = MODELS / "fno_best.pt"
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    checks["best_ckpt_sha256_matches"] = (ckpt_sha == EXPECTED_BEST_SHA)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    checks["best_ckpt_loads"] = True
    print(f"  fno_best.pt SHA: {ckpt_sha[:16]}... (expected {EXPECTED_BEST_SHA[:16]}...)")

    # 3. Re-inference with per-state RELATIVE tolerance
    norm = json.load(open(NORM_PATH))["global"]
    means = np.array([norm[v]["mean"] for v in STATE_ORDER])[:, None]
    stds  = np.array([norm[v]["std"]  for v in STATE_ORDER])[:, None]

    import pandas as pd
    test_ids = pd.read_csv(PROJECT / "results/phase_1/splitting/test_ids.csv")["scenario_id"].tolist()[:5]

    per_state_max_rel = {v: 0.0 for v in STATE_ORDER}
    per_scenario = []
    for sid in test_ids:
        forcing = np.load(RAW / sid / "forcing.npy").astype(np.float32)
        forcing_n = (forcing - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        x = torch.from_numpy(forcing_n).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            yhat_n = model(x).cpu().numpy()[0]   # (9,200)
        yhat_p = yhat_n * stds + means
        stored = np.load(EVAL / "test_predictions" / f"{sid}_pred_physical.npy")  # (9,200)

        # Per-state relative diff: max abs diff / state range (robust denominator)
        # Using max(|yhat_p[j]|, |stored[j]|) as scale, or state std — use max-abs scale
        scen_row = {"sid": sid}
        for j, v in enumerate(STATE_ORDER):
            scale = max(float(np.max(np.abs(yhat_p[j]))),
                        float(np.max(np.abs(stored[j]))),
                        1e-8)
            rel = float(np.max(np.abs(yhat_p[j] - stored[j]))) / scale
            per_state_max_rel[v] = max(per_state_max_rel[v], rel)
            scen_row[v] = rel
        per_scenario.append(scen_row)

    # Determinism: every state's max relative diff must be < REL_TOL across all 5 scenarios
    determinism_ok = all(v < REL_TOL for v in per_state_max_rel.values())
    checks["re_inference_determinism"] = determinism_ok
    print(f"  Per-state max relative diffs (across 5 scenarios):")
    for v in STATE_ORDER:
        print(f"    {v:>3}: {per_state_max_rel[v]:.2e}  [{'OK' if per_state_max_rel[v] < REL_TOL else 'EXCEEDS'}]")

    # 4. Phase 1 aggregate unchanged
    m = json.load(open(PROJECT / "results/phase_1/phase_1_dataset_manifest.json"))
    checks["phase_1_aggregate_unchanged"] = (m["dataset"]["aggregate_sha256"] == EXPECTED_AGG)

    # 5. Split SHAs
    split_shas = {
        "train": "16ad4670eee9555aec237c94d2d241e90b74b50ce8ea57eb07efe209fe852834",
        "validation": "19b781283503b1a1cdb313dc68ce153086f18c25043f1ff5123e3cb3e3205bb6",
        "test": "a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2",
    }
    for name, exp in split_shas.items():
        p = PROJECT / f"results/phase_1/splitting/{name}_ids.csv"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        checks[f"{name}_split_sha256"] = (got == exp)

    # 6. Normalization SHA
    norm_sha = hashlib.sha256(Path(NORM_PATH).read_bytes()).hexdigest()
    checks["normalization_sha256"] = (norm_sha == "4e832aa5cdaa940cb2e07bf64c5741cc5124ad40242dd9532da231ff257f6727")

    report = {
        "checks": checks,
        "all_passed": all(checks.values()),
        "tolerance": {"type": "per_state_relative", "threshold": REL_TOL,
                      "denominator": "max(abs(pred), abs(stored)) per state per scenario"},
        "measured": {
            "n_params": n_params,
            "best_ckpt_sha256": ckpt_sha,
            "per_state_max_relative_diff": per_state_max_rel,
            "per_scenario": per_scenario,
            "normalization_sha256": norm_sha,
        },
        "note": (
            "FP32 CUDA matmul/FFT are not bit-exact across process runs. "
            "Absolute tolerance is inappropriate for states with different "
            "physical scales. Relative tolerance 1e-4 confirms re-inference "
            "agrees with stored Step 2.4 predictions to FP32 noise level."
        ),
    }
    json.dump(report, open(OUT / "reproducibility.json", "w"), indent=2)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n2.5.A: {'PASS' if report['all_passed'] else 'FAIL'}")
    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
