"""Step 3.5.E — BDF speedup benchmark + resolution generalization."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from step3_5_eval_utils import (
    load_norm, load_test_ids, load_test_data, denorm_states,
    build_model_for_variant, predict_data_path, predict_decoder_path,
    rel_l2, STATES,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
norm = load_norm()
test_ids = load_test_ids()[:100]  # speed subset
test_data = load_test_data()

# =========== SPEEDUP BENCHMARK ===========
print("\n=== BDF Speedup Benchmark (100 test scenarios) ===\n")
speedup_results = {}
for v in ["p2", "p_fno", "pi_fno", "p_pi_fno"]:
    model, padding, _ = build_model_for_variant(v, device=device)

    # Warm-up
    d0 = test_data[test_ids[0]]
    rho0 = (d0["forcing"] - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
    for _ in range(3):
        _ = predict_data_path(model, padding, rho0[None, :], device)

    # Single-scenario latency
    torch.cuda.synchronize() if device == "cuda" else None
    times_b1 = []
    for sid in test_ids[:10]:
        d = test_data[sid]
        rho = (d["forcing"] - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        _ = predict_data_path(model, padding, rho[None, :], device)
        torch.cuda.synchronize() if device == "cuda" else None
        times_b1.append(time.perf_counter() - t0)

    # Batch B=100
    rho_batch = np.stack([
        (test_data[sid]["forcing"] - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        for sid in test_ids
    ])
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    _ = predict_data_path(model, padding, rho_batch, device)
    torch.cuda.synchronize() if device == "cuda" else None
    t_batch = time.perf_counter() - t0

    speedup_results[v] = {
        "single_scenario_median_ms": float(np.median(times_b1) * 1000),
        "single_scenario_mean_ms": float(np.mean(times_b1) * 1000),
        "batch_total_ms": float(t_batch * 1000),
        "batch_throughput_scenarios_per_sec": float(100 / t_batch),
        "batch_amortized_per_scenario_ms": float(t_batch * 1000 / 100),
    }
    print(f"{v:>10}: single {np.median(times_b1)*1000:.3f} ms, "
          f"batch throughput {100/t_batch:.0f}/s")

# BDF reference from Phase 1
speedup_results["_bdf_reference"] = {
    "single_scenario_median_ms": 35.3,
    "single_scenario_mean_ms": 350.3,
    "note": "From Phase 1 - locked baseline",
}
# Compute speedup ratios
for v in ["p2", "p_fno", "pi_fno", "p_pi_fno"]:
    single_ms = speedup_results[v]["single_scenario_median_ms"]
    speedup_results[v]["single_scenario_speedup_vs_bdf"] = 35.3 / single_ms
    batch_amort = speedup_results[v]["batch_amortized_per_scenario_ms"]
    speedup_results[v]["batch_amortized_speedup_vs_bdf"] = 35.3 / batch_amort

Path("results/phase_3/evaluation/speedup_benchmark.json").write_text(
    json.dumps(speedup_results, indent=2)
)
print("\n=== Speedup benchmark saved ===")

# =========== RESOLUTION GENERALIZATION ===========
print("\n=== Resolution Generalization (PI-FNO, best variant) ===\n")
model, padding, _ = build_model_for_variant("pi_fno", device=device)

# Use 10 test scenarios at multiple query resolutions
test_subset = test_ids[:10]
res_results = {}
for Nq in [200, 500, 1000, 5000, 10000]:
    t_query_full = np.linspace(0, 10, Nq)
    # Reference: cubic spline of ground truth at 200 grid to Nq points
    from scipy.interpolate import CubicSpline
    rel_l2_all = []
    for sid in test_subset:
        d = test_data[sid]
        rho = (d["forcing"] - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        u_pred, _ = predict_decoder_path(model, rho[None, :], t_query_full[None, :], device)
        u_pred_phys = denorm_states(torch.tensor(u_pred), norm).numpy()[0]  # [9, Nq]

        # Ground truth at Nq: cubic-spline from 200 grid
        gt_at_200 = d["trajectory"].T  # [9, 200]
        t_200 = d["time"]
        gt_at_Nq = np.stack([
            CubicSpline(t_200, gt_at_200[j])(t_query_full) for j in range(9)
        ])
        rel_l2_all.append(rel_l2(u_pred_phys.flatten(), gt_at_Nq.flatten()))

    res_results[f"Nq_{Nq}"] = {
        "n_query": Nq,
        "rel_l2_mean": float(np.mean(rel_l2_all)),
        "rel_l2_std": float(np.std(rel_l2_all)),
        "note": "Reference is cubic-spline interpolated from 200-point Phase 1 ground truth. "
                "Sub-grid dynamics are limited by Phase 1 temporal resolution (Δt=0.050251s).",
    }
    print(f"  Nq={Nq:>6}: rel_l2 = {np.mean(rel_l2_all):.4e}")

Path("results/phase_3/evaluation/resolution_generalization.json").write_text(
    json.dumps(res_results, indent=2)
)
print("\n=== 3.5.E PASS ===")
