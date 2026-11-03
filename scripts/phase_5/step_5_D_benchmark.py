"""Step 5.D.D — BDF vs SC-FNO timing comparison."""
import sys, json, time
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.integrate import solve_ivp

from physics.kinetics import equilibrium_precursors
from training.train_sc_fno import get_kin, get_th
from training.adaptive_optimizer import load_phase3_pi_fno_model
from phase_5_corrected.model_v2 import SCFNOModelV2

device = "cuda" if torch.cuda.is_available() else "cpu"
RAW = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
OUT = Path("results/phase_5/step_5_D/benchmark")
OUT.mkdir(parents=True, exist_ok=True)
STATES = ["n","C1","C2","C3","C4","C5","C6","Tf","Tc"]

norm = json.load(open("results/phase_1/splitting/normalization_statistics.json"))["global"]
test_ids = pd.read_csv("results/phase_1/splitting/test_ids.csv")["scenario_id"].tolist()[:30]

kin = get_kin(); th = get_th()
_beta_i = np.asarray(kin.beta_i, dtype=np.float64)
_lambda_i = np.asarray(kin.lambda_i, dtype=np.float64)
n0 = 1.0
C0 = equilibrium_precursors(n0, kin)
T_f0 = 900.0; T_c0 = 583.0
t_grid = np.linspace(0, 10, 200)

def coupled_rhs(t, y, rho_fn):
    n, C1, C2, C3, C4, C5, C6, T_f, T_c = y
    C = np.array([C1, C2, C3, C4, C5, C6])
    rho_ext = rho_fn(t)
    rho_fb = th.alpha_f * (T_f - th.T_f_ref) + th.alpha_c * (T_c - th.T_c_ref)
    rho_total = rho_ext + rho_fb
    dndt = (rho_total - kin.beta_eff) / kin.Lambda * n + np.sum(_lambda_i * C)
    dCdt = _beta_i / kin.Lambda * n - _lambda_i * C
    dTfdt = (th.f_f * th.P0 * n - th.U * th.A * (T_f - T_c)) / (th.M_f * th.c_p_f)
    dTcdt = ((1 - th.f_f) * th.P0 * n + th.U * th.A * (T_f - T_c)
             - 2 * th.m_dot * th.c_p_c * (T_c - th.T_in)) / (th.M_c * th.c_p_c)
    return [dndt, *dCdt, dTfdt, dTcdt]

def time_bdf(fn_arr, n_repeats=3):
    y0 = np.concatenate([[n0], C0, [T_f0, T_c0]])
    rho_fn = lambda t: float(np.interp(t, t_grid, fn_arr))
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        sol = solve_ivp(lambda t, y: coupled_rhs(t, y, rho_fn),
                        [0.0, 10.0], y0, method="BDF",
                        rtol=1e-8, atol=1e-10, t_eval=t_grid)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))

def load_model_5b():
    m3, _ = load_phase3_pi_fno_model("models/sc_fno/phase_3/pi_fno/best_val.pt", device=device)
    raw = m3.fno.fno if hasattr(m3.fno, "fno") else m3.fno
    m = SCFNOModelV2(raw, m3.decoder, pad_size=0).to(device)
    ck = torch.load("models/sc_fno/phase_5/step_5_B/seed_42_best.pt",
                    map_location=device, weights_only=False)
    sd = {k:v for k,v in ck["model_state_dict"].items() if k != "_metadata"}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def time_scfno_model_only(model, fn_norm, n_repeats=5, n_warmup=2):
    xb = torch.tensor(fn_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.forward_data(xb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.forward_data(xb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return float(np.median(times))

def time_scfno_e2e(model, fn_raw, n_repeats=5):
    times = []
    for _ in range(n_repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn_norm = (fn_raw - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        xb = torch.tensor(fn_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_n = model.forward_data(xb).cpu().numpy()[0]
        pred_phys = np.stack([pred_n[j] * norm[k]["std"] + norm[k]["mean"]
                              for j, k in enumerate(STATES)], axis=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))

def main():
    model = load_model_5b()
    print(f"Model loaded on {device}")

    results = {
        "n_scenarios": len(test_ids),
        "device": str(device),
        "bdf_times_s": [],
        "scfno_model_only_times_s": [],
        "scfno_end_to_end_times_s": [],
    }

    for i, sid in enumerate(test_ids):
        d = RAW / sid
        fn_raw = np.load(d / "forcing.npy")
        fn_norm = (fn_raw - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
        bdf_med = time_bdf(fn_raw, n_repeats=3)
        scfno_med = time_scfno_model_only(model, fn_norm, n_repeats=5, n_warmup=2)
        e2e_med = time_scfno_e2e(model, fn_raw, n_repeats=5)
        results["bdf_times_s"].append(bdf_med)
        results["scfno_model_only_times_s"].append(scfno_med)
        results["scfno_end_to_end_times_s"].append(e2e_med)
        if i % 5 == 0:
            print(f"  [{i+1}/{len(test_ids)}] BDF={bdf_med*1000:.1f}ms, "
                  f"SCFNO={scfno_med*1000:.2f}ms, E2E={e2e_med*1000:.2f}ms")

    print("\n=== Batch throughput (B=200) ===")
    fn_all = np.zeros((len(test_ids), 200), dtype=np.float32)
    for i, sid in enumerate(test_ids):
        fn_raw = np.load(RAW / sid / "forcing.npy")
        fn_all[i] = (fn_raw - norm["rho_ext"]["mean"]) / norm["rho_ext"]["std"]
    xb_all = torch.tensor(fn_all, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model.forward_data(xb_all)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    n_reps = 20
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = model.forward_data(xb_all)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    batch_total = (t1 - t0) / n_reps
    throughput = len(test_ids) / batch_total
    print(f"  Batch: {batch_total*1000:.2f} ms for {len(test_ids)} scenarios")
    print(f"  Throughput: {throughput:.0f} scen/sec")

    results["batch_total_s"] = batch_total
    results["batch_throughput_per_s"] = throughput
    results["bdf_median_ms"] = float(np.median(results["bdf_times_s"]) * 1000)
    results["scfno_model_median_ms"] = float(np.median(results["scfno_model_only_times_s"]) * 1000)
    results["scfno_e2e_median_ms"] = float(np.median(results["scfno_end_to_end_times_s"]) * 1000)
    results["speedup_model_only"] = results["bdf_median_ms"] / results["scfno_model_median_ms"]
    results["speedup_e2e"] = results["bdf_median_ms"] / results["scfno_e2e_median_ms"]
    results["peak_vram_MB"] = float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0

    json.dump(results, open(OUT / "benchmark_results.json", "w"), indent=2)

    print("\n" + "="*70)
    print("Benchmark summary")
    print("="*70)
    print(f"  BDF median:           {results['bdf_median_ms']:.2f} ms")
    print(f"  SC-FNO model-only:    {results['scfno_model_median_ms']:.2f} ms")
    print(f"  SC-FNO end-to-end:    {results['scfno_e2e_median_ms']:.2f} ms")
    print(f"  Speedup (model-only): {results['speedup_model_only']:.1f}×")
    print(f"  Speedup (end-to-end): {results['speedup_e2e']:.1f}×")
    print(f"  Batch throughput:     {results['batch_throughput_per_s']:.0f} scen/s")
    print(f"  Peak VRAM:            {results['peak_vram_MB']:.1f} MB")
    print("="*70)

if __name__ == "__main__":
    main()
