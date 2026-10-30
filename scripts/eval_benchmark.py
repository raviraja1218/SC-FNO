"""Step 2.4.F — Inference benchmark with warm-up + sync + repeats."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from neuralop.models import FNO

PROJECT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = PROJECT / "models/baseline"
OUT = PROJECT / "results/phase_2/evaluation"


def build_fno():
    return FNO(n_modes=(16,), hidden_channels=32, in_channels=1, out_channels=9,
               n_layers=4, positional_embedding=None, domain_padding=None)


def main():
    ckpt = torch.load(MODELS / "fno_best.pt", map_location=DEVICE, weights_only=False)
    model = build_fno().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = {}
    # BDF baseline from Phase 1
    BDF_MEAN_MS = 350.3
    BDF_MEDIAN_MS = 35.3

    for label, B in [("B1", 1), ("B16", 16), ("B200", 200)]:
        x = torch.randn(B, 1, 200, device=DEVICE)
        # warm-up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        torch.cuda.synchronize()

        times = []
        n_repeats = 20 if B <= 16 else 5   # B200 is heavier; 5 repeats enough
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times = np.array(times) * 1000  # ms
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6

        results[label] = {
            "batch_size": B,
            "warmup": 5,
            "repeats": n_repeats,
            "sync": "before_and_after_each_repeat",
            "timer": "time.perf_counter",
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "peak_alloc_MB": float(peak_alloc),
            "peak_reserved_MB": float(peak_reserved),
        }
        print(f"{label}: median={results[label]['median_ms']:.2f} ms  "
              f"mean={results[label]['mean_ms']:.2f} ms  VRAM={peak_alloc:.1f} MB")

    # Speedup vs BDF baseline (per-transient, median-vs-median primary)
    per_scen_median_B16 = results["B16"]["median_ms"] / 16.0
    per_scen_median_B200 = results["B200"]["median_ms"] / 200.0
    speedup = {
        "BDF_mean_ms": BDF_MEAN_MS,
        "BDF_median_ms": BDF_MEDIAN_MS,
        "FNO_per_scenario_ms_B16": per_scen_median_B16,
        "FNO_per_scenario_ms_B200": per_scen_median_B200,
        "speedup_vs_BDF_mean_using_B16": BDF_MEAN_MS / per_scen_median_B16,
        "speedup_vs_BDF_median_using_B16": BDF_MEDIAN_MS / per_scen_median_B16,
        "speedup_vs_BDF_mean_using_B200": BDF_MEAN_MS / per_scen_median_B200,
        "speedup_vs_BDF_median_using_B200": BDF_MEDIAN_MS / per_scen_median_B200,
        "primary_metric": "speedup_vs_BDF_median_using_B200",
    }
    results["speedup_vs_BDF"] = speedup
    json.dump(results, open(OUT / "inference_benchmark.json", "w"), indent=2)

    print(f"\nSpeedup (median vs median, B200): {speedup['speedup_vs_BDF_median_using_B200']:.1f}x")
    print(f"Speedup (mean vs median, B200):   {speedup['speedup_vs_BDF_mean_using_B200']:.1f}x")


if __name__ == "__main__":
    main()
