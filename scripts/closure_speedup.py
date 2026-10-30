"""Step 2.5.B — Three labeled speedup quantities."""
import json
from pathlib import Path

OUT = Path("results/phase_2/closure")
bench = json.load(open("results/phase_2/evaluation/inference_benchmark.json"))

bdf_med = bench["speedup_vs_BDF"]["BDF_median_ms"]
bdf_mean = bench["speedup_vs_BDF"]["BDF_mean_ms"]
b1_med = bench["B1"]["median_ms"]
b200_med = bench["B200"]["median_ms"]

single_scen_speedup = bdf_med / b1_med
throughput_sps = 200 / (b200_med / 1000.0)
per_scen_ms_b200 = b200_med / 200.0
batch_amortized_speedup = bdf_med / per_scen_ms_b200

out = {
    "single_scenario_latency_speedup": {
        "bdf_median_ms": bdf_med,
        "fno_B1_median_ms": b1_med,
        "speedup_x": round(single_scen_speedup, 2),
        "label": "single_scenario_latency",
        "interpretation": (
            f"Per-scenario direct latency: BDF {bdf_med:.2f} ms vs FNO B1 {b1_med:.3f} ms "
            f"→ {single_scen_speedup:.2f}×."
        ),
    },
    "batch_throughput": {
        "fno_B200_total_median_ms": b200_med,
        "scenarios_per_batch": 200,
        "scenarios_per_second": round(throughput_sps, 0),
        "label": "batch_throughput",
        "interpretation": (
            f"At batch 200: ~{throughput_sps:.0f} scenarios/sec on RTX 4050."
        ),
    },
    "batch_amortized_vs_bdf": {
        "bdf_median_ms_per_scenario": bdf_med,
        "fno_per_scenario_ms_at_B200": round(per_scen_ms_b200, 6),
        "speedup_x": round(batch_amortized_speedup, 1),
        "label": "batch_amortized",
        "interpretation": (
            f"Amortizing B200 across 200 scenarios gives {per_scen_ms_b200:.6f} ms/scenario "
            f"→ {batch_amortized_speedup:.1f}× vs BDF. Throughput ratio, not latency."
        ),
    },
    "bdf_mean_for_reference_ms": bdf_mean,
    "primary_for_manuscript": "single_scenario_latency_speedup",
    "note": "The ~2274× figure MUST always be labeled 'batch-amortized', never 'single-scenario latency'.",
}

json.dump(out, open(OUT / "speedup_decomposition.json", "w"), indent=2)
print(json.dumps(out, indent=2))
