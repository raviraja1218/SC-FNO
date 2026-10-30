# Audit Item 5b — Latency Scope Definition

## What the 3.059 ms FNO Single-Scenario Latency Measures

The Phase 2 report quoted 3.059 ms as the B=1 inference latency. Reviewer 2 asked
whether data loading, normalization, GPU transfer, and denormalization are included.
Here is the exact scope from `scripts/eval_benchmark.py`:

## Measurement Scope (from code)

    x = torch.randn(B, 1, 200, device=DEVICE)    # pre-loaded, on GPU
    # ... warm-up loop ...
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)                              # ← timed
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

**What IS measured:**
- Model forward pass on pre-loaded input tensor
- Time from `time.perf_counter()` to `torch.cuda.synchronize()` return

**What IS NOT measured:**
- Disk I/O (loading forcing.npy)
- Normalization (rho_ext - mean) / std
- CPU→GPU transfer of the input
- Any postprocessing on the output
- Denormalization of output to physical units
- Metric computation

## Measured Values by Batch Size

| Batch | Median (ms) | Mean (ms) | Peak VRAM (MB) |
|-------|-------------|-----------|----------------|
| B1 | 3.059 | 3.144 | 9.44 |
| B16 | 3.024 | 3.042 | 12.54 |
| B200 | 3.105 | 3.095 | 51.05 |

The near-identical median times across B=1, B=16, B=200 reflect that the FNO
forward pass is dominated by per-sample FFT and spectral convolution operations;
batching reduces wall time per scenario substantially but doesn't reduce the
single-scenario forward cost meaningfully.

## Estimated End-to-End Overhead (Not Measured)

Based on the actual data loader used in Phase 2.2 tests, one scenario load +
normalize + transfer is on the order of:
- `.npy` load: < 1 ms (SSD)
- Normalization: ~10 μs
- CPU→GPU transfer: ~50–100 μs (for 1 × 1 × 200 tensor)

So an approximate end-to-end single-scenario pipeline is ~3.2–3.5 ms, not 3.059 ms.
This estimate was NOT measured and should not be quoted as a Phase 2 result.

## For the Manuscript

Precise wording:

> "The 3.059 ms single-scenario latency reported in Table X is the **model
> forward pass only**, with the pre-loaded input tensor already on GPU and no
> data loading, normalization, transfer, denormalization, or metric computation
> included. End-to-end inference (load + preprocess + forward + postprocess)
> would be somewhat higher; the reported figure is a model-only characterization.
> BDF baseline times (35.3 ms median, Phase 1) are end-to-end solver times and
> include internal initialization. The 11.54× speedup figure compares these
> directly, so it should be read as 'FNO model forward versus BDF full solve' —
> the true end-to-end speedup may differ."

## Recommendation for Phase 3

If Phase 3 reports speedup again, measure and report both:
- **Model-only forward latency** (for architecture comparison)
- **End-to-end latency** (load + forward + postprocess) for deployment realism

Maintain the three-quantity discipline from Phase 2.5: single-scenario latency,
batch throughput, batch-amortized.
