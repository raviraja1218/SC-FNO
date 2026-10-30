# VRAM Measurement Methodology

## Definition
All VRAM figures in `results/phase_2/evaluation/inference_benchmark.json` are **`torch.cuda.max_memory_allocated()`** — peak allocated PyTorch tensor memory, not total process GPU usage.

## Method
Per measurement block:
1. `torch.cuda.empty_cache()`
2. `torch.cuda.reset_peak_memory_stats()`
3. Warm-up (5 iterations)
4. `torch.cuda.synchronize()`
5. Timed repeats (n=20 for B1/B16, n=5 for B200)
6. Read `max_memory_allocated()` / `max_memory_reserved()`

## Measured

### Inference (Step 2.4)
| Batch | Peak allocated (MB) | Peak reserved (MB) |
|-------|---------------------|-------------------|
| B1 | 9.44 | — (not recorded) |
| B16 | 12.54 | — (not recorded) |
| B200 | **51.05** | — (not recorded) |

### Training (Step 2.3)
| Phase | Peak allocated (MB) | Peak reserved (MB) |
|-------|---------------------|-------------------|
| Training (any epoch) | **32.58** | **39.85** |

## Target assessment
- Phase 2 VRAM target: **≤ 3,000 MB**
- Training: **MET** (32.58 MB, 92× headroom)
- Inference B200: **MET** (51.05 MB, 59× headroom)

## Caveats
- Values are **PyTorch tensor allocations only**. OS, driver, CUDA context overhead excluded.
- RTX 4050 total VRAM = 6,438 MB. Reported peaks leave >6,300 MB free at B200.
