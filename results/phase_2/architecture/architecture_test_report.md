# Step 2.1 — FNO Architecture Verification Report

**Date:** 2026-10-30  
**Phase 1 lock:** `1d9b320` (ancestor verified)  
**Phase 1 aggregate:** `9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6`

## Environment
- Python: 3.10.22
- Torch: 2.5.1+cu121 (CUDA 12.1)
- neuralop: 2.0.0
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6439 MB)
- FFT round-trip error: 1.1920928955078125e-06

## Architecture
- Implementation: `A_neuralop_FNO`
- Total trainable parameters: **50,409**
- Checkpoint size: 0.366 MB

## GPU Compatibility (B=16)
- Forward wall time: 0.250861 s
- Forward+Backward wall time: 0.204783 s
- Forward peak allocated: 21.72 MB
- Forward+Backward peak allocated: 34.55 MB
- Forward+Backward peak reserved: 50.33 MB
- 3 GB target (measured): **MET**

## Input-Channel Audit
- Decision: **PROCEED_SINGLE_INPUT**
- Rationale: All observed initial-condition and reactor parameters are constant across sampled scenarios. Single-input-channel (rho_ext only) assumption is supported.

## Gate 2.1 Checklist
| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | 4 Fourier layers | PASS | 4 |
| 2 | Width 32 | PASS | 32 |
| 3 | 16 Fourier modes | PASS | 16 |
| 4 | Input dim = 1 | PASS | 1 |
| 5 | Output dim = 9 | PASS | 9 |
| 6 | Parameter count measured | PASS | 50,409 |
| 7 | Forward shapes OK (B=1,4,16) | PASS | ['B=1', 'B=4', 'B=16'] |
| 8 | Backward gradients flow (finite + nonzero) | PASS | 40 grads per batch, all finite |
| 9 | CUDA compatibility recorded | PASS | fb peak alloc: 34.55 MB |
| 10 | Input-channel assumption verified | PASS | PROCEED_SINGLE_INPUT |
| 11 | 3 GB VRAM target assessed (measured, not claimed) | PASS | peak fwd+bwd alloc 34.55 MB → MET |

**Gate 2.1: PASS**  
(11/11 checks passed)
