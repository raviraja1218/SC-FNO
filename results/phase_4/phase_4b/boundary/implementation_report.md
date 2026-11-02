# Step 4B.3 — Unified Latent Padding Implementation Report

**Date:** 2026-11-02
**Status:** COMPLETE — 12/12 unit tests pass
**Predecessor:** Step 4B.2 fully closed at `9d0e0da`

## Overview

`src/models/boundary_aware_fno.py` provides `UnifiedLatentFNO`, a wrapper that
makes the data path and physics path consume the identical padded FNO latent
representation. This fixes the Phase 3 architectural flaw where the physics
path bypassed padding (used raw FNO on unpadded input).

`src/training/unified_latent.py` provides the canonical training-loop pattern
that guarantees bit-identity between data and physics paths.

## Phase 3 Architectural Flaw (Fixed)

**Phase 3 code** (`src/physics_informed/derivatives.py::_fno_latent`):

    is_wrapped = hasattr(self.fno, "pad_size") and hasattr(self.fno, "fno")
    raw_fno = self.fno.fno if is_wrapped else self.fno
    hook = raw_fno.projection.register_forward_hook(_capture_pre_projection)
    _ = raw_fno(rho_ext)   # ← raw forward on UNPADDED input

**Consequence:** the data path used padded FNO (240 points through FFT), while
the physics path used raw FNO (200 points through FFT). The two latents were
different tensors — even when the wrapper nominally had padding.

**Effect on P4-B:** padding affected only the data path. The physics residual
constraint saw no padding benefit. The intervention was diluted.

## Phase 4B.3 Architecture (Fixed)

**UnifiedLatentFNO** guarantees ONE forward pass per batch that produces ONE
latent, cropped to 200 physical points, used by BOTH data and physics paths.

    # Canonical training loop:
    latent = model.fno.get_latent(rho_ext)      # [B, 32, 200] — ONCE
    u_data = decoder(latent, t_data)             # [B, 9, 200]
    u_phys = decoder(latent, t_q)                # [B, 9, 50]

**Bit-identity is guaranteed by construction** — there is only one latent.
No second forward pass; no risk of FP non-determinism between paths.

## Design Details

### Padding

- **Size:** 20 timesteps each side (10% of 200)
- **Mode:** `reflect` (torch.nn.functional.pad with mode='reflect')
- **Padded length:** 240
- **Rationale:** reflect preserves continuity of the function and its first
  derivative at the boundary, reducing spectral leakage in the FFT

### Cropping

- Latent captured pre-projection: `[B, 32, 240]`
- Crop to `[B, 32, 200]` via `latent_padded[:, :, pad_size:-pad_size]`
- Position `i` in the cropped latent corresponds to physical time `t_i = i · Δt`
  where `Δt = 10/199 ≈ 0.05025 s` (matching Phase 1 grid)

### Physical Domain Constraint

- Physics queries remain in **`[0.5, 9.5]`** (subset of `[0, 10]`)
- **Padding is spectral context, NOT artificial physical observations**
- The 40 padded positions are never exposed to the decoder
- Verified by test T6: latent has exactly 200 points after crop

### Query Sampling (Frozen from 4B.0)

| Region | Count | Range |
|--------|-------|-------|
| Near-start | 4 | [0.5, 1.5] s |
| Middle uniform | 42 | [1.5, 8.5] s |
| Near-end | 4 | [8.5, 9.5] s |
| **Total** | **50** | in [0.5, 9.5] ⊂ [0, 10] |

### Causal Decoder Preserved

- `CoordinateDecoder` unchanged from Phase 3
- No attention mask added
- Decoder attends over the latent, which is causal by construction
- Padding affects the FFT input, not the decoder's temporal attention
- Verified by test T5

## Training-Loop Contract

For P4-B and P4-D, the training loop must:

1. Call `latent = model.fno.get_latent(rho_ext)` ONCE per batch
2. Use `decoder(latent, t_data)` for data loss
3. Use `decoder(latent, t_q)` for physics residual
4. Pass `latent` to any gradient computations (PCGrad, AdaptiveLambda)
5. Never call `model.fno(rho_ext)` and `model.fno.get_latent(rho_ext)`
   as separate forward passes in the same step

**Verified by test T10 (single forward call) and T11 (training-loop contract).**

## Interaction with Other Interventions

### With PCGrad (P4-D)

PCGrad operates on **gradients**, not latents. Unified latent gives PCGrad
consistent gradient structure from both paths. No algorithmic change needed.

### With AdaptiveLambda (P4-C, P4-D)

AdaptiveLambda operates on **cosines**, not latents. Unified latent produces
cleaner cosine signal (both paths see same features). No algorithmic change
needed.

### With P4-A and P4-C (no unified padding)

These variants do NOT use `UnifiedLatentFNO`. They use Phase 3 architecture
(raw FNO). `unified_latent_forward` falls back to raw FNO for these models.

## Unit Tests (12)

| Test | Purpose |
|------|---------|
| T1 | Padding shape (200 → 240 → 200) |
| T2 | **Latent identity (bit-exact `torch.equal`)** |
| T3 | Cropping matches manual `[:, :, 20:220]` |
| T4 | Query bounds in [0.5, 9.5] ⊂ [0, 10] |
| T5 | Causal decoder unchanged (no attention mask) |
| T6 | Padding is not physical (latent has 200 points, not 240) |
| T7 | Data + physics shapes correct |
| T8 | Boundary ratio diagnostic (measured, not pass/fail) |
| T9 | Numerical stability with extreme inputs |
| T10 | **Single forward call per batch** |
| T11 | Training-loop contract (one latent, two decoders) |
| T12 | Reproducibility (bit-exact across calls) |

**Result: 12/12 pass.**

## Test Development Note

One test helper bug was found and fixed during development:
- The initial `FakeFNO` used `nn.Linear` with axis swapping, passing
  `[B, N, 32]` to the projection. Real `neuralop.FNO` passes `[B, 32, N]`.
- Fixed by switching to `nn.Conv1d` so the hook captures `[B, C, N]` correctly.
- This was a **test fake bug**, not an implementation bug. The implementation
  correctly assumes the real FNO's channel-first convention.

## Files

- `src/models/boundary_aware_fno.py` — UnifiedLatentFNO implementation
- `src/training/unified_latent.py` — training-loop helper
- `tests/phase_4/phase_4b/test_unified_latent.py` — 12 unit tests
- `results/phase_4/phase_4b/boundary/unit_test_results.log` — pytest output
- `results/phase_4/phase_4b/boundary/boundary_ratio_measurement.json` — T8 diagnostic
- `results/phase_4/phase_4b/boundary/unified_latent_verification.json` — gate status
