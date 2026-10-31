# Step 3.2 Architecture Addendum — Shared Nine-State Prediction Path

**Date:** 2026-10-31
**Commit context:** Addendum to `46e4a2c` (Step 3.2 lock)
**Purpose:** Answer reviewer questions about the relationship between the
data-loss prediction path and the physics-loss prediction path.

---

## Question 1: What exact tensor is used for data-loss predictions?

**The full FNO forward** — the same path Phase 2 trained:


This routes through:
- `lifting` → [B, 32, 200]
- 4 × `fno_blocks` → [B, 32, 200]
- `projection` → [B, 9, 200]

The output `[B, 9, 200]` is the 9 predicted states at the fixed 200-point grid,
directly comparable to the Phase 1 ground-truth `trajectory.npy`.

**Parameters receiving gradient from data loss:** ALL 40 FNO parameter tensors
(including projection) + all 14 decoder parameter tensors (when decoder is in
the data path — see Question 4).

## Question 2: What exact tensor is used for physics-loss predictions?

**The decoder path** — via `compute_du_dt` in `derivatives.py`:


The output `[B, 9, Nq]` is the 9 predicted states at **arbitrary** query times,
not limited to the 200-point grid. Physics residuals (from Step 3.1) are
evaluated on these interpolated states and their derivatives.

**Parameters receiving gradient from physics loss:**
- 36 in-path FNO params (lifting + fno_blocks) — ✅ verified 36/36
- 4 projection params — ❌ BYPASSED (see Question 5)
- 13/14 decoder params — ✅ (1 is `head[-1].bias`, zero-init, will train via data loss in Step 3.3)

## Question 3: Do both represent the same nine physical states?

**Yes, semantically.** Both produce the state vector:


in the same units (normalized per Phase 1 training-only z-score).

**Numerically, they will differ** because:
- `u_data` is produced by `projection(z)` at the 200 grid points
- `u_phys` is produced by `decoder(z, t_q)` at arbitrary query times
- `projection` and `decoder` are DIFFERENT parameter sets (different architectures)

At the 200 grid points specifically, `u_data[i]` and `u_phys(t_grid[i])` are
both predictions of the same physical states — but via different mappings.

## Question 4: Do the two output paths share parameters?

**Partial overlap:**

| Parameter set | Data path | Physics path |
|---------------|-----------|--------------|
| `fno.lifting` | ✅ Trained | ✅ Trained |
| `fno.fno_blocks` (4) | ✅ Trained | ✅ Trained |
| `fno.projection` | ✅ Trained | ❌ Bypassed |
| `decoder` | ✅ Trained (when in data path) | ✅ Trained |

**The shared parameters are `lifting` + `fno_blocks`.** Both losses train these
36 tensors. This is the critical shared representation — the physics residual
**does** improve the spectral representation that the data prediction depends on.

**The `projection` layer is exclusively trained by the data loss.** The physics
residual does not reach it because `compute_du_dt` captures the pre-projection
latent `z` and routes it to the decoder.

**Design justification:** The physics residual is defined on **states**, and
requires the model's predictions to satisfy PKE + thermal-hydraulics. If we
routed physics residuals through the projection layer, we'd need to differentiate
through it — and the projection layer is a 2-layer MLP that would need to satisfy
the residuals *at grid points only*, which is the wrong functional form (the
residual is continuous in time, not discrete). The coordinate decoder is the
**correct** place to evaluate continuous-time residuals.

**The shared `lifting + fno_blocks` means physics constraints DO propagate into
the data-prediction pathway** — via the shared spectral representation. This is
the intended physics-informed improvement mechanism.

## Question 5: How are predictions kept consistent between data and physics?

**Three consistency mechanisms:**

### Mechanism 1 — Shared latent representation
Both paths derive from the same `z = lifting(rho_ext) → fno_blocks`. If
`fno_blocks` learns physically meaningful dynamics (via physics loss), the
latent `z` improves. Better `z` → better `projection(z)` = better `u_data`.

### Mechanism 2 — Data loss trains decoder too
In Step 3.3, the data loss will be evaluated on `u_data = projection(z)` at grid
points, **and** (design decision pending) potentially also on `decoder(z, t_grid)`
at grid points. If we train the decoder against ground truth at grid points, the
decoder's mapping becomes consistent with the projection's mapping at those points.

**This is a design decision for Step 3.3** — not frozen yet.

### Mechanism 3 — No direct weight sharing
`projection` and `decoder` are separate parameter sets. They are NOT the same
mapping. Step 3.3 must decide:
- **Option (i):** Train `projection` only via data loss, `decoder` only via physics loss
- **Option (ii):** Train both via data loss + physics loss, letting them converge to consistent mappings

**Step 3.3 must resolve this.** Item C (data-loss gradient to `head[-1].bias`)
is part of this decision.

---

## Summary for reviewer

| Sub-question | Answer |
|--------------|--------|
| Data tensor | `u_data = fno(rho_ext)` → `[B, 9, 200]` |
| Physics tensor | `u_phys = decoder(z, t_q)` → `[B, 9, Nq]`, `z = pre-projection latent` |
| Same states? | Yes — same 9 physical states, same units, same normalization |
| Shared params? | `lifting` + `fno_blocks` (36 tensors) shared; `projection` (4 tensors) data-only; `decoder` (14 tensors) physics-path-only unless Step 3.3 adds it to data path |
| Consistency? | Via shared `z`; Step 3.3 to decide if decoder also trains on data |

**The physics loss reaches the data prediction via the shared `lifting + fno_blocks`
latent representation.** This is the intended physics-informed improvement path.
