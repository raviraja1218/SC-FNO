
---

## R6 Addendum — Reviewer Items 1-2 Resolved (2026-11-01)

### Item 1 — Physics Loss Δ (RESOLVED)

**Reviewer:** "Document whether the two physics values use identical query points, residual scaling, aggregation, and evaluation mode."

**Finding:** Phase 3's training log shows `val_phys_loss = 0.000000` at ALL 67 epochs (0/67 non-zero). Phase 3 **never computed validation physics loss** — only `train_phys_loss`. My 4A.0.R5 `1.68e-3` is the **first measurement** of validation physics loss at epoch 36.

**Comparison is apples-to-oranges:**

| Value | Phase 3 (`8.559691e-4`) | 4A.0.R5 (`1.68e-3`) |
|-------|------------------------|---------------------|
| Field | `train_phys_loss` | `loss_fn.compute_physics_loss()` on validation |
| Data | 1,600 training scenarios | 200 validation scenarios |
| Epoch | 66 (final) | 36 (best-val) |
| λ_phys | 0.0436 | 0.002754 |

**Same function, same query points, same residual scaling, same aggregation — different data + different epoch.**

**No discrepancy exists. Both numbers correct for their definitions.**

**Phase 3 log verification:**

### Item 2 — Optimizer State (RESOLVED)

**Reviewer:** "Initialize a fresh Adam optimizer unless you can establish that a matching optimizer state was saved at epoch 36."

**Finding:** Phase 3 saved NO optimizer state. All checkpoints contain only weights + metadata.

**Verified across all Phase 3 variants:**

**Decision for 4A.1:** Fresh Adam optimizer from epoch-36 weights. Phase 3 hyperparameters (lr=1e-3, betas=(0.9,0.999)) reused for continuity. Fresh CosineAnnealingLR. **No risk of optimizer-state mismatch because no optimizer state exists.**

**Rationale documented:** reproducible from verified weights; Phase 3's Adam trajectory at epoch 66 (if it had been saved) is not aligned with epoch-36 weights anyway.

### Item 3 — λ_phys Transition Policy (PENDING 4A.1 PLAN)

Deferred to Phase 4A.1 plan. Recommended policy: **L-BFGS inherits λ_phys at plateau detection** (do NOT jump to 1.0). Pre-registered before any 4A.1 training.

---

## R9 Addendum — Loader Bug Post-Mortem (2026-11-01)

**Investigation:** After R7/R8, the fixed and "buggy" loaders produced identical
training outputs. Deep diagnostic revealed why.

### Truth

Phase 3's checkpoint was saved from `SCFNOModel(PaddedFNO(FNO, 0), decoder)`.
PaddedFNO **auto-registers** `self.fno`, `self.lifting`, `self.fno_blocks`,
`self.projection` as nn.Module submodules (PyTorch auto-registers any
nn.Module assignment). Even though `.lifting`, `.fno_blocks`, `.projection`
are aliases to the same underlying tensors as `.fno.lifting` etc., PyTorch's
state_dict includes **both**:
- `fno.fno.lifting.*` (from the FNO submodule inside PaddedFNO)
- `fno.lifting.*` (from the aliased lifting submodule directly on PaddedFNO)

**Both have identical values** because they point to the same tensors.

### Why the "buggy" single-wrap loader still loaded Phase 3

The "buggy" model was `SCFNOModel(FNO_raw, decoder)` with keys `fno.lifting.*`
+ `decoder.*`. `load_state_dict(strict=False)` found that `fno.lifting.*` keys
**matched the checkpoint's `fno.lifting.*` alias keys** (from PaddedFNO's
dual registration). Phase 3 weights loaded successfully. The 40 "unexpected"
keys were `fno.fno.*` duplicates — harmless, same values already loaded.

### Conclusion

- **The load always worked.** Buggy and FIXED loaders produced identical
  Phase 3 PI-FNO models.
- **R7's loader fix is still correct for hygiene** — explicit PaddedFNO wrap
  + verification test prevents future confusion.
- **The "consequential" claim in R7's commit message was wrong** — the bug was
  a load-key ambiguity, not a data-integrity failure.
- **Phase 4A.0 measurements and 4A.1 training (v1 and v2) are all valid** —
  they measured/trained the correct Phase 3 PI-FNO.

### Verified diagnostic


Both loaders reached the same correct state via different key paths.

### Corrected narrative

The R7 commit title ("critical loader bug fix") should be read as
"loader clarity fix + regression test", not "data integrity bug fix".
Phase 4A.0 and 4A.1 v1/v2 remain valid.
