# Phase 4A.1 — Reviewer Corrections Addendum

**Date:** 2026-11-01
**Predecessor commit:** `b9210fa` (Phase 4A.1 completion)
**Status:** Reporting corrections applied per reviewer feedback
**Re-run status:** NOT re-run (reviewer explicitly advised against)

---

## Corrections Applied

Seven reporting clarifications were raised by the reviewer. All are
accepted. None affect the experimental validity — the Phase 4A.1
implementation and execution are unchanged. Only the narrative is
corrected for accuracy.

---

### C1 — Test Count

**Original report inconsistency:** "9 tests" in implementation section,
"10/10" in gate table.

**Corrected:** The final test suite contains **10 tests**:
- **T1–T9** (original adaptive optimizer tests): Adam init, EMA math,
  plateau non-trigger on decreasing loss, plateau trigger on flat loss,
  no premature transition, automatic transition, L-BFGS closure
  determinism, checkpoint roundtrip, NaN/Inf detection
- **T10** (added in R7 during loader fix): Phase 3 PI-FNO loads
  bit-exact via `load_phase3_pi_fno_model`

**Final suite count: 10 tests. All 10 pass.**

---

### C2 — Gate Count Clarification

**Original report ambiguity:** Headline "16/16 PASS", evidence file "14/14".

**Corrected:** The plan defined **16 acceptance criteria** (Phase 4A.1
plan Section 8). The gate verification script (`step_4a_1_gate_check.py`)
implements **14 explicit checks** — criteria 4–9 (the nine unit tests)
are consolidated into a single check for "unit test evidence captured."

**Statement:** "Gate 4A.1 verification: **14/14 implemented checks PASS**,
which together cover **all 16 plan criteria**. The script intentionally
consolidates the 9 individual unit tests into one check; the count
difference reflects check granularity, not uncovered criteria."

---

### C3 — Best Validation Epoch Not Preserved

**Reviewer's flag:** Epoch 28 achieved best val_loss (0.066084), but
that checkpoint was not saved. `phase_4a_final.pt` is epoch 54 (0.066702).

**Corrected — accepted as a limitation:**

> The Adam optimizer's best recorded validation loss occurred at epoch 28
> (0.066084, 0.37% below the Phase 3 baseline of 0.066331). However, the
> training script's checkpoint policy saved only at (a) plateau detection
> (epoch 54) and (b) L-BFGS completion. **The epoch-28 weights were not
> preserved as a recoverable checkpoint** — they exist only as a log entry
> in `adam_training_log.csv`.
>
> **This is a limitation of the checkpoint-saving policy, not the Adam
> optimizer.** The optimizer *did* reach a lower validation loss at
> epoch 28; the checkpoint mechanism did not capture it. Phase 4A.2+ must
> save BOTH a transition checkpoint AND a best-validation checkpoint
> independently.

**Note:** The negative-result scope is narrower than originally stated —
Phase 4A.1 does **not** demonstrate "Adam fine-tuning cannot improve
validation accuracy." It demonstrates "the plateau-triggered checkpoint
policy does not preserve the best-observed validation point."

---

### C4 — L-BFGS Interpretation

**Reviewer's flag:** "Model reached a local minimum" overstates what
L-BFGS established.

**Corrected — adopting reviewer's wording verbatim:**

> L-BFGS terminated after **17 closure evaluations** on the fixed
> 16-scenario training batch because its configured `tolerance_change=1e-9`
> was satisfied. **This establishes convergence on the fixed batch only —
> not over the full training dataset, and not over the 200 validation
> scenarios.** The final loss of 0.599033 reflects the composite training
> objective (data + 0.398 × physics + IC) on those 16 scenarios, not the
> validation data loss (which is 0.066702).

**Terminology correction:** The report used "17 closures" ambiguously.
**Precise language: "17 closure evaluations"** — each closure call may
internally run multiple line-search evaluations. The L-BFGS internal
iteration count is not directly logged by the current implementation.

**Removed:** The claim that "the model was already at a local minimum
after Adam training." This is speculation. The correct factual statement
is that L-BFGS's configured convergence tolerance was met on the fixed
batch after 17 closures.

---

### C5 — Physics Loss Prediction Removed from Findings

**Reviewer's flag:** "Physics loss probably slightly worsened" is a
prediction, not a measurement.

**Corrected:** All physics-loss prediction language removed from the
findings section. Replaced with:

> **Physics loss change is not yet measured.** The adaptive fine-tune may
> have improved, worsened, or left unchanged the physical residuals.
> Step 4A.2 will measure this directly by comparing 9-equation residuals
> between `best_val.pt` (Phase 3 PI-FNO) and `phase_4a_final.pt`
> (Phase 4A.1), using identical query grid, scaling, and aggregation.

---

### C6 — Scope of Negative Result Narrowed

**Reviewer's flag:** "Adaptive fine-tuning does not improve" is too broad.
The experiment tested one specific configuration.

**Corrected — narrow scope statement:**

> The **specific optimizer configuration tested** — fresh Adam
> (lr=1e-3, betas=(0.9,0.999), CosineAnnealingLR T_max=100) with
> plateau-triggered transition to L-BFGS at λ_phys=0.398 on a fixed
> 16-scenario batch — starting from the Phase 3 PI-FNO epoch-36 checkpoint
> and fine-tuning for up to 100 epochs — **did not improve validation data
> loss over the Phase 3 baseline** on the 200-scenario validation split.
> The final checkpoint achieved val_data_loss = 0.066702 vs Phase 3's
> 0.066331 (+0.56% relative degradation).
>
> This finding **does not generalize** to:
> - Other optimizer settings (different lr, betas, scheduler)
> - Other λ_phys transition policies (A or C in the Phase 4A.1 plan)
> - Larger L-BFGS fixed batches
> - Different loss objectives (e.g., physics-only L-BFGS)
> - Checkpoint-saving policies that capture the best-validation point
> - Architectures or feature representations different from Phase 3 PI-FNO
>
> It establishes only that **this specific sequencing**, from **this
> specific starting checkpoint**, on **this specific dataset**, does not
> produce a validation improvement.

---

### C7 — 10⁻⁵ Physics Target Prediction Removed

**Reviewer's flag:** "10⁻⁵ target will not be achieved" is prediction.

**Corrected:** Removed from findings. Replaced with:

> The 10⁻⁵ normalized physics loss target defined in the Phase 4 plan will
> be **evaluated against measurement** in Step 4A.2. At Phase 4A.0.F
> baseline measurement (before adaptive fine-tune), the aggregate physics
> loss was **1.68e-3** — approximately 168× above the 10⁻⁵ target.
> Whether the adaptive fine-tune moved this value closer or further from
> the target is deferred to Step 4A.2's direct measurement.

---

## What Was NOT Changed

- **The Phase 4A.1 experimental run** — not re-run, not modified. All
  artifacts (`adam_stage.pt`, `phase_4a_final.pt`, `optimizer_state.pt`,
  logs) remain at their committed SHAs from `3507f72`.
- **The pre-registration** — unchanged from `67edac3`.
- **The gate verification** — 14/14 checks remain valid; only their
  relationship to the 16 plan criteria is clarified (C2).
- **The loader fix (R7–R9)** — retained as-is, evidence intact.
- **The archived v1 training** — retained as duplicate evidence.

---

## Deferred Measurements — Explicit List

The following are **deferred to Step 4A.2** (measurement-only, no training):

| Measurement | Method | Expected to show |
|-------------|--------|-------------------|
| 9 per-equation residuals | `compute_residuals` on both checkpoints | Unknown — measurement |
| Aggregate physics loss | `compute_physics_loss` on both checkpoints | Unknown — measurement |
| Initial-condition loss | `compute_ic_loss` on both checkpoints | Unknown — measurement |
| Validation data loss | Protocol A on 200 val scenarios (already measured: 0.066331 vs 0.066702) | +0.56% degradation |
| Boundary behavior | Residuals at t near 0 | Unknown — measurement |
| NaN/Inf check | Finite check on all above | Expected: no NaN/Inf |
| Physics 10⁻⁵ target comparison | `L_phys < 1e-5` on both | Unknown — measurement |

**No prediction is asserted for any of these before Step 4A.2 executes.**

---

## Reviewer's Guidance Adopted for Step 4A.2

Per reviewer's recommendation:

1. **Step 4A.2 is measurement-only** — no training, no hyperparameter changes
2. **Use 200 validation scenarios** — test set preserved for later
3. **Use identical query grid, scaling, aggregation** — same as Phase 4A.0.F
4. **Handle per-state derivative denormalization consistently** — per Phase 4A.0.R4 diagnostic
5. **Compare before/after** — Phase 3 PI-FNO vs Phase 4A.1 final
6. **No test-set tuning** — no use of test scenarios for any step of 4A.2

---

## Corrected Negative-Result Statement (Authoritative)

Final scope-bounded scientific finding for Phase 4A.1:

> **Phase 4A.1 did not improve validation data loss over Phase 3 PI-FNO.**
> The specific adaptive Adam→L-BFGS sequence tested (fresh Adam lr=1e-3
> CosineAnnealingLR T_max=100, plateau transition to L-BFGS at λ_phys=0.398
> on fixed 16-scenario batch), starting from the Phase 3 PI-FNO epoch-36
> checkpoint, achieved final val_data_loss = 0.066702 vs Phase 3 baseline
> 0.066331 (+0.56%). The plateau-triggered checkpoint policy did not
> capture the epoch-28 best (0.066084, −0.37% vs baseline). L-BFGS
> terminated after 17 closure evaluations on the fixed batch due to
> `tolerance_change=1e-9`; this does not establish local minimum over the
> full training set. Physics residual change — the deeper scientific
> question — is deferred to Step 4A.2 measurement.

---

## Milestone

**`PHASE_4A.1_LOCKED_WITH_REVIEWER_CORRECTIONS`**

All 7 reviewer clarifications resolved with corrected wording in this
addendum. Experimental evidence unchanged. Proceed to Step 4A.2
(measurement-only) after plan approval.

---

## Reference Files

| File | Purpose |
|------|---------|
| `adam_training_log.csv` | Per-epoch val losses (source of epoch-28 best) |
| `lbfgs_training_log.csv` | 17 closure evaluations |
| `transition_event.json` | Epoch 54, λ=0.398 |
| `training_summary.json` | Aggregate summary |
| `gate_4a_1_verification.json` | 14 check results (covering 16 plan criteria) |
| `../preflight/clarifications.md` | R1–R9 diagnostic trail |
| `pre_registration.yaml` | Signed-off configuration |

**No new measurements in this addendum. All values in the addendum cite
previously committed data files.**
