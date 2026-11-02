# Step 5.1 — Buggy First Run (Preserved as Evidence)

**Date:** 2026-11-02
**Status:** INVALID — 3/4 variants broken by API bugs

## Bugs Discovered

1. **AdaptiveLambda.update() called with wrong arity**
   - Signature: `update(epoch: int, raw_cosine: float)`
   - Called: `update(epoch_cos)` — missing `epoch`
   - Effect: P5-C and P5-D λ stuck at 0.0001 entire run

2. **PCGrad set_loss_values() never called**
   - PCGrad.step() references `_loss_values["lambda_phys"]`
   - Not set before step()
   - Effect: P5-A and P5-D `gn=0.000`, no weight updates

3. **get_logged_metrics() returns list, not dict**
   - Returns `List[GradHistoryEntry]`
   - Treated as `dict["raw_cosine"]`
   - Effect: epoch_cos always None → 0.0 passed to update()

## Variant Status

| Variant | Console | Trained? | Verdict |
|---------|---------|----------|---------|
| P5-A | 63 lines | gn=0.000 all epochs | INVALID — no updates |
| P5-B | 63 lines | gn varies, best_val 0.068097 | SUSPICIOUS — physics=0.0000 |
| P5-C | 382 lines | λ stuck at 0.0001 | INVALID — AdaptiveLambda frozen |
| P5-D | 223 lines | killed mid-run | INCOMPLETE — no final artifacts |

## Checkpoints Preserved But Invalid

- p5-a_best.pt, p5-a_final.pt, p5-a_state.pt — invalid (no training)
- p5-b_best.pt, p5-b_final.pt, p5-b_state.pt — suspicious (physics=0)
- p5-c_best.pt, p5-c_final.pt, p5-c_state.pt — invalid (λ frozen)
- p5-d_best.pt — incomplete (killed mid-run)

## Rerun Required

After fixing the three bugs in `scripts/phase_5/step_5_1_train.py`,
all 4 variants must be retrained from scratch.
