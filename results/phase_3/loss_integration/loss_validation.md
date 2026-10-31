# Step 3.3 — Loss Integration Validation Report

**Date:** 2026-10-31  
**Gate 3.3:** PASS  
**Checks:** 16/16

## Checklist

| # | Criterion | Result | Detail |
|---|-----------|--------|--------|
| 1 | CompositeLoss module exists | PASS |  |
| 2 | Per-state scaling merged (9 states) | PASS | 9 states |
| 3 | Item C1: IC loss reaches decoder bias | PASS | bias_grad_norm = 6.374725341796875 |
| 4 | Item C2: data loss does NOT reach decoder | PASS | projection_grad_norm = 0.05299289897084236 |
| 5 | Composite loss: all 3 finite | PASS | total = 0.532729983329773 |
| 6 | Shared FNO blocks receive gradient | PASS | grad norms: data=6.081e-02, phys=6.613e+01, ic=2.548e-02 |
| 7 | Gradient cos recorded (informational) | PASS | cos(data,phys) = -0.23893931813819705 |
| 8 | 4-variant configs verified | PASS | ['p2_fno', 'p_fno', 'pi_fno', 'p_pi_fno'] |
| 9 | All composite-loss tests pass | PASS | 3 passed, 0 failed |
| 10 | Locked Phase 1/2/3.1/3.2 untouched | PASS | empty |
| 11 | Phase 2 ckpt SHA unchanged | PASS | 24fce0ba... |
| 12 | Decoder init SHA unchanged | PASS | c30dd6ca... |
| 13 | Variant YAML: p2_fno | PASS |  |
| 14 | Variant YAML: p_fno | PASS |  |
| 15 | Variant YAML: pi_fno | PASS |  |
| 16 | Variant YAML: p_pi_fno | PASS |  |

## Item C — Reviewer Verification

C1: IC loss reaches decoder final bias — PASS
C2: Data loss does NOT reach decoder; projection bias receives grad — PASS

## 4-Variant Matrix

| Variant | Physics | Padding |
|---------|---------|---------|
| P2-FNO | No | No |
| P-FNO | No | Yes (reflection, 20) |
| PI-FNO | Yes | No |
| P-PI-FNO | Yes | Yes (reflection, 20) |
