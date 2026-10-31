# Step 3.5 — Final Evaluation Report

**Gate 3.5:** PASS  
**Checks:** 11/11

## Overall Rel L2 Median

| Model | Rel L2 Median | Δ vs P2-FNO |
|-------|--------------|-------------|
| P2-FNO | 9.232144e-04 | baseline |
| P-FNO | 8.177713e-04 | +11.42% |
| PI-FNO | 7.680455e-04 | +16.81% |
| P-PI-FNO | 8.154379e-04 | +11.67% |

## Per-Family Rel L2 Median

| Family | P2-FNO | P-FNO | PI-FNO | P-PI-FNO |
|--------|--------|-------|--------|----------|
| step | — | 3.8948e-03 | 3.8902e-03 | 3.9034e-03 |
| ramp | — | 1.4787e-04 | 1.6433e-04 | 2.5443e-04 |
| sinusoidal | — | 5.2212e-04 | 4.4731e-04 | 6.0576e-04 |
| grf | — | 1.3888e-03 | 1.4463e-03 | 1.6085e-03 |

## P-PI-FNO Physics Audit Summary

- λ_phys nonzero all epochs: True
- Physics loss on test (recomputed): 2.51e-04
- Genuine zero: False
- Shared grad from physics: 0.2830
- Verdict: P-PI-FNO physics loss was NOT zero. It was ~2.4e-4 in the sanity check, consistent with the recomputed value here. The training log showed 0.00 because of epoch print formatting. The checkpoint IS a physics-trained model.

## Speedup Summary

- p2: 3.14 ms single, 1863/s throughput, 11.2× vs BDF
- p_fno: 3.14 ms single, 9641/s throughput, 11.2× vs BDF
- pi_fno: 3.14 ms single, 30686/s throughput, 11.2× vs BDF
- p_pi_fno: 3.20 ms single, 29136/s throughput, 11.0× vs BDF

## Resolution Generalization (PI-FNO)

- 200 query points: rel L2 = 3.4833e-02 ± 4.7810e-03
- 500 query points: rel L2 = 3.4854e-02 ± 4.7839e-03
- 1000 query points: rel L2 = 3.4860e-02 ± 4.7849e-03
- 5000 query points: rel L2 = 3.4866e-02 ± 4.7857e-03
- 10000 query points: rel L2 = 3.4866e-02 ± 4.7858e-03

## Honest Verdict

**Best SC-FNO variant: PI-FNO at 7.680455e-04 (+16.81% vs P2-FNO)**

SC-FNO achieves a measurable 16.81% improvement over the Phase 2 baseline on the locked test set.