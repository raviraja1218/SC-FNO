| Variant | Padding | Physics | IC | λ_phys init | λ_phys target | λ_phys ramp | Batch | LR | Max Epochs | Patience | Seed |
|---------|---------|---------|-----|------------|---------------|-------------|-------|-----|-----------|----------|------|
| p2_fno | 0 | False | False | 0.0 | 0.0 | 0 | 16 | 0.001 | N/A (Phase 2 baseline) | N/A | N/A |
| p_fno | 20 | False | False | 0.0001 | 1.0 | 100 | 16 | 0.001 | 200 | 30 | 42 |
| pi_fno | 0 | True | True | 0.0001 | 1.0 | 100 | 16 | 0.001 | 200 | 30 | 42 |
| p_pi_fno | 20 | True | True | 0.0001 | 1.0 | 100 | 16 | 0.001 | 200 | 30 | 42 |