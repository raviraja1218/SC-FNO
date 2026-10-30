# Step 1.5 Block A — Reconciliation Report

## Scope
Verify that Step 1.4 did not modify any Step 1.3 artifacts.

## Checks performed

| # | Check | Result | Detail |
|---|-------|--------|--------|
| 1 | git_ancestry_00d2819_ancestor_of_4306ce5 | PASS | rc=0 |
| 2 | working_tree_clean (excluding reconciliation outputs) | FAIL | M reports/figures/phase_1/figure_04_neutron_density.png;  M reports/figures/phase_1/figure_05_precursor_evolution.png;  M reports/figures/phase_1/figure_06_fuel_temperature.png;  M reports/figures/phase_1/figure_07_coolant_temperature.png;  M reports/figures/phase_1/figure_08_state_distributions.png;  M results/phase_1/gate_1_result.json;  M results/phase_1/generation/block_B_summary.json;  M results/phase_1/generation/block_C_summary.json;  M results/phase_1/generation/block_D_summary.json;  M results/phase_1/generation/step_1_3_manifest.json;  M results/phase_1/phase_1_dataset_manifest.json;  M results/phase_1/pilot/block_C_summary.json;  M results/phase_1/splitting/aggregate_checksums.json;  M results/phase_1/splitting/block_A_summary.json;  M results/phase_1/splitting/block_C_summary.json;  M results/phase_1/splitting/block_D_summary.json;  M results/phase_1/splitting/block_D_supplement_summary.json;  M results/phase_1/splitting/block_E_summary.json;  M results/phase_1/splitting/distribution_report.md;  M results/phase_1/splitting/distribution_supplement.md;  M results/phase_1/splitting/integrity_report.md;  M results/phase_1/splitting/normalization_report.md;  M results/phase_1/splitting/normalization_statistics.json;  M scripts/compare_bdf_radau.py;  M scripts/radau_compare_production.py;  M scripts/validate_production.py; ?? configs/thermal_hydraulics.v3.yaml; ?? configs/thermal_hydraulics.v4.yaml; ?? results/phase_1/generation/block_B_run_log_v4.txt; ?? results/phase_1/generation/block_C_run_log_v4.txt; ?? results/phase_1/generation/block_C_run_log_v4_final.txt; ?? results/phase_1/generation/block_C_run_log_v4_final2.txt; ?? results/phase_1/generation/block_C_run_log_v4_final3.txt; ?? results/phase_1/generation/block_D_physics_verify_v4.txt; ?? results/phase_1/generation/block_D_physics_verify_v4_v2.txt; ?? results/phase_1/generation/block_D_run_log_v4.txt; ?? results/phase_1/generation/block_D_run_log_v4_final.txt; ?? results/phase_1/generation/block_D_run_log_v4_final2.txt; ?? results/phase_1/generation/block_D_run_log_v4_final3.txt; ?? results/phase_1/phase_1_completion_report_v2.md; ?? results/phase_1/reviewer_figures/; ?? results/phase_1/solver/bdf_radau_methodology.v3.md; ?? results/phase_1/splitting/block_A_run_log_v4.txt; ?? results/phase_1/splitting/block_B_run_log_v4.txt; ?? results/phase_1/splitting/block_C_run_log_v4.txt; ?? results/phase_1/splitting/block_D_run_log_v4.txt; ?? results/phase_1/splitting/block_D_supplement_run_log_v4.txt; ?? results/phase_1/splitting/block_E_run_log_v4.txt |
| 3 | step_1_4_only_adds_new_files | PASS | 36 files, 0 non-additions |
| 4 | step_1_3_production_data_unchanged (200 sample) | PASS | 0 mismatches |
| 5 | step_1_3_evidence_intact (20 sample) | PASS | 0 mismatches from 20 sampled |

## Samples verified
- Production data: 200 scenarios (every 10th)
- Evidence files: 20 Step 1.3 evidence files

## Conclusion
**FAIL** — Step 1.4 only added new files under `results/phase_1/splitting/` and `reports/figures/phase_1/`; all Step 1.3 artifacts verified intact.
