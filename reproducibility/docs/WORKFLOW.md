# Reproducibility Workflows

## For a Reviewer Verifying the Project

    git clone <repo-url> ~/projects/SC-FNO
    cd ~/projects/SC-FNO
    git checkout phase-5-locked
    bash reproducibility/verify_locks.sh
    bash reproducibility/reproduce_phase.sh 5B
    bash reproducibility/reproduce_phase.sh 5C
    bash reproducibility/reproduce_phase.sh 5D
    bash reproducibility/reproduce_phase.sh 5E
    cat reports/phase_reports/phase_5_completion_report.md

## For a Collaborator Extending the Work

    bash reproducibility/install.sh
    bash reproducibility/verify_locks.sh
    git checkout -b extension/my-new-idea
    python3 -c "import torch; ckpt = torch.load('models/sc_fno/final_sc_fno.pt', map_location='cpu', weights_only=False); print('val:', ckpt['val_loss'])"

## For a Student Learning the Methods

Read in this order:

1. docs/PHASES_SUMMARY.md - high-level overview
2. reports/phase_reports/phase_0_completion_report.md - problem setup
3. reports/phase_reports/phase_1_completion_report.md - data generation
4. reports/phase_reports/phase_2_completion_report.md - FNO baseline
5. results/phase_5/step_5_A/B/C/D/*.log - Phase 5 execution traces
6. reports/phase_reports/phase_5_completion_report.md - final results

## For a Professor Evaluating Research Readiness

The MIT NSE narrative draws on:

- Phase 3 negative result (4 failure mechanisms identified)
- Phase 5 correction (first valid physics-informed model)
- Honest OOD limitation (10x worse extrapolation)
- Reproducibility package (this directory)
- Two manuscripts in preparation (Annals of Nuclear Energy, NED)
