# SC-FNO Reproducibility Package

Reproduces every phase of the SC-FNO project from a clean environment.
Every phase is locked with a Git tag and SHA-256 manifest.

## Quick Start

    git clone <repo-url> ~/projects/SC-FNO
    cd ~/projects/SC-FNO
    bash reproducibility/install.sh
    bash reproducibility/verify_locks.sh
    bash reproducibility/reproduce_phase.sh 5B

## What Is Locked

| Phase | Tag | Lock Commit | Content |
|-------|-----|-------------|---------|
| 0 | - | e7e70c0 | Environment, physics parameters |
| 1 | - | 1d9b320 | 2,000-scenario dataset |
| 2 | phase-2-lock | 0c9dfd7 | Standard FNO baseline |
| 3 | phase-3-closed | bc80727 | SC-FNO (degenerate physics, documented) |
| 4A | phase-4a-locked | dea218a | Phase 4A closure |
| 4B | phase-4b-locked | dd392bb | Phase 4B closure |
| 4 | phase-4-locked | 3b622f7 | Phase 4 overall |
| 5A | - | ec9029e | Corrected architecture |
| 5B | - | 68c1ce1 | Multi-seed physics (CV 0.41%) |
| 5C | - | 6bb3d12 | Causal boundary (B_n 3.75x) |
| 5D | - | 093e5c0 | Unseen + benchmark (8.6x) |
| 5 | phase-5-locked | 03a7402 | Final model selected |

Note: Phase tags may point to later commits that include corrections.
verify_locks.sh confirms each tag descends from its original lock commit.

## Critical SHA-256 Identifiers

- Phase 1 aggregate: 9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6
- Phase 1 test split: a71514eacaee3d09f8c00b34b6070c61ae919a59e8208d6164903d5fed44ffb2
- Phase 2 baseline checkpoint: 24fce0bad8ec9c665649f6d73f4c56efd66f92667d95da3011d32dcbf4388889
- Phase 3 PI-FNO checkpoint: a0522e57f0b43c8a96f2ad06a0d6fda15f8f47b0281ce02e9b5612dd8b02ee50
- Final selected model: 746431709cc7a4be...

## Phase-by-Phase Reproduction

    bash reproducibility/reproduce_phase.sh 0
    bash reproducibility/reproduce_phase.sh 1
    bash reproducibility/reproduce_phase.sh 2
    bash reproducibility/reproduce_phase.sh 3
    bash reproducibility/reproduce_phase.sh 4
    bash reproducibility/reproduce_phase.sh 5A
    bash reproducibility/reproduce_phase.sh 5B
    bash reproducibility/reproduce_phase.sh 5C
    bash reproducibility/reproduce_phase.sh 5D
    bash reproducibility/reproduce_phase.sh 5E

## What Reproduce Means

For each phase, reproduce_phase.sh confirms:

1. Tag exists and descends from the expected lineage
2. All upstream phase SHAs match
3. The phase gate script re-verifies (acceptance criteria still hold)
4. Committed checkpoints and metrics remain valid

It does NOT re-train from scratch by default. For a full retrain,
delete a phase checkpoint directory and re-run its training script.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04 on WSL2 | Ubuntu 22.04 on WSL2 |
| GPU | CUDA 11.8+ | RTX 3060+ (6 GB VRAM) |
| VRAM | 3 GB | 6 GB |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB free | 40 GB free |

Developed and verified on RTX 4050 Laptop (6 GB VRAM),
Ubuntu 22.04 on WSL2, NVIDIA driver 610.88.

## Common Issues

- nvidia-smi not found -> Enable WSL GPU passthrough
- ModuleNotFoundError neuralop -> Package neuraloperator, import neuralop
- CUDA out of memory -> Reduce batch size
- git merge-base --is-ancestor fails -> Re-clone from origin

## Citation

    [Author]. (2026). SC-FNO: Spectral-Coupled Physics-Informed
    Fourier Neural Operators for Real-Time Nuclear Reactor
    Transient Simulation. Manuscripts in preparation for
    Annals of Nuclear Energy and Nuclear Engineering and Design.

## Version History

- v1.0 (2026-11-03): Initial reproducibility package.
  Phase 0-5 locked, final model selected, all gates passed.
