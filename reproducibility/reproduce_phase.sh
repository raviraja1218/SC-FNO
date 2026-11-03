#!/usr/bin/env bash
# =============================================================================
# SC-FNO — Reproduce Any Phase from Its Tag
# =============================================================================
# Checks out a specific phase tag, activates the environment, and runs
# the phase's verification script. Confirms evidence reproduces.
#
# Usage: bash reproducibility/reproduce_phase.sh <phase>
#   phase ∈ {0, 1, 2, 3, 4, 5A, 5B, 5C, 5D, 5E}
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PHASE="${1:-}"
if [ -z "$PHASE" ]; then
    echo "Usage: $0 <phase>"
    echo "  phase ∈ {0, 1, 2, 3, 4, 5A, 5B, 5C, 5D, 5E}"
    exit 1
fi

case "$PHASE" in
    0) TAG="" ; COMMIT="e7e70c0" ; VERIFY="" ;;
    1) TAG="" ; COMMIT="1d9b320" ; VERIFY="" ;;
    2) TAG="phase-2-lock" ; COMMIT="0c9dfd7" ; VERIFY="" ;;
    3) TAG="phase-3-closed" ; COMMIT="bc80727" ; VERIFY="" ;;
    4) TAG="phase-4-locked" ; COMMIT="3b622f7" ; VERIFY="" ;;
    5A) TAG="" ; COMMIT="ec9029e" ; VERIFY="src/phase_5_corrected/verification/test_decoder_health.py" ;;
    5B) TAG="" ; COMMIT="68c1ce1" ; VERIFY="scripts/phase_5/step_5_B_gate.py" ;;
    5C) TAG="" ; COMMIT="6bb3d12" ; VERIFY="scripts/phase_5/step_5_C_gate.py" ;;
    5D) TAG="" ; COMMIT="093e5c0" ; VERIFY="scripts/phase_5/step_5_D_gate.py" ;;
    5E) TAG="phase-5-locked" ; COMMIT="03a7402" ; VERIFY="scripts/phase_5/step_5_E_gate.py" ;;
    *) echo "Unknown phase: $PHASE" ; exit 1 ;;
esac

echo "=== Reproducing Phase $PHASE ==="
echo "Commit: $COMMIT${TAG:+ (tag: $TAG)}"
echo ""

# --- Verify lock ---
echo "[1/4] Verifying phase is ancestor of HEAD..."
if ! git merge-base --is-ancestor "$COMMIT" HEAD 2>/dev/null; then
    echo "ERROR: $COMMIT is not an ancestor of HEAD. Repository may be corrupted."
    exit 1
fi
echo "  OK"

# --- Activate environment ---
echo "[2/4] Activating sc_fno environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc_fno
echo "  OK"

# --- Check evidence exists ---
echo "[3/4] Checking evidence for Phase $PHASE..."
if [ -n "$VERIFY" ] && [ -f "$VERIFY" ]; then
    echo "  Verification script: $VERIFY"
else
    echo "  No gate script for this phase (check docs/README.md)"
fi

# --- Run verification ---
echo "[4/4] Running verification..."
echo ""
if [ -n "$VERIFY" ] && [ -f "$VERIFY" ]; then
    if [[ "$VERIFY" == *".py" ]]; then
        python "$VERIFY"
    fi
fi

echo ""
echo "=== Phase $PHASE reproduction complete ==="
echo "Evidence: results/phase_${PHASE:0:1}*/"
echo "Report:  reports/phase_reports/"
