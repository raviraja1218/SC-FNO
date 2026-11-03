#!/usr/bin/env bash
# =============================================================================
# SC-FNO — Verify Phase 0-5 Locks Intact
# =============================================================================
# Confirms all phase tags exist, point to commits that descend from the
# expected lineage, and that critical SHAs (dataset, models) match.
#
# NOTE: Phase tags may be updated during phase corrections (e.g., Phase 4A
# after 5 reviewer corrections, Phase 5 after commit amendments). The script
# verifies that (a) tags exist, (b) their commit ancestors include the
# original lock SHAs, and (c) selected model / dataset SHAs match.
#
# Usage: bash reproducibility/verify_locks.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0

check() {
    local name="$1"
    local ok="$2"
    local detail="${3:-}"
    if [ "$ok" = "true" ]; then
        echo "  [PASS] $name${detail:+: $detail}"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name${detail:+: $detail}"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== SC-FNO Phase Lock Verification ==="
echo "Repo: $REPO_ROOT"
echo "HEAD: $(git rev-parse --short HEAD)"
echo ""

# --- Tags that must exist (SHA is checked as prefix-ancestor) ---
# For phase tags updated during corrections, we require the ORIGINAL lock
# commit to be reachable from the tag (proving tag lineage is intact).
declare -A PHASE_TAGS=(
    ["phase-2-lock"]="0c9dfd7"
    ["phase-3-closed"]="bc80727"
    ["phase-4a-locked"]="dea218a"
    ["phase-4b-locked"]="dd392bb"
    ["phase-4-locked"]="3b622f7"
    ["phase-5-locked"]="03a7402"
)

echo "--- Phase Tags (existence + ancestry) ---"
for tag in "${!PHASE_TAGS[@]}"; do
    original_sha="${PHASE_TAGS[$tag]}"
    if ! git rev-parse "$tag" &>/dev/null; then
        check "$tag exists" false "tag missing"
        continue
    fi
    # Confirm the tag commit descends from (or equals) the original lock SHA
    if git merge-base --is-ancestor "$original_sha" "$tag" 2>/dev/null; then
        actual=$(git rev-parse "$tag" | cut -c1-7)
        check "$tag" true "tag at $actual, descends from $original_sha"
    else
        check "$tag" false "tag $tag does NOT descend from original lock $original_sha"
    fi
done

# --- Untagged phase commits (must be ancestors of HEAD) ---
declare -A PHASE_COMMITS=(
    ["phase-0"]="e7e70c0"
    ["phase-1"]="1d9b320"
    ["phase-5A"]="ec9029e"
    ["phase-5B"]="68c1ce1"
    ["phase-5C"]="6bb3d12"
    ["phase-5D"]="093e5c0"
)

echo ""
echo "--- Untagged Phase Commits (ancestors of HEAD) ---"
for name in "${!PHASE_COMMITS[@]}"; do
    sha="${PHASE_COMMITS[$name]}"
    if git merge-base --is-ancestor "$sha" HEAD 2>/dev/null; then
        check "$name" true "$sha"
    else
        check "$name" false "$sha not ancestor of HEAD"
    fi
done

# --- Selected model SHA ---
echo ""
echo "--- Selected Model ---"
MODEL="models/sc_fno/final_sc_fno.pt"
if [ -f "$MODEL" ]; then
    actual_sha=$(sha256sum "$MODEL" | cut -c1-16)
    expected_prefix="746431709cc7a4be"
    if [[ "$actual_sha" == "$expected_prefix"* ]]; then
        check "selected model SHA" true "$actual_sha"
    else
        check "selected model SHA" false "expected ${expected_prefix}*, got $actual_sha"
    fi
else
    check "selected model exists" false "missing $MODEL"
fi

# --- Phase 1 dataset aggregate SHA ---
echo ""
echo "--- Phase 1 Dataset ---"
P1_MANIFEST="results/phase_1/phase_1_dataset_manifest.json"
if [ -f "$P1_MANIFEST" ]; then
    ok=$(python3 -c "
import json
m = json.load(open('$P1_MANIFEST'))
expected = '9049c7b42996fee7c08202703086bf0aebd513dbd1f77b1265ae546d6f241bf6'
print('true' if m['dataset']['aggregate_sha256'] == expected else 'false')
")
    if [ "$ok" = "true" ]; then
        check "Phase 1 aggregate SHA" true "9049c7b4..."
    else
        check "Phase 1 aggregate SHA" false "mismatch"
    fi
else
    check "Phase 1 manifest exists" false "missing"
fi

# --- Test set SHA ---
TEST_IDS="results/phase_1/splitting/test_ids.csv"
if [ -f "$TEST_IDS" ]; then
    actual=$(sha256sum "$TEST_IDS" | cut -c1-16)
    expected="a71514eacaee3d09"
    if [[ "$actual" == "$expected"* ]]; then
        check "test set SHA" true "$actual"
    else
        check "test set SHA" false "expected ${expected}*, got $actual"
    fi
fi

# --- Locked paths clean ---
echo ""
echo "--- Locked Paths (must be clean) ---"
status=$(git status --porcelain \
    results/phase_1 results/phase_2 results/phase_3 results/phase_4 \
    results/phase_5/step_5_A results/phase_5/step_5_B \
    results/phase_5/step_5_C results/phase_5/step_5_D \
    models/baseline models/sc_fno/phase_3 models/sc_fno/phase_4 \
    models/sc_fno/final_sc_fno.pt 2>/dev/null || echo "git error")
if [ -z "$status" ]; then
    check "locked paths clean" true "empty"
else
    check "locked paths clean" false "$status"
fi

# --- Summary ---
echo ""
echo "=========================================="
if [ "$FAIL" -eq 0 ]; then
    echo "=== ALL $PASS CHECKS PASSED ==="
    echo "Phase 0-5 locks intact. Project reproducible."
    exit 0
else
    echo "=== $FAIL CHECKS FAILED, $PASS PASSED ==="
    echo "Investigate failures before proceeding."
    exit 1
fi
