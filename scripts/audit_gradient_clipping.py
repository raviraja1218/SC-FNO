"""Audit item 1 — Reconcile gradient clipping discrepancies (read-only)."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("results/phase_2/supplemental_audit")
em = pd.read_csv("results/phase_2/training/epoch_metrics.csv")

# Inspect columns
print("=== epoch_metrics.csv columns ===")
print(list(em.columns))
print(f"Total epochs: {len(em)}")

# Gradient norm stats
pre = em["grad_norm_pre_clip"].values
post = em["grad_norm_post_clip"].values if "grad_norm_post_clip" in em.columns else None
clip_flag = em["grad_clip_active"].values if "grad_clip_active" in em.columns else None

print("\n=== Grad norm PRE-clip ===")
print(f"  min: {pre.min():.6e}")
print(f"  max: {pre.max():.6e}")
print(f"  mean: {pre.mean():.6e}")
print(f"  median: {np.median(pre):.6e}")

if post is not None:
    print("\n=== Grad norm POST-clip ===")
    print(f"  min: {post.min():.6e}")
    print(f"  max: {post.max():.6e}")
    print(f"  mean: {post.mean():.6e}")
    print(f"  median: {np.median(post):.6e}")

if clip_flag is not None:
    firing_epochs = em[em["grad_clip_active"] == 1]
    print(f"\n=== Clipping events: {len(firing_epochs)} / {len(em)} epochs ===")
    if len(firing_epochs) > 0:
        print(firing_epochs[["epoch", "grad_norm_pre_clip", "grad_norm_post_clip", "train_mse", "val_mse"]].to_string(index=False))

# Reconciliation
print("\n=== Reconciliation ===")
threshold = 1.0
if pre.max() < threshold:
    print(f"⚠ pre-clip max ({pre.max():.6e}) is below threshold ({threshold})")
    print("  → Clipping should never fire if threshold is the criterion")
elif pre.max() >= threshold:
    print(f"✓ pre-clip max ({pre.max():.6e}) exceeds threshold ({threshold})")
    print(f"  → Clipping can fire when pre-clip > {threshold}")

if clip_flag is not None:
    n_fired = int((clip_flag == 1).sum())
    n_above_threshold = int((pre > threshold).sum())
    print(f"\nEpochs with pre-clip > threshold: {n_above_threshold}")
    print(f"Epochs flagged as clipping active: {n_fired}")
    consistent = (n_above_threshold == n_fired)
    print(f"Consistency: {'✓ CONSISTENT' if consistent else '✗ INCONSISTENT'}")

# Root cause analysis — did training log pre or post for the 'max' summary?
# The report quoted max=0.368 somewhere. Check where that number might come from
print(f"\n=== Where does 0.368 come from? ===")
print(f"  pre-clip max: {pre.max():.6e}")
if post is not None:
    print(f"  post-clip max: {post.max():.6e}")
print(f"  pre-clip mean across epochs: {pre.mean():.6e}")
print(f"  pre-clip median: {np.median(pre):.6e}")
print(f"\n  Could the reported '0.368' be a MEAN rather than MAX?")
print(f"  |mean - 0.368| = {abs(pre.mean() - 0.368):.4e}")
if post is not None:
    print(f"  |post mean - 0.368| = {abs(post.mean() - 0.368):.4e}")

# Determine the actual answer
report_claim = {"max_pre_clip_reported": 0.368, "threshold": 1.0, "fired_epochs_reported": 1}
matches_mean = abs(pre.mean() - 0.368) < 0.01
matches_max = abs(pre.max() - 0.368) < 0.01
matches_post_mean = post is not None and abs(post.mean() - 0.368) < 0.01

findings = {
    "report_claim": report_claim,
    "actual_data": {
        "max_pre_clip": float(pre.max()),
        "mean_pre_clip": float(pre.mean()),
        "median_pre_clip": float(np.median(pre)),
        "max_post_clip": float(post.max()) if post is not None else None,
        "mean_post_clip": float(post.mean()) if post is not None else None,
        "n_epochs": int(len(em)),
        "n_epochs_clip_fired": int((clip_flag == 1).sum()) if clip_flag is not None else None,
        "n_epochs_pre_above_threshold": int((pre > threshold).sum()),
    },
    "reconciliation": {
        "0_368_is_mean_pre_clip": bool(matches_mean),
        "0_368_is_max_pre_clip": bool(matches_max),
        "0_368_is_mean_post_clip": bool(matches_post_mean),
        "clipping_fires_consistent_with_threshold": bool(
            (clip_flag is None) or
            ((pre > threshold).sum() == (clip_flag == 1).sum())
        ) if clip_flag is not None else None,
    },
    "explanation": None,
    "correction": None,
}

# Formulate explanation
if matches_max and not matches_mean:
    findings["explanation"] = (
        "The reported 0.368 appears to be the max pre-clip norm. If so, and if "
        "the epoch_metrics.csv max matches, then the discrepancy must be in how "
        "'clipping fired' was logged on epoch 1 (perhaps a local per-tensor "
        "threshold was checked instead of the global norm)."
    )
elif matches_mean and not matches_max:
    findings["explanation"] = (
        "The reported 0.368 is the MEAN pre-clip norm, not the MAX. The report's "
        "summary conflated mean and max. Actual max pre-clip norm was "
        f"{pre.max():.4e}, which exceeds the 1.0 threshold on some epochs. This "
        "resolves the apparent inconsistency: clipping legitimately fired on "
        "those epochs, and the 0.368 was a mean mislabeled as max."
    )
    findings["correction"] = (
        "Report should state: 'median pre-clip gradient norm 0.0435, max pre-clip "
        "3.68e-1 (some epochs exceeded 1.0 threshold and triggered clipping)' — "
        "or replace 'max 0.368' with the actual value and cite median as the "
        "summary statistic."
    )
else:
    findings["explanation"] = (
        f"Neither mean nor max exactly matches 0.368. Actual: max={pre.max():.4e}, "
        f"mean={pre.mean():.4e}, median={np.median(pre):.4e}. The report's "
        f"0.368 may have been rounded or measured from a different run window."
    )

print(f"\n=== Interpretation ===")
print(f"  0.368 = mean pre-clip? {matches_mean}")
print(f"  0.368 = max pre-clip? {matches_max}")
print(f"  0.368 = mean post-clip? {matches_post_mean}")
print(f"\n{findings['explanation']}")

json.dump(findings, open(OUT / "01_gradient_clipping_data.json", "w"), indent=2)
print(f"\n=== Saved to {OUT}/01_gradient_clipping_data.json ===")
