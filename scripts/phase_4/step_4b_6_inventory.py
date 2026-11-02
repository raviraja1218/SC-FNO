"""Step 4B.6.B — Enumerate all Phase 4B artifacts with SHA-256."""
import hashlib
import json
import csv
from pathlib import Path

OUT = Path("results/phase_4/phase_4b")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# Category mapping
def categorize(path):
    s = str(path)
    if "preflight" in s: return "4B.0_preflight"
    if "pcgrad" in s and "training" not in s: return "4B.1_pcgrad"
    if "adaptive_lambda" in s and "training" not in s: return "4B.2_adaptive_lambda"
    if "boundary" in s and "training" not in s: return "4B.3_boundary"
    if "training" in s: return "4B.4_training"
    if "evaluation" in s and "physics_revalidation_addendum" in s: return "4B.5_correction"
    if "evaluation" in s: return "4B.5_evaluation"
    if "closure" in s: return "4B.6_closure"
    return "other"

rows = []
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.suffix in (".json", ".csv", ".md", ".log", ".yaml"):
        rows.append({
            "path": str(p),
            "category": categorize(p),
            "sha256": sha(p),
            "bytes": p.stat().st_size,
        })

# Also add figures + tables
for fig_dir in [Path("reports/figures/phase_4b"), Path("reports/tables/phase_4b")]:
    if fig_dir.exists():
        for p in sorted(fig_dir.rglob("*")):
            if p.is_file():
                rows.append({
                    "path": str(p),
                    "category": "4B.5_figures_tables",
                    "sha256": sha(p),
                    "bytes": p.stat().st_size,
                })

# Add P4 model checkpoints
for p in sorted(Path("models/sc_fno/phase_4/phase_4b").rglob("*.pt")):
    if p.is_file():
        rows.append({
            "path": str(p),
            "category": "4B.4_models",
            "sha256": sha(p),
            "bytes": p.stat().st_size,
        })

# Write inventory
with open(OUT / "closure" / "phase_4b_artifact_inventory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "category", "sha256", "bytes"])
    w.writeheader()
    w.writerows(rows)

# Summary by category
summary = {}
for r in rows:
    cat = r["category"]
    summary.setdefault(cat, {"count": 0, "total_bytes": 0})
    summary[cat]["count"] += 1
    summary[cat]["total_bytes"] += r["bytes"]

print(f"Total artifacts: {len(rows)}")
for cat, s in sorted(summary.items()):
    print(f"  {cat}: {s['count']} files, {s['total_bytes']/1e6:.2f} MB")

with open(OUT / "closure" / "artifact_inventory_summary.json", "w") as f:
    json.dump({"total": len(rows), "by_category": summary}, f, indent=2)

print(f"\n[OK] Inventory: results/phase_4/phase_4b/closure/phase_4b_artifact_inventory.csv")
