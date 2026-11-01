"""Step 4A.0.R1 — Investigate Phase 3 λ_phys schedule and checkpoint value."""
import json
import yaml
import pandas as pd
import torch
from pathlib import Path

out = {}

# 1. Training config
print("=" * 60)
print("1. TRAINING CONFIG")
print("=" * 60)
cfg_path = "results/phase_3/training/pi_fno/training_config.yaml"
if Path(cfg_path).exists():
    cfg = yaml.safe_load(open(cfg_path))
    out["training_config"] = cfg
    print(json.dumps(cfg, indent=2, default=str))

# 2. Training summary
print("\n" + "=" * 60)
print("2. TRAINING SUMMARY")
print("=" * 60)
sum_path = "results/phase_3/training/pi_fno/summary.json"
if Path(sum_path).exists():
    summ = json.load(open(sum_path))
    out["training_summary"] = summ
    print(json.dumps(summ, indent=2, default=str))

# 3. Training log structure
print("\n" + "=" * 60)
print("3. TRAINING LOG")
print("=" * 60)
log_path = "results/phase_3/training/pi_fno/training_log.csv"
if Path(log_path).exists():
    log = pd.read_csv(log_path)
    print(f"Columns: {log.columns.tolist()}")
    print(f"Rows: {len(log)}")
    print(f"Min epoch: {log['epoch'].min()}")
    print(f"Max epoch: {log['epoch'].max()}")
    if 'val_data_loss' in log.columns:
        best_idx = log['val_data_loss'].idxmin()
        best_row = log.iloc[best_idx]
        print(f"Best val_data_loss: {best_row['val_data_loss']:.6f} at epoch {best_row['epoch']}")
    if 'lam_phys' in log.columns or 'lam_phys_current' in log.columns:
        lam_col = 'lam_phys' if 'lam_phys' in log.columns else 'lam_phys_current'
        print(f"\nFirst 5 lam_phys values:")
        print(log[['epoch', lam_col]].head(5).to_string())
        print(f"\nAround epoch 36:")
        print(log[log['epoch'].between(30, 45)][['epoch', lam_col]].to_string())
        print(f"\nLast 5 lam_phys values:")
        print(log[['epoch', lam_col]].tail(5).to_string())
        out["lam_phys_trajectory"] = {
            "first": float(log[lam_col].iloc[0]),
            "at_epoch_36": float(log[log['epoch'] == 36][lam_col].iloc[0]) if 36 in log['epoch'].values else None,
            "last": float(log[lam_col].iloc[-1]),
        }

# 4. Checkpoint
print("\n" + "=" * 60)
print("4. CHECKPOINT")
print("=" * 60)
ckpt = torch.load("models/sc_fno/phase_3/pi_fno/best_val.pt",
                   map_location="cpu", weights_only=False)
for k in ["epoch", "val_data_loss", "lam_phys", "variant", "padding", "use_physics"]:
    if k in ckpt:
        v = ckpt[k]
        if isinstance(v, torch.Tensor):
            v = v.item()
        print(f"{k}: {v}")
        out.setdefault("checkpoint", {})[k] = str(v)

# 5. Source code grep
print("\n" + "=" * 60)
print("5. SOURCE CODE: λ_phys SCHEDULE")
print("=" * 60)
src = Path("src/training/train_sc_fno.py").read_text().splitlines()
for i, line in enumerate(src, 1):
    if "lam_phys" in line.lower() or "lambda_phys" in line.lower():
        print(f"  {i:4d}  {line}")
out["source_lines_with_lam_phys"] = [
    {"line": i, "text": line.strip()}
    for i, line in enumerate(src, 1)
    if "lam_phys" in line.lower() or "lambda_phys" in line.lower()
]

# 6. Reconcile
print("\n" + "=" * 60)
print("6. RECONCILIATION")
print("=" * 60)
print(f"Checkpoint epoch: {ckpt.get('epoch')}")
print(f"Checkpoint lam_phys: {ckpt.get('lam_phys')}")
print(f"Checkpoint val_data_loss: {ckpt.get('val_data_loss')}")

with open("results/phase_4/phase_4a/preflight/lambda_schedule_evidence.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("\n=== 4A.0.R1 Lambda schedule evidence: SAVED ===")
