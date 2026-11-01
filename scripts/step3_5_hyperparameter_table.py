"""Missing Item 3 — Consolidated hyperparameter table."""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "scripts")

variants = ["p_fno", "pi_fno", "p_pi_fno"]
rows = []
for v in variants:
    cfg_path = f"results/phase_3/training/{v}/training_config.yaml"
    if not Path(cfg_path).exists():
        continue
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    rows.append({
        "variant": cfg.get("variant", v),
        "padding": cfg.get("padding", 0),
        "use_physics": cfg.get("use_physics", False),
        "use_ic": cfg.get("use_ic", False),
        "lam_phys_init": cfg.get("lam_phys_init", 0.0),
        "lam_phys_target": cfg.get("lam_phys_target", 0.0),
        "lam_phys_ramp_epochs": cfg.get("lam_phys_ramp_epochs", 0),
        "batch_size": cfg.get("batch_size", 16),
        "lr": cfg.get("lr", 1e-3),
        "max_epochs": cfg.get("max_epochs", 200),
        "patience": cfg.get("patience", 30),
        "seed": cfg.get("seed", 42),
    })

# Add P2-FNO reference row
rows.insert(0, {
    "variant": "p2_fno",
    "padding": 0,
    "use_physics": False,
    "use_ic": False,
    "lam_phys_init": 0.0,
    "lam_phys_target": 0.0,
    "lam_phys_ramp_epochs": 0,
    "batch_size": 16,
    "lr": 1e-3,
    "max_epochs": "N/A (Phase 2 baseline)",
    "patience": "N/A",
    "seed": "N/A",
})

df = pd.DataFrame(rows)
out_path = "results/phase_3/evaluation/table_hyperparameters.csv"
df.to_csv(out_path, index=False)
print(f"Saved {out_path}")
print(df.to_string())

# Also as markdown for direct report inclusion
md_lines = ["| Variant | Padding | Physics | IC | λ_phys init | λ_phys target | λ_phys ramp | Batch | LR | Max Epochs | Patience | Seed |",
            "|---------|---------|---------|-----|------------|---------------|-------------|-------|-----|-----------|----------|------|"]
for _, r in df.iterrows():
    md_lines.append(f"| {r['variant']} | {r['padding']} | {r['use_physics']} | {r['use_ic']} | "
                    f"{r['lam_phys_init']} | {r['lam_phys_target']} | {r['lam_phys_ramp_epochs']} | "
                    f"{r['batch_size']} | {r['lr']} | {r['max_epochs']} | {r['patience']} | {r['seed']} |")

md_path = "results/phase_3/evaluation/table_hyperparameters.md"
Path(md_path).write_text("\n".join(md_lines))
print(f"Saved {md_path}")
