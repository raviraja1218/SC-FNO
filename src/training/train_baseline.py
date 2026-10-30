"""Phase 2 Step 2.3 training entry point (skeleton for Step 2.2.F).

Step 2.2 does NOT execute training. This file only establishes:
  - deterministic seeds
  - logging hooks
  - config loading

Actual training loop implementation belongs to Step 2.3.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 500
    early_stopping_patience: int = 30
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    device: str = "cuda"


def prepare_run(output_dir: Path, cfg: TrainingConfig) -> None:
    """Create output dirs and log the config. No training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_config.yaml", "w") as f:
        import yaml
        yaml.safe_dump(asdict(cfg), f)
    print(f"Prepared: {output_dir}")
    print(f"Config: {asdict(cfg)}")


if __name__ == "__main__":
    set_deterministic_seeds(42)
    cfg = TrainingConfig()
    prepare_run(PROJECT / "results/phase_2/training", cfg)
    print("Step 2.2.F: pipeline scaffolding OK (no training executed)")
