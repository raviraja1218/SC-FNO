"""Phase 2 Step 2.2.D — Phase 1 dataset loader for FNO training.

Loads the exact Phase 1 v4 dataset (aggregate SHA 9049c7b4...) using the
locked 1,600/200/200 scenario-level split and the locked training-only
z-score normalization. Read-only: does NOT modify any Phase 1 artifact.

Tensor contract:
    input:  [B, 1, 200]   rho_ext(t) normalized
    output: [B, 9, 200]   [n, C1..C6, Tf, Tc] normalized
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path.home() / "Research/Datasets/SC-FNO/raw/scenarios_2000"
SPLIT_DIR = PROJECT / "results/phase_1/splitting"
NORM_PATH = SPLIT_DIR / "normalization_statistics.json"

STATE_ORDER = ["n", "C1", "C2", "C3", "C4", "C5", "C6", "Tf", "Tc"]


class Phase1TransientDataset(Dataset):
    """Loads normalized (forcing, trajectory) pairs from Phase 1 dataset."""

    def __init__(
        self,
        split: Literal["train", "validation", "test"],
        normalization: dict | None = None,
    ):
        split_file = {
            "train": SPLIT_DIR / "train_ids.csv",
            "validation": SPLIT_DIR / "validation_ids.csv",
            "test": SPLIT_DIR / "test_ids.csv",
        }[split]
        self.scenario_ids = pd.read_csv(split_file)["scenario_id"].tolist()
        self.raw_root = RAW_ROOT
        self.split = split

        if normalization is None:
            normalization = json.load(open(NORM_PATH))
        self.norm = normalization["global"]

    def __len__(self) -> int:
        return len(self.scenario_ids)

    def _normalize(self, arr: np.ndarray, var: str) -> np.ndarray:
        m = self.norm[var]["mean"]
        s = self.norm[var]["std"]
        return (arr - m) / s

    def __getitem__(self, idx: int):
        sid = self.scenario_ids[idx]
        d = self.raw_root / sid

        forcing = np.load(d / "forcing.npy").astype(np.float32)     # (200,)
        traj = np.load(d / "trajectory.npy").astype(np.float32)     # (200, 9)

        forcing_n = self._normalize(forcing, "rho_ext")             # (200,)
        traj_n = np.empty_like(traj)
        for j, var in enumerate(STATE_ORDER):
            traj_n[:, j] = self._normalize(traj[:, j], var)

        # Add channel dim: [1, 200] and [9, 200]
        x = torch.from_numpy(forcing_n).unsqueeze(0)                # (1, 200)
        y = torch.from_numpy(traj_n).T                              # (9, 200)
        return x, y, sid


def make_dataloader(
    split: Literal["train", "validation", "test"],
    batch_size: int = 16,
    shuffle: bool | None = None,
    num_workers: int = 0,
    seed: int = 42,
    normalization: dict | None = None,
) -> DataLoader:
    """Deterministic dataloader for a Phase 1 split."""
    if shuffle is None:
        shuffle = (split == "train")

    ds = Phase1TransientDataset(split=split, normalization=normalization)

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=g,
    )
