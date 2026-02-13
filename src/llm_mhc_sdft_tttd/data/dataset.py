from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PackedTokenDataset(Dataset):
    """
    Memory-mapped token dataset for next-token prediction.

    File format:
      - raw binary file containing token ids in little-endian uint16 or uint32
      - shape: [N] tokens

    This dataset returns (x, y) where:
      x: [seq_len]
      y: [seq_len] shifted by 1
    """
    path: str
    seq_len: int
    dtype: str = "uint16"  # or uint32
    seed: int = 1337

    def __post_init__(self):
        assert self.dtype in ("uint16", "uint32")
        np_dtype = np.uint16 if self.dtype == "uint16" else np.uint32
        self.data = np.memmap(self.path, dtype=np_dtype, mode="r")
        self.n_tokens = self.data.shape[0]
        assert self.n_tokens > self.seq_len + 1, "Dataset too small for seq_len"
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        # approximate number of samples; random sampling anyway
        return self.n_tokens // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # random offset
        start = self.rng.integers(0, self.n_tokens - (self.seq_len + 1))
        chunk = np.array(self.data[start : start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
