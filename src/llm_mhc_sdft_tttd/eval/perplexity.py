from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.transformer import MHCTransformerLM
from ..data.dataset import PackedTokenDataset


@torch.no_grad()
def perplexity(model: MHCTransformerLM, data_path: str, seq_len: int, batch_size: int, device: str = "cuda", dtype: str = "uint16", max_batches: int = 100) -> float:
    ds = PackedTokenDataset(data_path, seq_len=seq_len, dtype=dtype, seed=123)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    losses = []
    model.eval()
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    mean_loss = sum(losses) / max(1, len(losses))
    return math.exp(min(20, mean_loss))
