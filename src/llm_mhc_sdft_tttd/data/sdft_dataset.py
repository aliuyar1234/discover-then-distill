from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset


@dataclass
class SDFTExample:
    prompt: str
    demonstration: str


class SDFTJsonlDataset(Dataset):
    """
    Reads jsonl with fields:
      - "prompt": string
      - "demonstration": string
    """
    def __init__(self, path: str):
        self.path = path
        self.items: List[SDFTExample] = []
        # Use utf-8-sig to transparently handle BOM-prefixed files (common on Windows/PowerShell).
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(SDFTExample(prompt=obj["prompt"], demonstration=obj["demonstration"]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SDFTExample:
        return self.items[idx]


def identity_collate(batch):
    """Return the batch as-is.

    Why?
      `SDFTJsonlDataset` yields `SDFTExample` dataclass objects. PyTorch's default
      `collate_fn` does not know how to stack / collate arbitrary dataclass
      instances and will raise a `TypeError` during DataLoader iteration.

      For SDFT we intentionally keep examples as simple Python objects
      (prompt/demonstration strings). Returning a Python list is sufficient.

    Important:
      This is a *top-level* function so it is picklable for multi-worker
      DataLoaders.
    """
    return batch


def make_teacher_prompt(prompt: str, demonstration: str) -> str:
    """
    Prompt template described in the SDFT paper (Sec. 3).
    """
    return (
        "<Question>\n"
        f"{prompt}\n\n"
        "This is an example for a response to the question:\n"
        "<Demonstration>\n"
        f"{demonstration}\n\n"
        "Now answer with a response of your own, including the thinking process:\n"
    )
