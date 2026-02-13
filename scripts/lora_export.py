#!/usr/bin/env python3
"""
Export LoRA A/B weights from a checkpoint.

Supports:
  - a full training checkpoint containing `model` state_dict
  - a plain model state_dict
  - an existing LoRA-only state_dict
"""
from __future__ import annotations

import argparse
import os

import torch

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.model.lora import checkpoint_to_lora_state_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Input checkpoint path")
    ap.add_argument("--out", required=True, help="Output LoRA .pt path")
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu")
    lora_sd = checkpoint_to_lora_state_dict(payload)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(lora_sd, args.out)
    print(f"[lora_export] wrote {args.out} ({len(lora_sd)} tensors)")


if __name__ == "__main__":
    main()
