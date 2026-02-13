#!/usr/bin/env python3
"""
Pretrain the mHC Transformer LM on a packed token dataset.

Usage:
  python scripts/pretrain.py \
    --train_bin data/packed/train.bin \
    --val_bin data/packed/val.bin \
    --out runs/pretrain_mhc_350m \
    --model configs/model_mhc_350m.json \
    --steps 200000
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.config import ModelConfig, PretrainConfig
from llm_mhc_sdft_tttd.training.pretrain import train_pretrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_bin", required=True)
    ap.add_argument("--val_bin", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="Path to model config json")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path, or 'auto' to use <out>/ckpt_latest.pt.",
    )
    args = ap.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        model_cfg = ModelConfig.from_json(f.read())

    pre_cfg = PretrainConfig(
        total_steps=args.steps,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_bs,
        grad_accum_steps=args.grad_accum,
        out_dir=args.out,
        device=args.device,
        dtype=args.dtype,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
    )

    train_pretrain(
        model_cfg=model_cfg,
        train_data_path=args.train_bin,
        val_data_path=args.val_bin,
        out_dir=args.out,
        pre_cfg=pre_cfg,
        resume_from=args.resume,
        tracker_command=shlex.join(sys.argv),
    )


if __name__ == "__main__":
    main()
