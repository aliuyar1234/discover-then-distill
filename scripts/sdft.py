#!/usr/bin/env python3
"""
Run Self-Distillation Fine-Tuning (SDFT) continual learning.

Dataset format (jsonl):
  {"prompt": "...", "demonstration": "..."}  # one per line

Usage:
  python scripts/sdft.py \
    --ckpt runs/pretrain/ckpt_latest.pt \
    --tokenizer data/tokenizer/spm32k.model \
    --data data/sdft/demo.jsonl \
    --out runs/sdft_run
"""
from __future__ import annotations

import argparse
import shlex
import sys

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.config import SDFTConfig
from llm_mhc_sdft_tttd.training.sdft import train_sdft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--micro_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--replay_ratio", type=float, default=0.2)
    ap.add_argument("--replay_buffer_size", type=int, default=50_000)
    ap.add_argument("--gate_every", type=int, default=0, help="0 disables regression gates")
    ap.add_argument("--gate_val_bin", type=str, default=None, help="Packed token validation bin for perplexity gate")
    ap.add_argument("--gate_val_seq_len", type=int, default=256)
    ap.add_argument("--gate_val_bs", type=int, default=4)
    ap.add_argument("--gate_val_dtype", type=str, default="uint16", choices=["uint16", "uint32"])
    ap.add_argument("--gate_max_ppl_rel_increase", type=float, default=0.20)
    ap.add_argument("--gate_max_probe_drop", type=float, default=0.10)
    ap.add_argument("--gate_probe_prompts", type=str, default=None, help="Optional path to text file, one prompt per line")
    ap.add_argument("--eval_report_json", type=str, default=None, help="Path for final SDFT eval JSON (default: <out>/sdft_eval_report.json)")
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path, or 'auto' to use <out>/sdft_latest.pt.",
    )
    args = ap.parse_args()

    cfg = SDFTConfig(
        total_steps=args.steps,
        out_dir=args.out,
        device=args.device,
        dtype=args.dtype,
        micro_batch_size=args.micro_bs,
        grad_accum_steps=args.grad_accum,
        log_every=args.log_every,
        save_every=args.save_every,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        replay_ratio=args.replay_ratio,
        replay_buffer_size=args.replay_buffer_size,
        gate_every=args.gate_every,
        gate_max_ppl_rel_increase=args.gate_max_ppl_rel_increase,
        gate_max_probe_score_drop=args.gate_max_probe_drop,
    )

    probe_prompts = None
    if args.gate_probe_prompts:
        with open(args.gate_probe_prompts, "r", encoding="utf-8") as f:
            probe_prompts = [line.strip() for line in f if line.strip()]

    train_sdft(
        base_model_ckpt=args.ckpt,
        tokenizer_path=args.tokenizer,
        sdft_data_path=args.data,
        out_dir=args.out,
        cfg=cfg,
        gate_val_data_path=args.gate_val_bin,
        gate_val_seq_len=args.gate_val_seq_len,
        gate_val_batch_size=args.gate_val_bs,
        gate_val_dtype=args.gate_val_dtype,
        probe_prompts=probe_prompts,
        eval_report_path=args.eval_report_json,
        resume_from=args.resume,
        tracker_command=shlex.join(sys.argv),
    )


if __name__ == "__main__":
    main()
