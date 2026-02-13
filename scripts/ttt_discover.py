#!/usr/bin/env python3
"""
Run TTT-Discover on a toy environment (string matching).
This is only to validate the loop end-to-end. Replace with real environments.

Usage:
  python scripts/ttt_discover.py \
    --ckpt runs/pretrain/ckpt_latest.pt \
    --tokenizer data/tokenizer/spm32k.model \
    --out runs/ttt_toy \
    --target "HELLO"
"""
from __future__ import annotations

import argparse
import shlex
import sys

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.config import TTTDiscoverConfig
from llm_mhc_sdft_tttd.training.ttt_discover import ToyStringMatchEnv, run_ttt_discover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--objective", type=str, default="entropic", choices=["entropic", "expected"])
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <out>/ttt_state_latest.pt if available.",
    )
    args = ap.parse_args()

    env = ToyStringMatchEnv(problem_description="Guess the hidden target string.", target=args.target)
    cfg = TTTDiscoverConfig(
        out_dir=args.out,
        ttt_steps=args.steps,
        rollouts_per_step=args.rollouts,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        objective=args.objective,
    )

    result = run_ttt_discover(
        base_ckpt=args.ckpt,
        tokenizer_path=args.tokenizer,
        env=env,
        cfg=cfg,
        resume=args.resume,
        tracker_command=shlex.join(sys.argv),
    )

    print("BEST:", result["best_reward"], result["best_state"])


if __name__ == "__main__":
    main()
