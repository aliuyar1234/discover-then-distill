#!/usr/bin/env python3
"""
Convert TTT discovery logs into SDFT jsonl format:
  {"prompt": "...", "demonstration": "..."}
"""
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.data.ttt_to_sdft import (
    convert_discovery_logs_to_sdft_rows,
    write_sdft_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Comma-separated list of .json/.jsonl logs")
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--min_reward", type=float, default=None)
    ap.add_argument("--default_prompt", type=str, default=None)
    args = ap.parse_args()

    inputs = [x.strip() for x in args.input.split(",") if x.strip()]
    rows = convert_discovery_logs_to_sdft_rows(
        input_paths=inputs,
        min_reward=args.min_reward,
        default_prompt=args.default_prompt,
    )
    write_sdft_jsonl(args.out, rows)
    print(f"[ttt_to_sdft_jsonl] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
