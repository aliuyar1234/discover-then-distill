#!/usr/bin/env python3
"""
Prepare packed binary token datasets for fast training.

Input:
  - raw text file(s), one document per line (or at least line-delimited)
  - sentencepiece model

Output:
  - .bin file of uint16/uint32 tokens
  - metadata json

Usage:
  python scripts/prepare_data.py \
    --tokenizer data/tokenizer/spm32k.model \
    --input data/raw/train.txt \
    --output data/packed/train.bin \
    --append_eos 1
"""
from __future__ import annotations

import argparse
import os
import json
from typing import List

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.data.tokenizer import SpmTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", required=True, help="Comma-separated text files")
    ap.add_argument("--output", required=True, help="Output .bin")
    ap.add_argument("--append_eos", type=int, default=1)
    ap.add_argument("--max_lines", type=int, default=-1, help="For debugging; -1 means all")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tok = SpmTokenizer(args.tokenizer)
    vocab_size = tok.vocab_size
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    tokens: List[int] = []
    n_docs = 0
    n_lines = 0

    for path in args.input.split(","):
        path = path.strip()
        if not path:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if args.max_lines > 0 and n_lines >= args.max_lines:
                    break
                line = line.strip("\n")
                if not line:
                    continue
                ids = tok.encode(line, add_bos=False, add_eos=False)
                tokens.extend(ids)
                if args.append_eos:
                    tokens.append(tok.eos_id())
                n_docs += 1
                n_lines += 1

    arr = np.array(tokens, dtype=dtype)
    arr.tofile(args.output)

    meta = {
        "tokenizer": args.tokenizer,
        "input": args.input,
        "output": args.output,
        "dtype": str(dtype),
        "vocab_size": vocab_size,
        "n_tokens": int(arr.shape[0]),
        "n_docs": n_docs,
    }
    with open(args.output + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote", args.output, "tokens:", arr.shape[0], "dtype:", dtype)


if __name__ == "__main__":
    main()
