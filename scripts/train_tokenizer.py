#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer from raw text files.

Usage:
  python scripts/train_tokenizer.py \
    --input data/raw/train.txt \
    --model_prefix data/tokenizer/spm32k \
    --vocab_size 32000 \
    --model_type bpe
"""
from __future__ import annotations

import argparse
import os

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Comma-separated list of input text files")
    ap.add_argument("--model_prefix", required=True, help="Output prefix (dir/prefix)")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--character_coverage", type=float, default=0.9995)
    ap.add_argument("--normalization_rule_name", type=str, default="nmt_nfkc_cf")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        normalization_rule_name=args.normalization_rule_name,
        # special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    print("Wrote:", args.model_prefix + ".model", args.model_prefix + ".vocab")


if __name__ == "__main__":
    main()
