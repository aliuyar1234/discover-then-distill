#!/usr/bin/env python3
"""
Evaluate a checkpoint on perplexity and/or capability probes.

Usage:
  python scripts/eval.py \
    --ckpt runs/sdft_smoke/sdft_latest.pt \
    --tokenizer data/tokenizer/spm.model \
    --ppl_bin data/packed/val.bin \
    --out_json runs/eval/report.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.config import ModelConfig
from llm_mhc_sdft_tttd.data.tokenizer import SpmTokenizer
from llm_mhc_sdft_tttd.eval.perplexity import perplexity
from llm_mhc_sdft_tttd.eval.probes import default_probes, probes_from_prompt_list, run_probes
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[eval] CUDA not available; falling back to CPU.")
        return "cpu"
    return device


def load_model_cfg(ckpt_path: str) -> ModelConfig:
    ckpt_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(ckpt_dir, "model_config.json")
    if not os.path.exists(cfg_path):
        raise ValueError(f"model_config.json not found next to checkpoint: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return ModelConfig.from_json(f.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ppl_bin", default=None)
    ap.add_argument("--ppl_seq_len", type=int, default=256)
    ap.add_argument("--ppl_bs", type=int, default=4)
    ap.add_argument("--ppl_dtype", choices=["uint16", "uint32"], default="uint16")
    ap.add_argument("--ppl_max_batches", type=int, default=50)
    ap.add_argument("--probe_max_new_tokens", type=int, default=64)
    ap.add_argument("--probe_temperature", type=float, default=0.7)
    ap.add_argument("--probe_top_p", type=float, default=0.95)
    ap.add_argument("--probe_prompts", default=None, help="Optional text file, one prompt per line")
    ap.add_argument("--skip_probes", action="store_true")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model_cfg = load_model_cfg(args.ckpt)
    model = MHCTransformerLM(model_cfg).to(device)

    payload = torch.load(args.ckpt, map_location="cpu")
    if "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    else:
        model.load_state_dict(payload, strict=True)
    model.eval()

    tok = SpmTokenizer(args.tokenizer)
    report = {
        "checkpoint": args.ckpt,
        "device": device,
        "perplexity": None,
        "probes": None,
    }

    if args.ppl_bin:
        report["perplexity"] = perplexity(
            model=model,
            data_path=args.ppl_bin,
            seq_len=args.ppl_seq_len,
            batch_size=args.ppl_bs,
            device=device,
            dtype=args.ppl_dtype,
            max_batches=args.ppl_max_batches,
        )

    if not args.skip_probes:
        if args.probe_prompts:
            with open(args.probe_prompts, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            probes = probes_from_prompt_list(prompts)
        else:
            probes = default_probes()
        report["probes"] = run_probes(
            model=model,
            tokenizer=tok,
            probes=probes,
            device=device,
            max_new_tokens=args.probe_max_new_tokens,
            temperature=args.probe_temperature,
            top_p=args.probe_top_p,
        )

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.out_json}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
