#!/usr/bin/env python3
"""
Apply LoRA adapters to a base checkpoint for inference, optionally merge into base weights.
"""
from __future__ import annotations

import argparse
import os

import torch

from _bootstrap import bootstrap

bootstrap()

from llm_mhc_sdft_tttd.config import ModelConfig, TTTDiscoverConfig
from llm_mhc_sdft_tttd.data.tokenizer import SpmTokenizer
from llm_mhc_sdft_tttd.model.lora import apply_lora, load_lora, merge_lora_linears
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[lora_apply] CUDA not available; falling back to CPU.")
        return "cpu"
    return device


def load_model_cfg(base_ckpt: str, model_cfg_path: str | None) -> ModelConfig:
    if model_cfg_path:
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            return ModelConfig.from_json(f.read())
    default_path = os.path.join(os.path.dirname(base_ckpt), "model_config.json")
    if not os.path.exists(default_path):
        raise ValueError("model_config.json not found next to checkpoint; provide --model_cfg")
    with open(default_path, "r", encoding="utf-8") as f:
        return ModelConfig.from_json(f.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--lora", required=True, help="LoRA state dict .pt")
    ap.add_argument("--model_cfg", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target_modules", default=",".join(TTTDiscoverConfig().lora_target_modules))
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--merge", action="store_true", help="Merge LoRA into base and unload wrappers")
    ap.add_argument("--out_ckpt", default=None, help="Optional output checkpoint path")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model_cfg = load_model_cfg(args.base_ckpt, args.model_cfg)

    model = MHCTransformerLM(model_cfg).to(device)
    payload = torch.load(args.base_ckpt, map_location="cpu")
    if "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    else:
        model.load_state_dict(payload, strict=True)
    model.eval()

    target_modules = tuple(x.strip() for x in args.target_modules.split(",") if x.strip())
    replaced = apply_lora(
        model,
        target_module_suffixes=target_modules,
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if not replaced:
        raise RuntimeError("No target modules were instrumented with LoRA wrappers.")

    load_lora(model, args.lora)
    print(f"[lora_apply] loaded LoRA into {len(replaced)} modules")

    if args.merge:
        merged = merge_lora_linears(model, unload=True)
        print(f"[lora_apply] merged and unloaded {len(merged)} LoRA modules")

    if args.out_ckpt:
        out_dir = os.path.dirname(args.out_ckpt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, args.out_ckpt)
        print(f"[lora_apply] wrote checkpoint {args.out_ckpt}")

    if args.prompt is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer is required when using --prompt")
        tok = SpmTokenizer(args.tokenizer)
        ids = tok.encode(args.prompt, add_bos=True, add_eos=False)
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out, lens = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tok.eos_id(),
                prompt_lens=[len(ids)],
                pad_token_id=tok.pad_id(),
                return_lens=True,
            )
        out_ids = out[0, len(ids) : lens[0]].tolist()
        print(tok.decode(out_ids))


if __name__ == "__main__":
    main()
