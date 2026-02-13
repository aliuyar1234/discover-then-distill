from __future__ import annotations

import argparse
import json

import torch

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_mhc_sdft_tttd.config import ModelConfig
from llm_mhc_sdft_tttd.eval.perplexity import perplexity
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_model(ckpt_path: str, cfg_path: str, device: str) -> MHCTransformerLM:
    cfg = ModelConfig.from_json(Path(cfg_path).read_text(encoding="utf-8"))
    m = MHCTransformerLM(cfg).to(device)
    payload = torch.load(ckpt_path, map_location="cpu")
    sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    m.load_state_dict(sd, strict=True)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--base_cfg", required=True)
    ap.add_argument("--updated_ckpt", required=True)
    ap.add_argument("--updated_cfg", required=True)
    ap.add_argument("--val_bin", required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    base = load_model(args.base_ckpt, args.base_cfg, device=device)
    upd = load_model(args.updated_ckpt, args.updated_cfg, device=device)

    ppl_base = perplexity(
        base,
        args.val_bin,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        dtype=args.dtype,
        max_batches=args.max_batches,
    )
    ppl_upd = perplexity(
        upd,
        args.val_bin,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        dtype=args.dtype,
        max_batches=args.max_batches,
    )
    out = {"ppl_base": float(ppl_base), "ppl_updated": float(ppl_upd), "delta_ppl": float(ppl_upd - ppl_base)}

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {p}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
