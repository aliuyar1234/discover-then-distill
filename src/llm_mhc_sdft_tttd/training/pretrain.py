from __future__ import annotations

import os
import math
import time
import json
from dataclasses import asdict
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import ModelConfig, PretrainConfig
from ..model.transformer import MHCTransformerLM
from ..data.dataset import PackedTokenDataset
from ..tracking import RunTracker, GracefulStopper


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[pretrain] CUDA not available; falling back to CPU.")
        return "cpu"
    return device


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    out_dir: str,
    step: int,
    micro_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfgs: Dict[str, Any],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"ckpt_step_{step:07d}.pt")
    payload = {
        "step": step,
        "micro_step": micro_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfgs": cfgs,
    }
    torch.save(payload, ckpt_path)
    # Always write a stable 'latest' checkpoint file (do NOT rely on symlinks).
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    torch.save(payload, latest)
    return ckpt_path


@torch.no_grad()
def eval_loss(model: torch.nn.Module, dl: DataLoader, device: str, max_batches: int = 20) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train_pretrain(
    model_cfg: ModelConfig,
    train_data_path: str,
    val_data_path: Optional[str],
    out_dir: str,
    pre_cfg: PretrainConfig,
    train_dtype: str = "uint16",
    val_dtype: str = "uint16",
    resume_from: Optional[str] = None,
    tracker_command: Optional[str] = None,
) -> None:
    set_seed(pre_cfg.seed)
    device = resolve_device(pre_cfg.device)

    model = MHCTransformerLM(model_cfg).to(device)

    # dtype
    amp_device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    if pre_cfg.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif pre_cfg.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    if amp_device_type != "cuda" and amp_dtype == torch.float16:
        amp_dtype = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pre_cfg.lr,
        betas=pre_cfg.betas,
        eps=pre_cfg.eps,
        weight_decay=pre_cfg.weight_decay,
    )

    train_ds = PackedTokenDataset(train_data_path, seq_len=pre_cfg.seq_len, dtype=train_dtype, seed=pre_cfg.seed)
    train_dl = DataLoader(train_ds, batch_size=pre_cfg.micro_batch_size, num_workers=2, pin_memory=True)

    if val_data_path:
        val_ds = PackedTokenDataset(val_data_path, seq_len=pre_cfg.seq_len, dtype=val_dtype, seed=pre_cfg.seed + 1)
        val_dl = DataLoader(val_ds, batch_size=pre_cfg.micro_batch_size, num_workers=2, pin_memory=True)
    else:
        val_dl = None

    # optional compile
    if pre_cfg.use_compile:
        model = torch.compile(model)

    os.makedirs(out_dir, exist_ok=True)

    resume_ckpt = None
    if resume_from:
        if resume_from == "auto":
            candidate = os.path.join(out_dir, "ckpt_latest.pt")
            if os.path.exists(candidate):
                resume_ckpt = candidate
            else:
                print(f"[pretrain] --resume auto: no checkpoint found at {candidate}; starting fresh.")
        else:
            candidate = resume_from
            if os.path.exists(candidate):
                resume_ckpt = candidate
            else:
                raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")

    step = 0
    micro_step = 0
    if resume_ckpt is not None:
        payload = torch.load(resume_ckpt, map_location="cpu")
        if "model" not in payload:
            raise ValueError(f"Invalid pretrain checkpoint (missing 'model'): {resume_ckpt}")
        model.load_state_dict(payload["model"], strict=True)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        step = int(payload.get("step", 0))
        micro_step = int(payload.get("micro_step", step * pre_cfg.grad_accum_steps))
        print(f"[pretrain] resumed from {resume_ckpt} at step={step} micro_step={micro_step}")

    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        f.write(model_cfg.to_json())
    with open(os.path.join(out_dir, "pretrain_config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(pre_cfg), indent=2))
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    tracker = RunTracker(
        out_dir=out_dir,
        run_type="pretrain",
        total_steps=pre_cfg.total_steps,
        command=tracker_command,
        resume=(resume_ckpt is not None),
    )
    tracker.register_artifact("model_config", os.path.join(out_dir, "model_config.json"), required=True)
    tracker.register_artifact("pretrain_config", os.path.join(out_dir, "pretrain_config.json"), required=True)
    tracker.register_artifact("metrics", metrics_path, required=True)
    tracker.register_artifact("latest_checkpoint", os.path.join(out_dir, "ckpt_latest.pt"), required=True)
    tracker.register_artifact(
        "final_checkpoint",
        os.path.join(out_dir, f"ckpt_step_{pre_cfg.total_steps:07d}.pt"),
        required=True,
    )

    if step >= pre_cfg.total_steps:
        tracker.finalize(
            status="completed",
            step=step,
            total_steps=pre_cfg.total_steps,
            message="No-op: requested total steps already completed.",
        )
        return

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    train_it = iter(train_dl)
    metrics_mode = "a" if (resume_ckpt is not None and os.path.exists(metrics_path)) else "w"

    stopper = GracefulStopper()
    stopper.install()
    try:
        with open(metrics_path, metrics_mode, encoding="utf-8") as metrics_f:
            while step < pre_cfg.total_steps:
                try:
                    x, y = next(train_it)
                except StopIteration:
                    train_it = iter(train_dl)
                    x, y = next(train_it)

                # lr schedule
                lr = cosine_lr(step, pre_cfg.total_steps, pre_cfg.lr, pre_cfg.warmup_steps) if pre_cfg.lr_decay == "cosine" else pre_cfg.lr
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / pre_cfg.grad_accum_steps

                loss.backward()
                micro_step += 1

                if micro_step % pre_cfg.grad_accum_steps == 0:
                    if pre_cfg.grad_clip is not None and pre_cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), pre_cfg.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    # logging
                    if step % pre_cfg.log_every == 0:
                        dt = time.time() - t0
                        toks = pre_cfg.micro_batch_size * pre_cfg.grad_accum_steps * pre_cfg.seq_len
                        toks_per_s = toks / max(1e-6, dt)
                        loss_scalar = float(loss.item() * pre_cfg.grad_accum_steps)
                        print(f"[step {step}] loss={loss_scalar:.4f} lr={lr:.3e} toks/s={toks_per_s:.1f}")
                        metrics_f.write(json.dumps({
                            "event": "train",
                            "step": step,
                            "loss": loss_scalar,
                            "lr": float(lr),
                            "toks_per_s": float(toks_per_s),
                        }) + "\n")
                        metrics_f.flush()
                        tracker.heartbeat(
                            step=step,
                            total_steps=pre_cfg.total_steps,
                            metrics={"loss": loss_scalar, "lr": float(lr), "toks_per_s": float(toks_per_s)},
                        )
                        t0 = time.time()
                    else:
                        tracker.heartbeat(step=step, total_steps=pre_cfg.total_steps)

                    # eval
                    if val_dl is not None and step % pre_cfg.eval_every == 0:
                        vloss = eval_loss(model, val_dl, device=device, max_batches=20)
                        vppl = math.exp(min(20, vloss))
                        print(f"[step {step}] val_loss={vloss:.4f} ppl={vppl:.2f}")
                        metrics_f.write(json.dumps({
                            "event": "eval",
                            "step": step,
                            "val_loss": float(vloss),
                            "val_ppl": float(vppl),
                        }) + "\n")
                        metrics_f.flush()
                        tracker.event("eval", step=step, val_loss=float(vloss), val_ppl=float(vppl))

                    # checkpoint
                    if step % pre_cfg.save_every == 0:
                        ckpt = save_checkpoint(
                            out_dir=out_dir,
                            step=step,
                            micro_step=micro_step,
                            model=model,
                            optimizer=optimizer,
                            cfgs={"model": asdict(model_cfg), "pretrain": asdict(pre_cfg)},
                        )
                        print(f"[step {step}] saved {ckpt}")
                        tracker.event("checkpoint", step=step, path=ckpt)
                        tracker.register_artifact(f"checkpoint_step_{step:07d}", ckpt, required=False)

                    if stopper.stop_requested:
                        ckpt = save_checkpoint(
                            out_dir=out_dir,
                            step=step,
                            micro_step=micro_step,
                            model=model,
                            optimizer=optimizer,
                            cfgs={"model": asdict(model_cfg), "pretrain": asdict(pre_cfg)},
                        )
                        print(f"[pretrain] pause checkpoint saved: {ckpt}")
                        tracker.event("pause_checkpoint", step=step, path=ckpt)
                        tracker.finalize(
                            status="paused",
                            step=step,
                            total_steps=pre_cfg.total_steps,
                            message="Graceful pause requested by signal.",
                        )
                        return

        # Always write a final checkpoint + ckpt_latest, even for very short runs
        # (e.g., smoke tests with total_steps < save_every).
        ckpt = save_checkpoint(
            out_dir=out_dir,
            step=step,
            micro_step=micro_step,
            model=model,
            optimizer=optimizer,
            cfgs={"model": asdict(model_cfg), "pretrain": asdict(pre_cfg)},
        )
        print(f"[final step {step}] saved {ckpt}")
        tracker.event("checkpoint", step=step, path=ckpt, final=True)
        tracker.finalize(
            status="completed",
            step=step,
            total_steps=pre_cfg.total_steps,
            message="Pretraining completed.",
        )
    except Exception as exc:
        tracker.finalize(
            status="failed",
            step=step,
            total_steps=pre_cfg.total_steps,
            message=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        stopper.uninstall()
