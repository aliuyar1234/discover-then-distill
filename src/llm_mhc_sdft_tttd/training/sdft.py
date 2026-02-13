from __future__ import annotations

import copy
import os
import math
import time
import json
import random
from dataclasses import asdict
from collections import deque
from typing import Optional, Dict, Any, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import ModelConfig, SDFTConfig
from ..model.transformer import MHCTransformerLM
from ..data.tokenizer import SpmTokenizer
from ..data.sdft_dataset import SDFTJsonlDataset, SDFTExample, make_teacher_prompt, identity_collate
from ..eval.perplexity import perplexity
from ..eval.probes import ProbeCase, default_probes, probes_from_prompt_list, run_probes
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
        print("[sdft] CUDA not available; falling back to CPU.")
        return "cpu"
    return device


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def pad_to_max(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


class ReplayBuffer:
    """Minimal bounded replay buffer for (prompt, demonstration) pairs."""

    def __init__(self, max_size: int):
        self.items: deque[SDFTExample] = deque(maxlen=max_size)

    def add(self, ex: SDFTExample) -> None:
        self.items.append(ex)

    def sample(self, rng: random.Random) -> SDFTExample:
        if not self.items:
            raise ValueError("Cannot sample from empty replay buffer")
        return self.items[rng.randrange(len(self.items))]

    def __len__(self) -> int:
        return len(self.items)


def sample_batch_with_replay(
    incoming_batch: Sequence[SDFTExample],
    replay: ReplayBuffer,
    replay_ratio: float,
    rng: random.Random,
) -> List[SDFTExample]:
    """Mix incoming examples with replayed examples according to replay_ratio."""
    out: List[SDFTExample] = []
    for ex in incoming_batch:
        use_replay = len(replay) > 0 and replay_ratio > 0.0 and (rng.random() < replay_ratio)
        if use_replay:
            out.append(replay.sample(rng))
        else:
            out.append(ex)
    # Add current stream examples after sampling, so replay only uses past examples.
    for ex in incoming_batch:
        replay.add(ex)
    return out


def should_revert_update(
    before: Dict[str, Optional[float]],
    after: Dict[str, Optional[float]],
    max_ppl_rel_increase: float,
    max_probe_score_drop: float,
) -> bool:
    """Return True if regression exceeds configured thresholds."""
    before_ppl = before.get("ppl")
    after_ppl = after.get("ppl")
    if before_ppl is not None and after_ppl is not None:
        if after_ppl > before_ppl * (1.0 + max_ppl_rel_increase):
            return True

    before_probe = before.get("probe_score")
    after_probe = after.get("probe_score")
    if before_probe is not None and after_probe is not None:
        if (before_probe - after_probe) > max_probe_score_drop:
            return True

    return False


@torch.no_grad()
def compute_gate_metrics(
    model: MHCTransformerLM,
    tokenizer: SpmTokenizer,
    device: str,
    gate_val_data_path: Optional[str],
    gate_val_seq_len: int,
    gate_val_batch_size: int,
    gate_val_dtype: str,
    gate_ppl_max_batches: int,
    probes: Sequence[ProbeCase],
    gate_probe_max_new_tokens: int,
    gate_probe_temperature: float,
    gate_probe_top_p: float,
    probe_seed: int = 1234,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {"ppl": None, "probe_score": None}

    if gate_val_data_path:
        metrics["ppl"] = perplexity(
            model=model,
            data_path=gate_val_data_path,
            seq_len=gate_val_seq_len,
            batch_size=gate_val_batch_size,
            device=device,
            dtype=gate_val_dtype,
            max_batches=gate_ppl_max_batches,
        )

    if probes:
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(probe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(probe_seed)
        try:
            probe_report = run_probes(
                model=model,
                tokenizer=tokenizer,
                probes=probes,
                device=device,
                max_new_tokens=gate_probe_max_new_tokens,
                temperature=gate_probe_temperature,
                top_p=gate_probe_top_p,
            )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        metrics["probe_score"] = float(probe_report["score"])

    return metrics


def compute_reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Reverse KL: KL(p_s || p_t) = sum_v p_s(v) [log p_s(v) - log p_t(v)]

    student_logits, teacher_logits: [B, L, V]
    Returns:
      kl: [B, L] (per position)
    """
    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_t = F.log_softmax(teacher_logits.detach(), dim=-1)
    p_s = log_p_s.exp()
    kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
    return kl


def save_sdft_checkpoint(
    out_dir: str,
    step: int,
    micro_step: int,
    model: MHCTransformerLM,
    ema_teacher: MHCTransformerLM,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    replay_rng: random.Random,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sdft_step_{step:07d}.pt")
    payload = {
        "step": step,
        "micro_step": micro_step,
        "model": model.state_dict(),
        "ema_teacher": ema_teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay_items": [asdict(x) for x in replay.items],
        "replay_rng_state": replay_rng.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)
    latest = os.path.join(out_dir, "sdft_latest.pt")
    torch.save(payload, latest)
    return path


def train_sdft(
    base_model_ckpt: str,
    tokenizer_path: str,
    sdft_data_path: str,
    out_dir: str,
    cfg: SDFTConfig,
    model_cfg: Optional[ModelConfig] = None,
    gate_val_data_path: Optional[str] = None,
    gate_val_seq_len: int = 256,
    gate_val_batch_size: int = 4,
    gate_val_dtype: str = "uint16",
    probe_prompts: Optional[List[str]] = None,
    eval_report_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    tracker_command: Optional[str] = None,
) -> None:
    """
    SDFT continual learning training loop.

    Inputs:
      base_model_ckpt: path to a checkpoint saved by pretraining (ckpt_step_*.pt) OR a HF-like state_dict.
      tokenizer_path: sentencepiece model file
      sdft_data_path: jsonl dataset path
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    tok = SpmTokenizer(tokenizer_path)
    os.makedirs(out_dir, exist_ok=True)

    resume_ckpt = None
    if resume_from:
        if resume_from == "auto":
            candidate = os.path.join(out_dir, "sdft_latest.pt")
            if os.path.exists(candidate):
                resume_ckpt = candidate
            else:
                print(f"[sdft] --resume auto: no checkpoint found at {candidate}; starting fresh.")
        else:
            candidate = resume_from
            if os.path.exists(candidate):
                resume_ckpt = candidate
            else:
                raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")

    # load model config from checkpoint dir if not provided
    if model_cfg is None:
        cfg_path = os.path.join(out_dir, "model_config.json") if resume_ckpt else ""
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                model_cfg = ModelConfig.from_json(f.read())
        # expect sibling file model_config.json
        ckpt_dir = os.path.dirname(base_model_ckpt)
        if model_cfg is None:
            cfg_path = os.path.join(ckpt_dir, "model_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                model_cfg = ModelConfig.from_json(f.read())
        else:
            raise ValueError("model_cfg not provided and model_config.json not found")

    model = MHCTransformerLM(model_cfg).to(device)
    ema_teacher = MHCTransformerLM(model_cfg).to(device)

    resume_payload: Optional[Dict[str, Any]] = None
    if resume_ckpt is not None:
        resume_payload = torch.load(resume_ckpt, map_location="cpu")
        if "model" not in resume_payload:
            raise ValueError(f"Invalid SDFT checkpoint (missing 'model'): {resume_ckpt}")
        model.load_state_dict(resume_payload["model"], strict=True)
        teacher_sd = resume_payload.get("ema_teacher", resume_payload["model"])
        ema_teacher.load_state_dict(teacher_sd, strict=True)
        print(f"[sdft] resumed model from {resume_ckpt}")
    else:
        # load weights
        payload = torch.load(base_model_ckpt, map_location="cpu")
        if "model" in payload:
            model.load_state_dict(payload["model"], strict=True)
            ema_teacher.load_state_dict(payload["model"], strict=True)
        else:
            model.load_state_dict(payload, strict=True)
            ema_teacher.load_state_dict(payload, strict=True)

    ema_teacher.eval()
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    # dtype
    amp_device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    if cfg.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    # fp16 autocast is CUDA-only.
    if amp_device_type != "cuda" and amp_dtype == torch.float16:
        amp_dtype = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    ds = SDFTJsonlDataset(sdft_data_path)
    if len(ds) == 0:
        raise ValueError(f"SDFT dataset is empty: {sdft_data_path}")
    dl = DataLoader(
        ds,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=identity_collate,
    )

    with open(os.path.join(out_dir, "sdft_config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(cfg), indent=2))
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        f.write(model_cfg.to_json())
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    # IMPORTANT: `total_steps`, `log_every`, and `save_every` are defined in
    # *optimizer steps* (i.e., after gradient accumulation), not micro-batches.
    opt_step = 0
    micro_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(cfg.seed + 17)
    replay = ReplayBuffer(max_size=max(1, cfg.replay_buffer_size))

    if resume_payload is not None:
        if "optimizer" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer"])
        opt_step = int(resume_payload.get("step", 0))
        micro_step = int(resume_payload.get("micro_step", opt_step * cfg.grad_accum_steps))
        for item in resume_payload.get("replay_items", []):
            if isinstance(item, dict) and "prompt" in item and "demonstration" in item:
                replay.add(SDFTExample(prompt=str(item["prompt"]), demonstration=str(item["demonstration"])))
        replay_state = resume_payload.get("replay_rng_state")
        if replay_state is not None:
            rng.setstate(replay_state)
        torch_state = resume_payload.get("torch_rng_state")
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)
        cuda_state = resume_payload.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
        print(
            f"[sdft] resumed optimizer/replay from {resume_ckpt} at step={opt_step} "
            f"micro_step={micro_step} replay={len(replay)}"
        )

    tracker = RunTracker(
        out_dir=out_dir,
        run_type="sdft",
        total_steps=cfg.total_steps,
        command=tracker_command,
        resume=(resume_payload is not None),
    )
    tracker.register_artifact("sdft_config", os.path.join(out_dir, "sdft_config.json"), required=True)
    tracker.register_artifact("model_config", os.path.join(out_dir, "model_config.json"), required=True)
    tracker.register_artifact("metrics", metrics_path, required=True)
    tracker.register_artifact("latest_checkpoint", os.path.join(out_dir, "sdft_latest.pt"), required=True)
    tracker.register_artifact(
        "final_checkpoint",
        os.path.join(out_dir, f"sdft_step_{cfg.total_steps:07d}.pt"),
        required=True,
    )
    if eval_report_path is None:
        tracker.register_artifact("final_eval_report", os.path.join(out_dir, "sdft_eval_report.json"), required=True)
    else:
        tracker.register_artifact("final_eval_report", eval_report_path, required=True)

    probes = (
        probes_from_prompt_list(probe_prompts)
        if probe_prompts is not None
        else default_probes()
    )

    gate_enabled = cfg.gate_every > 0 and (gate_val_data_path is not None or len(probes) > 0)
    gate_block_start_step = 0
    gate_before_metrics: Optional[Dict[str, Optional[float]]] = None
    gate_model_snapshot: Optional[Dict[str, torch.Tensor]] = None
    gate_teacher_snapshot: Optional[Dict[str, torch.Tensor]] = None
    gate_opt_snapshot: Optional[Dict[str, Any]] = None
    gate_events: List[Dict[str, Any]] = []
    metrics_mode = "a" if (resume_payload is not None and os.path.exists(metrics_path)) else "w"
    stopper = GracefulStopper()
    stopper.install()
    metrics_f = open(metrics_path, metrics_mode, encoding="utf-8")
    try:
        while opt_step < cfg.total_steps:
            if gate_enabled and gate_before_metrics is None:
                gate_before_metrics = compute_gate_metrics(
                    model=model,
                    tokenizer=tok,
                    device=device,
                    gate_val_data_path=gate_val_data_path,
                    gate_val_seq_len=gate_val_seq_len,
                    gate_val_batch_size=gate_val_batch_size,
                    gate_val_dtype=gate_val_dtype,
                    gate_ppl_max_batches=cfg.gate_ppl_max_batches,
                    probes=probes,
                    gate_probe_max_new_tokens=cfg.gate_probe_max_new_tokens,
                    gate_probe_temperature=cfg.gate_probe_temperature,
                    gate_probe_top_p=cfg.gate_probe_top_p,
                    probe_seed=cfg.seed + 101,
                )
                gate_model_snapshot = copy.deepcopy(model.state_dict())
                gate_teacher_snapshot = copy.deepcopy(ema_teacher.state_dict())
                gate_opt_snapshot = copy.deepcopy(optimizer.state_dict())
                gate_block_start_step = opt_step
                print(f"[gate] block start at step={opt_step} metrics={gate_before_metrics}")
                metrics_f.write(json.dumps({
                    "event": "gate_start",
                    "step": opt_step,
                    "metrics": gate_before_metrics,
                }) + "\n")
                metrics_f.flush()
                tracker.event("gate_start", step=opt_step, metrics=gate_before_metrics)

            for batch in dl:
                if opt_step >= cfg.total_steps:
                    break
                if gate_enabled and gate_before_metrics is None:
                    gate_before_metrics = compute_gate_metrics(
                        model=model,
                        tokenizer=tok,
                        device=device,
                        gate_val_data_path=gate_val_data_path,
                        gate_val_seq_len=gate_val_seq_len,
                        gate_val_batch_size=gate_val_batch_size,
                        gate_val_dtype=gate_val_dtype,
                        gate_ppl_max_batches=cfg.gate_ppl_max_batches,
                        probes=probes,
                        gate_probe_max_new_tokens=cfg.gate_probe_max_new_tokens,
                        gate_probe_temperature=cfg.gate_probe_temperature,
                        gate_probe_top_p=cfg.gate_probe_top_p,
                        probe_seed=cfg.seed + 101,
                    )
                    gate_model_snapshot = copy.deepcopy(model.state_dict())
                    gate_teacher_snapshot = copy.deepcopy(ema_teacher.state_dict())
                    gate_opt_snapshot = copy.deepcopy(optimizer.state_dict())
                    gate_block_start_step = opt_step
                    print(f"[gate] block start at step={opt_step} metrics={gate_before_metrics}")
                    metrics_f.write(json.dumps({
                        "event": "gate_start",
                        "step": opt_step,
                        "metrics": gate_before_metrics,
                    }) + "\n")
                    metrics_f.flush()
                    tracker.event("gate_start", step=opt_step, metrics=gate_before_metrics)

                incoming_batch = batch if isinstance(batch, list) else [batch]
                batch_examples = sample_batch_with_replay(
                    incoming_batch=incoming_batch,
                    replay=replay,
                    replay_ratio=cfg.replay_ratio,
                    rng=rng,
                )

                # build contexts
                prompts = [ex.prompt for ex in batch_examples]
                demos = [ex.demonstration for ex in batch_examples]

                # student context: just the question/prompt
                student_ctx = [tok.encode(p, add_bos=True, add_eos=False) for p in prompts]
                # teacher context: question + demonstration in template
                teacher_ctx = [tok.encode(make_teacher_prompt(p, d), add_bos=True, add_eos=False) for p, d in zip(prompts, demos)]

                # generate from student
                prompt_lens = [len(s) for s in student_ctx]
                student_ctx_tensor = pad_to_max(student_ctx, pad_id=tok.pad_id()).to(device)
                # IMPORTANT: prompts are right-padded; pass `prompt_lens` so generation starts
                # immediately after the true prompt, not after PAD.
                gen, gen_lens = model.generate(
                    student_ctx_tensor,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    eos_token_id=tok.eos_id(),
                    prompt_lens=prompt_lens,
                    pad_token_id=tok.pad_id(),
                    return_lens=True,
                )
                # split out generated tokens (remove context)
                gen_tokens = []
                for i in range(gen.shape[0]):
                    ctx_len = prompt_lens[i]
                    out_len = int(gen_lens[i])
                    gen_tokens.append(gen[i, ctx_len:out_len].tolist())
                mean_gen_len = float(sum(len(g) for g in gen_tokens) / max(1, len(gen_tokens)))

                # build full sequences for forward passes
                student_full = [student_ctx[i] + gen_tokens[i] for i in range(len(prompts))]
                teacher_full = [teacher_ctx[i] + gen_tokens[i] for i in range(len(prompts))]

                student_full_tensor = pad_to_max(student_full, pad_id=tok.pad_id()).to(device)
                teacher_full_tensor = pad_to_max(teacher_full, pad_id=tok.pad_id()).to(device)

            # We compute logits for all positions; then select the positions that predict the generated tokens.
            # For student: positions [ctx_len-1 : ctx_len+L-1]
            # For teacher: positions [teacher_ctx_len-1 : teacher_ctx_len+L-1]
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    student_logits_all = model(student_full_tensor)  # [B, S, V]
                    with torch.no_grad():
                        teacher_logits_all = ema_teacher(teacher_full_tensor)  # [B, S2, V]

            # collect per-sample aligned logits slices (ragged -> pad)
                student_slices = []
                teacher_slices = []
                max_L = 0
                for i in range(len(prompts)):
                    L = len(gen_tokens[i])
                    max_L = max(max_L, L)
                V = student_logits_all.size(-1)

                for i in range(len(prompts)):
                    L = len(gen_tokens[i])
                    # if generation ended early, skip empty
                    if L == 0:
                        # create dummy one token to avoid empty tensors
                        L = 1
                        gen_tokens[i] = [tok.eos_id()]

                    s_ctx_len = len(student_ctx[i])
                    t_ctx_len = len(teacher_ctx[i])

                    s_start = s_ctx_len - 1
                    s_end = s_start + L
                    t_start = t_ctx_len - 1
                    t_end = t_start + L

                    s_logits = student_logits_all[i, s_start:s_end, :]  # [L,V]
                    t_logits = teacher_logits_all[i, t_start:t_end, :]  # [L,V]

                    # pad to max_L
                    if s_logits.size(0) < max_L:
                        pad = torch.zeros(max_L - s_logits.size(0), V, device=device, dtype=s_logits.dtype)
                        s_logits = torch.cat([s_logits, pad], dim=0)
                        t_logits = torch.cat([t_logits, pad], dim=0)

                    student_slices.append(s_logits)
                    teacher_slices.append(t_logits)

                student_logits = torch.stack(student_slices, dim=0)  # [B,max_L,V]
                teacher_logits = torch.stack(teacher_slices, dim=0)

                kl = compute_reverse_kl(student_logits, teacher_logits)  # [B,max_L]
                loss = cfg.kl_coef * kl.mean() / cfg.grad_accum_steps

                loss.backward()
                micro_step += 1

                if micro_step % cfg.grad_accum_steps == 0:
                    if cfg.grad_clip is not None and cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # update EMA teacher
                    update_ema(ema_teacher, model, decay=cfg.teacher_ema_decay)

                    opt_step += 1
                    tracker.heartbeat(step=opt_step, total_steps=cfg.total_steps)

                    if opt_step % cfg.log_every == 0:
                        dt = time.time() - t0
                        loss_scalar = float(loss.item() * cfg.grad_accum_steps)
                        print(
                            f"[sdft step {opt_step}] loss={loss_scalar:.4f} "
                            f"micro_step={micro_step} dt={dt:.2f}s"
                        )
                        metrics_f.write(json.dumps({
                            "event": "train",
                            "step": opt_step,
                            "micro_step": micro_step,
                            "loss": loss_scalar,
                            "mean_gen_len": mean_gen_len,
                            "replay_buffer_size": len(replay),
                        }) + "\n")
                        metrics_f.flush()
                        tracker.heartbeat(
                            step=opt_step,
                            total_steps=cfg.total_steps,
                            metrics={
                                "loss": loss_scalar,
                                "mean_gen_len": mean_gen_len,
                                "replay_buffer_size": len(replay),
                            },
                        )
                        t0 = time.time()

                    if opt_step % cfg.save_every == 0:
                        ckpt_path = save_sdft_checkpoint(
                            out_dir=out_dir,
                            step=opt_step,
                            micro_step=micro_step,
                            model=model,
                            ema_teacher=ema_teacher,
                            optimizer=optimizer,
                            replay=replay,
                            replay_rng=rng,
                        )
                        print(f"[sdft step {opt_step}] saved {ckpt_path}")
                        tracker.event("checkpoint", step=opt_step, path=ckpt_path)
                        tracker.register_artifact(f"checkpoint_step_{opt_step:07d}", ckpt_path, required=False)

                    if gate_enabled and (opt_step - gate_block_start_step) >= cfg.gate_every:
                        gate_after_metrics = compute_gate_metrics(
                            model=model,
                            tokenizer=tok,
                            device=device,
                            gate_val_data_path=gate_val_data_path,
                            gate_val_seq_len=gate_val_seq_len,
                            gate_val_batch_size=gate_val_batch_size,
                            gate_val_dtype=gate_val_dtype,
                            gate_ppl_max_batches=cfg.gate_ppl_max_batches,
                            probes=probes,
                            gate_probe_max_new_tokens=cfg.gate_probe_max_new_tokens,
                            gate_probe_temperature=cfg.gate_probe_temperature,
                            gate_probe_top_p=cfg.gate_probe_top_p,
                            probe_seed=cfg.seed + 101,
                        )
                        assert gate_before_metrics is not None
                        should_revert = should_revert_update(
                            before=gate_before_metrics,
                            after=gate_after_metrics,
                            max_ppl_rel_increase=cfg.gate_max_ppl_rel_increase,
                            max_probe_score_drop=cfg.gate_max_probe_score_drop,
                        )
                        if should_revert:
                            assert gate_model_snapshot is not None
                            assert gate_teacher_snapshot is not None
                            assert gate_opt_snapshot is not None
                            model.load_state_dict(gate_model_snapshot, strict=True)
                            ema_teacher.load_state_dict(gate_teacher_snapshot, strict=True)
                            optimizer.load_state_dict(gate_opt_snapshot)
                            status = "reverted"
                            print(
                                f"[gate] regression detected at step={opt_step}; "
                                f"before={gate_before_metrics} after={gate_after_metrics}; reverted block weights."
                            )
                        else:
                            status = "accepted"
                            print(
                                f"[gate] accepted block ending at step={opt_step}; "
                                f"before={gate_before_metrics} after={gate_after_metrics}."
                            )

                        gate_events.append(
                            {
                                "step": opt_step,
                                "status": status,
                                "before": gate_before_metrics,
                                "after": gate_after_metrics,
                            }
                        )
                        metrics_f.write(json.dumps({
                            "event": "gate_end",
                            "step": opt_step,
                            "status": status,
                            "before": gate_before_metrics,
                            "after": gate_after_metrics,
                        }) + "\n")
                        metrics_f.flush()
                        tracker.event(
                            "gate_end",
                            step=opt_step,
                            status=status,
                            before=gate_before_metrics,
                            after=gate_after_metrics,
                        )
                        gate_before_metrics = None
                        gate_model_snapshot = None
                        gate_teacher_snapshot = None
                        gate_opt_snapshot = None
                        gate_block_start_step = opt_step

                    if stopper.stop_requested:
                        ckpt_path = save_sdft_checkpoint(
                            out_dir=out_dir,
                            step=opt_step,
                            micro_step=micro_step,
                            model=model,
                            ema_teacher=ema_teacher,
                            optimizer=optimizer,
                            replay=replay,
                            replay_rng=rng,
                        )
                        print(f"[sdft] pause checkpoint saved: {ckpt_path}")
                        tracker.event("pause_checkpoint", step=opt_step, path=ckpt_path)
                        tracker.finalize(
                            status="paused",
                            step=opt_step,
                            total_steps=cfg.total_steps,
                            message="Graceful pause requested by signal.",
                        )
                        return
    except Exception as exc:
        tracker.finalize(
            status="failed",
            step=opt_step,
            total_steps=cfg.total_steps,
            message=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        metrics_f.close()
        stopper.uninstall()

    # Always write a final checkpoint + a stable 'latest' pointer, so short runs
    # (e.g., smoke tests with total_steps < save_every) still produce artifacts.
    final_path = save_sdft_checkpoint(
        out_dir=out_dir,
        step=opt_step,
        micro_step=micro_step,
        model=model,
        ema_teacher=ema_teacher,
        optimizer=optimizer,
        replay=replay,
        replay_rng=rng,
    )
    latest_path = os.path.join(out_dir, "sdft_latest.pt")
    tracker.event("checkpoint", step=opt_step, path=final_path, final=True)
    if gate_events:
        gate_path = os.path.join(out_dir, "sdft_gate_events.json")
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(gate_events, f, indent=2)
        print(f"[sdft] gate log: {gate_path}")
        tracker.register_artifact("gate_events", gate_path, required=False)

    # Final evaluation artifact (Phase 12): always emit a JSON report.
    final_eval = {
        "step": opt_step,
        "perplexity": None,
        "probes": None,
    }
    if gate_val_data_path:
        final_eval["perplexity"] = perplexity(
            model=model,
            data_path=gate_val_data_path,
            seq_len=gate_val_seq_len,
            batch_size=gate_val_batch_size,
            device=device,
            dtype=gate_val_dtype,
            max_batches=cfg.gate_ppl_max_batches,
        )
    if probes:
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(cfg.seed + 101)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + 101)
        try:
            final_eval["probes"] = run_probes(
                model=model,
                tokenizer=tok,
                probes=probes,
                device=device,
                max_new_tokens=cfg.gate_probe_max_new_tokens,
                temperature=cfg.gate_probe_temperature,
                top_p=cfg.gate_probe_top_p,
            )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    if eval_report_path is None:
        eval_report_path = os.path.join(out_dir, "sdft_eval_report.json")
    eval_dir = os.path.dirname(eval_report_path)
    if eval_dir:
        os.makedirs(eval_dir, exist_ok=True)
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(final_eval, f, indent=2)
    with open(metrics_path, "a", encoding="utf-8") as metrics_f:
        metrics_f.write(json.dumps({
            "event": "final_eval",
            "step": opt_step,
            "perplexity": final_eval["perplexity"],
            "probe_score": (None if final_eval["probes"] is None else final_eval["probes"]["score"]),
        }) + "\n")
    tracker.event(
        "final_eval",
        step=opt_step,
        perplexity=final_eval["perplexity"],
        probe_score=(None if final_eval["probes"] is None else final_eval["probes"]["score"]),
    )
    print(f"[sdft] eval report: {eval_report_path}")
    print(f"[sdft] final checkpoint: {final_path}")
    print(f"[sdft] latest checkpoint: {latest_path}")
    tracker.finalize(
        status="completed",
        step=opt_step,
        total_steps=cfg.total_steps,
        message="SDFT completed.",
        metrics={
            "perplexity": final_eval["perplexity"],
            "probe_score": (None if final_eval["probes"] is None else final_eval["probes"]["score"]),
        },
    )
