"""
Project: LLM from scratch combining:
- mHC (Manifold-Constrained Hyper-Connections)
- SDFT continual learning (Self-Distillation Fine-Tuning)
- TTT-Discover (Learning to Discover at Test Time)

This file defines configuration dataclasses that are intentionally explicit.
Codex (or any agent) should NOT invent new parameters; add them here first.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Dict, Any

import json


DTypeStr = Literal["fp32", "bf16", "fp16"]


@dataclass
class TokenizerConfig:
    """SentencePiece tokenizer training + usage config."""
    vocab_size: int = 32000
    model_type: Literal["bpe", "unigram"] = "bpe"
    character_coverage: float = 0.9995
    # Special tokens (must be consistent everywhere)
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    # text normalization
    normalization_rule_name: str = "nmt_nfkc_cf"


@dataclass
class MHCConfig:
    """
    mHC hyperparameters (from the mHC paper, adapted to our single-GPU context).

    In the paper, they use expansion rate n=4, gating init alpha=0.01, Sinkhorn tmax=20. 
    (see Table 5 and Eq. 8-9 in the paper).
    """
    n_streams: int = 4  # expansion rate n
    alpha_init: float = 0.01  # gating factor init alpha
    sinkhorn_tmax: int = 20
    # initialization choices (not fully specified in the paper; we make explicit)
    init_hpre: Literal["uniform_sum1", "sigmoid_half"] = "uniform_sum1"
    init_hpost: Literal["uniform_sum1"] = "uniform_sum1"
    init_hres: Literal["approx_identity", "uniform"] = "approx_identity"
    # numerical stability
    sinkhorn_eps: float = 1e-6
    sinkhorn_clamp_min: float = -15.0
    sinkhorn_clamp_max: float = 15.0


@dataclass
class ModelConfig:
    """
    Decoder-only Transformer LM configuration (LLaMA-like),
    plus mHC residual-stream expansion.

    NOTE: This is intentionally dense-model only by default. 
    The mHC paper uses a DeepSeek-V3-inspired MoE architecture; implementing MoE is optional.
    """
    # vocab
    vocab_size: int = 32000
    # transformer
    n_layers: int = 16
    d_model: int = 2048
    n_heads: int = 16
    d_head: int = 128  # must satisfy d_model = n_heads * d_head
    # MLP
    d_ff: int = 5632  # typical 2.75x expansion for SwiGLU
    mlp_act: Literal["swiglu", "gelu"] = "swiglu"
    # dropout
    resid_dropout: float = 0.0
    attn_dropout: float = 0.0
    # norms
    rmsnorm_eps: float = 1e-6
    # RoPE
    rope_theta: float = 10000.0
    rope_partial_rotary_factor: float = 1.0  # 1.0 = full rotary
    max_seq_len: int = 2048
    # output head tie
    tie_embeddings: bool = True
    # mHC
    # IMPORTANT: use default_factory to avoid mutable-default dataclass error.
    mhc: MHCConfig = field(default_factory=MHCConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "ModelConfig":
        d = json.loads(s)
        # nested dataclass reconstruction
        mhc = MHCConfig(**d.pop("mhc"))
        return ModelConfig(mhc=mhc, **d)


@dataclass
class PretrainConfig:
    """
    Pretraining config for next-token prediction.
    """
    seed: int = 1337
    dtype: DTypeStr = "bf16"
    device: str = "cuda"
    # optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    # schedule
    warmup_steps: int = 2000
    total_steps: int = 200_000
    lr_decay: Literal["cosine", "linear", "constant"] = "cosine"
    # batch
    micro_batch_size: int = 2
    grad_accum_steps: int = 16
    # data
    seq_len: int = 2048
    # logging/checkpoint
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    out_dir: str = "runs/pretrain"
    # stability toggles
    use_compile: bool = False
    gradient_checkpointing: bool = True


@dataclass
class SDFTConfig:
    """
    Continual learning config (Self-Distillation Fine-Tuning).
    Implements reverse KL distillation from a demonstration-conditioned teacher.
    """
    seed: int = 1337
    dtype: DTypeStr = "bf16"
    device: str = "cuda"

    # generation (student policy sampling)
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95

    # distillation loss
    kl_coef: float = 1.0  # scales the reverse-KL loss
    # teacher construction
    teacher_ema_decay: float = 0.999
    # optimization
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0

    # training loop
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    # IMPORTANT: `total_steps`, `log_every`, `save_every` are *optimizer steps*
    # (after gradient accumulation), matching the pretrain loop.
    total_steps: int = 10_000
    log_every: int = 10
    save_every: int = 500
    out_dir: str = "runs/sdft"

    # replay (optional)
    replay_ratio: float = 0.2  # fraction of batches from replay buffer/dataset
    replay_buffer_size: int = 50_000

    # regression gates (Phase 8)
    # If gate_every > 0, evaluate before/after each block of gate_every optimizer steps.
    gate_every: int = 0
    # If after_ppl > before_ppl * (1 + gate_max_ppl_rel_increase), revert the block.
    gate_max_ppl_rel_increase: float = 0.20
    # If probe score drops by more than this amount, revert the block.
    gate_max_probe_score_drop: float = 0.10
    # eval budget for gate perplexity.
    gate_ppl_max_batches: int = 20
    # generation settings for capability probes.
    gate_probe_max_new_tokens: int = 64
    gate_probe_temperature: float = 0.7
    gate_probe_top_p: float = 0.95


@dataclass
class TTTDiscoverConfig:
    """
    Test-time training config (Learning to Discover at Test Time).
    We implement an entropic policy optimization objective (Eq. 1/2 in the paper),
    with optional adaptive beta and PUCT-based state reuse.
    """
    seed: int = 1337
    dtype: DTypeStr = "bf16"
    device: str = "cuda"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # which modules to LoRA-adapt
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")

    # RL / optimization steps
    ttt_steps: int = 50
    rollouts_per_step: int = 64  # reduce for single GPU
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95

    # entropic objective
    objective: Literal["entropic", "expected"] = "entropic"
    beta: float = 10.0  # constant beta; adaptive beta optional
    adaptive_beta: bool = True
    # Adaptive beta is set by constraining the KL of entropic weights q_beta against the uniform batch (Appendix A.1).
    # We default to gamma=ln(2), matching the paper.
    target_kl_gamma: float = 0.6931471805599453
    # regularization to base (theta0)
    kl_penalty_lambda: float = 0.1

    # state reuse via PUCT
    reuse_enabled: bool = True
    puct_c: float = 2.0

    # buffer
    buffer_max_size: int = 4096
    # optimization
    lr: float = 5e-5
    grad_clip: float = 1.0

    out_dir: str = "runs/ttt_discover"


def dump_config(path: str, cfg: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if hasattr(cfg, "to_json"):
            f.write(cfg.to_json())
        else:
            f.write(json.dumps(asdict(cfg), indent=2))
