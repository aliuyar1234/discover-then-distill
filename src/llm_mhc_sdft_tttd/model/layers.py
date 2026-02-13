from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm_x = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return x * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (x1, x2) -> (-x2, x1)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """
    Rotary positional embeddings (RoPE). Applies to query and key.

    We generate cached cos/sin up to max_seq_len. Supports partial rotary via rotary_dim.
    """
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        # inv_freq shape [dim/2]
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq, dim]
        cos = emb.cos()[None, None, :, :]  # [1,1,seq,dim]
        sin = emb.sin()[None, None, :, :]  # [1,1,seq,dim]
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B, n_heads, T, head_dim]
        if self.cos_cached is None or seq_len > self.cos_cached.shape[2]:
            self._build_cache(max(seq_len, self.max_seq_len))
        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, max_seq_len: int, rope_theta: float, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model == n_heads * d_head, "d_model must equal n_heads*d_head"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.rope = RoPE(dim=d_head, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,T,C]
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, nh, T, dh]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k = self.rope(q, k, seq_len=T)

        # PyTorch SDPA expects [B, nh, T, dh]
        # use is_causal=True to apply causal mask. If attn_mask provided, merge.
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        out = self.o_proj(out)
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # LLaMA style:
        # gate_proj: d_model -> d_ff
        # up_proj: d_model -> d_ff
        # down_proj: d_ff -> d_model
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, act: str = "swiglu"):
        super().__init__()
        if act == "swiglu":
            self.mlp = SwiGLU(d_model, d_ff)
        elif act == "gelu":
            self.fc1 = nn.Linear(d_model, d_ff, bias=False)
            self.fc2 = nn.Linear(d_ff, d_model, bias=False)
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown act: {act}")
        self.act_name = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name == "swiglu":
            return self.mlp(x)
        else:
            return self.fc2(self.act(self.fc1(x)))
