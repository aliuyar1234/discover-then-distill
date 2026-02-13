from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .layers import RMSNorm, CausalSelfAttention, MLP
from .mhc import MHCResidual


class AttentionSublayer(nn.Module):
    """Pre-norm attention sublayer."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            max_seq_len=cfg.max_seq_len,
            rope_theta=cfg.rope_theta,
            attn_dropout=cfg.attn_dropout,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        return self.attn(x, attn_mask=attn_mask)


class MLPSublayer(nn.Module):
    """Pre-norm MLP sublayer."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, act=cfg.mlp_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.mlp(x)


class MHCTransformerBlock(nn.Module):
    """
    Transformer block with mHC residual stream for both attention and MLP.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_res = MHCResidual(cfg.d_model, cfg.mhc, AttentionSublayer(cfg))
        self.mlp_res = MHCResidual(cfg.d_model, cfg.mhc, MLPSublayer(cfg))

    def forward(self, X: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        X = self.attn_res(X, attn_mask=attn_mask)
        X = self.mlp_res(X, attn_mask=None)
        return X


class MHCTransformerLM(nn.Module):
    """
    Decoder-only LM with mHC residual streams.

    Forward inputs:
      input_ids: [B,T]
    Output:
      logits: [B,T,V]
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.n_streams = cfg.mhc.n_streams
        # stream embeddings to break symmetry across streams (learnable)
        self.stream_emb = nn.Parameter(torch.zeros(self.n_streams, cfg.d_model))

        self.blocks = nn.ModuleList([MHCTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.stream_emb, mean=0.0, std=0.02)
        # output head weight is tied; no init needed.

    def _init_streams(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C] -> X: [B,T,n,C]
        """
        B, T, C = x.shape
        n = self.n_streams
        X = x.unsqueeze(2).repeat(1, 1, n, 1)
        X = X + self.stream_emb.view(1, 1, n, C)
        return X

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)  # [B,T,C]
        X = self._init_streams(x)

        for blk in self.blocks:
            X = blk(X, attn_mask=attn_mask)

        # readout: mean across streams (stable due to doubly-stochastic H_res)
        x_out = X.mean(dim=2)  # [B,T,C]
        x_out = self.final_norm(x_out)
        logits = self.lm_head(x_out)  # [B,T,V]
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        *,
        min_new_tokens: int = 0,
        prompt_lens: Optional[List[int]] = None,
        pad_token_id: int = 0,
        return_lens: bool = False,
        forbid_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Nucleus sampling generation that correctly handles right-padded prompts.

        Important: the *old* implementation assumed `input_ids` had no padding and always
        sampled from the last column (`logits[:, -1]`). That is **wrong** when prompts in
        a batch have different lengths and are right-padded: the last column may be a PAD
        token for shorter prompts, causing the model to generate after PAD instead of after
        the real prompt.

        This implementation supports a padded prompt batch by requiring `prompt_lens`.

        Args:
          input_ids: [B, T_prompt_max] with prompts right-padded by `pad_token_id`.
          min_new_tokens: minimum number of generated tokens before EOS can finish a sequence.
          prompt_lens: true prompt lengths (excluding padding). If None, assumes no padding.
          pad_token_id: id used for padding.
          return_lens: if True, also return per-sample output lengths.
          forbid_token_ids: token ids that must never be sampled (defaults to [pad_token_id]).

        Returns:
          ids: [B, T_out_max] padded on the right.
          (optional) lens: List[int] output lengths per sample.
        """

        self.eval()
        device = input_ids.device
        B, T = input_ids.shape

        if prompt_lens is None:
            prompt_lens = [T] * B
        if len(prompt_lens) != B:
            raise ValueError(f"prompt_lens must have length B={B}, got {len(prompt_lens)}")
        if max(prompt_lens) <= 0:
            raise ValueError("prompt_lens must be >= 1 for all samples")
        min_new_tokens = max(0, int(min_new_tokens))
        if min_new_tokens > max_new_tokens:
            min_new_tokens = int(max_new_tokens)

        # Default: never sample PAD.
        if forbid_token_ids is None:
            forbid_token_ids = [pad_token_id]
        # de-duplicate while preserving order
        forbid_token_ids = list(dict.fromkeys(forbid_token_ids))

        max_prompt = max(int(x) for x in prompt_lens)
        total_cap = max_prompt + max_new_tokens
        out = torch.full((B, total_cap), pad_token_id, dtype=input_ids.dtype, device=device)
        for i in range(B):
            L = int(prompt_lens[i])
            out[i, :L] = input_ids[i, :L]

        cur_lens = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step_idx in range(max_new_tokens):
            # Build a conditional batch by taking up to the last `max_seq_len` tokens
            # for each sample, and right-padding to a common window length.
            cur_max = int(cur_lens.max().item())
            window_len = min(cur_max, int(self.cfg.max_seq_len))
            if window_len <= 0:
                raise RuntimeError("window_len <= 0; check prompt_lens")

            ids_cond = torch.full((B, window_len), pad_token_id, dtype=input_ids.dtype, device=device)
            last_pos = torch.zeros(B, dtype=torch.long, device=device)
            for i in range(B):
                Li = int(cur_lens[i].item())
                start = max(0, Li - window_len)
                seg = out[i, start:Li]
                seg_len = int(seg.numel())
                if seg_len <= 0:
                    raise RuntimeError("empty segment in generate(); check prompt_lens")
                ids_cond[i, :seg_len] = seg
                last_pos[i] = seg_len - 1

            logits = self(ids_cond)  # [B, window_len, V]
            next_logits = logits[torch.arange(B, device=device), last_pos, :]
            # Forbid certain tokens from being generated.
            if forbid_token_ids:
                next_logits[:, forbid_token_ids] = -1e9
            # Optionally prevent immediate EOS so callers can enforce non-empty outputs.
            if eos_token_id is not None and step_idx < min_new_tokens:
                next_logits[:, eos_token_id] = -1e9

            next_logits = next_logits / max(temperature, 1e-6)
            probs = F.softmax(next_logits, dim=-1)

            # top-p (nucleus) sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[:, 0] = False  # keep at least one token
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_idx).squeeze(1)  # [B]

            # Only extend active (not-finished) sequences.
            active = ~finished
            if active.any():
                rows = torch.arange(B, device=device)[active]
                cols = cur_lens[active]
                out[rows, cols] = next_token[active]
                if eos_token_id is not None and (step_idx + 1) >= min_new_tokens:
                    finished = finished | (active & (next_token == eos_token_id))
                cur_lens = cur_lens + active.long()

            if eos_token_id is not None and finished.all():
                break

        max_out = int(cur_lens.max().item())
        out = out[:, :max_out]
        if return_lens:
            return out, cur_lens.tolist()
        return out
