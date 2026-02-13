from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MHCConfig
from .layers import RMSNorm


def sinkhorn_knopp(
    log_alpha: torch.Tensor,
    tmax: int = 20,
    eps: float = 1e-6,
    clamp_min: float = -15.0,
    clamp_max: float = 15.0,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp projection onto the Birkhoff polytope (doubly-stochastic matrices).

    Paper mapping:
      M(0) = exp(H~_res)
      M(t) = Tr( Tc( M(t-1) ) )
    where Tc normalizes columns to sum to 1 and Tr normalizes rows to sum to 1. (Eq. 9)

    Here we operate in normal space (not log-space) but we clamp log_alpha for numerical stability.
    Input:
        log_alpha: [..., n, n] unconstrained (real-valued).
    Output:
        P: [..., n, n] approximately doubly stochastic
    """
    # clamp before exp to avoid overflow
    log_alpha = torch.clamp(log_alpha, clamp_min, clamp_max)
    P = torch.exp(log_alpha)  # positive
    # iterative normalization
    for _ in range(tmax):
        # column normalization
        col_sum = P.sum(dim=-2, keepdim=True)  # sum over rows -> shape [..., 1, n]
        P = P / (col_sum + eps)
        # row normalization
        row_sum = P.sum(dim=-1, keepdim=True)  # shape [..., n, 1]
        P = P / (row_sum + eps)
    return P


class MHCMapping(nn.Module):
    """
    Computes (H_pre, H_post, H_res) for a given residual stream X.

    Shapes:
      X: [B, T, n, C]
      flatten: [B, T, n*C]
      H_pre:  [B, T, n]
      H_post: [B, T, n]
      H_res:  [B, T, n, n]

    Based on Eq. (7) and Eq. (8) in the paper.
    """
    def __init__(self, d_model: int, cfg: MHCConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.n_streams
        self.n = n
        self.d_model = d_model
        self.d_flat = n * d_model

        self.norm = RMSNorm(self.d_flat, eps=1e-6)  # separate eps; can be config if needed

        # dynamic mapping projections (phi matrices)
        self.phi_pre = nn.Linear(self.d_flat, n, bias=False)
        self.phi_post = nn.Linear(self.d_flat, n, bias=False)
        self.phi_res = nn.Linear(self.d_flat, n * n, bias=False)

        # static biases
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))

        # gating alphas (scalars)
        self.alpha_pre = nn.Parameter(torch.tensor(cfg.alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(cfg.alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(cfg.alpha_init))

        self._init_parameters()

    def _init_parameters(self) -> None:
        """
        The paper specifies alpha init and Sinkhorn tmax/n.
        Bias init is not fully specified; we make it explicit and stable:

        - H_pre = sigmoid(H~_pre). To make initial aggregation sum to 1 (uniform),
          set b_pre = logit(1/n). Then sigmoid(b_pre)=1/n.
        - H_post = 2*sigmoid(H~_post). To make initial distribution sum to 1 (uniform),
          we want H_post_i = 1/n => 2*sigmoid(b_post)=1/n => sigmoid(b_post)=1/(2n).
        - H_res is Sinkhorn(exp(...)). To approximate identity mapping at init:
          set b_res diagonal high and off-diagonal low. After Sinkhorn, this becomes close to a permutation.
        """
        n = self.n
        if self.cfg.init_hpre == "uniform_sum1":
            p = 1.0 / n
            self.b_pre.data.fill_(math.log(p / (1 - p)))
        elif self.cfg.init_hpre == "sigmoid_half":
            self.b_pre.data.zero_()

        if self.cfg.init_hpost == "uniform_sum1":
            p = 1.0 / (2 * n)
            self.b_post.data.fill_(math.log(p / (1 - p)))

        if self.cfg.init_hres == "approx_identity":
            # diag = +2, off = -2 (tunable)
            self.b_res.data.fill_(-2.0)
            self.b_res.data.diagonal().fill_(2.0)
        elif self.cfg.init_hres == "uniform":
            self.b_res.data.zero_()

        # init linear weights
        for lin in [self.phi_pre, self.phi_post, self.phi_res]:
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, n, C = X.shape
        assert n == self.n and C == self.d_model, f"Expected X shape [B,T,{self.n},{self.d_model}]"
        x_flat = X.reshape(B, T, n * C)
        x_norm = self.norm(x_flat)

        hpre_tilde = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre  # [B,T,n]
        hpost_tilde = self.alpha_post * self.phi_post(x_norm) + self.b_post  # [B,T,n]
        hres_tilde = self.alpha_res * self.phi_res(x_norm).reshape(B, T, n, n) + self.b_res  # [B,T,n,n]

        H_pre = torch.sigmoid(hpre_tilde)  # (0,1)
        H_post = 2.0 * torch.sigmoid(hpost_tilde)  # (0,2)
        H_res = sinkhorn_knopp(
            hres_tilde,
            tmax=self.cfg.sinkhorn_tmax,
            eps=self.cfg.sinkhorn_eps,
            clamp_min=self.cfg.sinkhorn_clamp_min,
            clamp_max=self.cfg.sinkhorn_clamp_max,
        )
        return H_pre, H_post, H_res


class MHCResidual(nn.Module):
    """
    Wrap a sublayer F (attention or MLP) with mHC residual update.

    Update:
      X_next = H_res * X + H_post^T * F( H_pre * X )

    Where:
      - X is the n-stream residual: [B,T,n,C]
      - H_pre aggregates to [B,T,C]
      - F outputs [B,T,C]
      - H_post distributes back to streams [B,T,n,C]
      - H_res mixes streams [B,T,n,n] -> [B,T,n,C]
    """
    def __init__(self, d_model: int, cfg: MHCConfig, sublayer: nn.Module):
        super().__init__()
        self.mapping = MHCMapping(d_model=d_model, cfg=cfg)
        self.sublayer = sublayer

    def forward(self, X: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        H_pre, H_post, H_res = self.mapping(X)  # shapes above

        # aggregate: x_in = sum_i H_pre_i * X_i
        x_in = torch.einsum("btn,btnc->btc", H_pre, X)  # [B,T,C]

        # apply sublayer
        if attn_mask is None:
            y = self.sublayer(x_in)
        else:
            # attention sublayer expects attn_mask
            y = self.sublayer(x_in, attn_mask=attn_mask)

        # residual mixing: res = H_res @ X
        res = torch.einsum("btnm,btmc->btnc", H_res, X)  # [B,T,n,C] with n==m

        # distribute: upd_i = H_post_i * y
        upd = torch.einsum("btn,btc->btnc", H_post, y)
        return res + upd
