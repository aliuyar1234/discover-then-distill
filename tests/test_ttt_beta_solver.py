import math

import torch

from llm_mhc_sdft_tttd.training.ttt_discover import _kl_q_u, solve_beta_by_kl


def test_solve_beta_by_kl_hits_target_kl():
    rewards = torch.tensor([0.1, 0.5, 0.2, 1.1, -0.3], dtype=torch.float32)
    gamma = math.log(2.0)
    beta = solve_beta_by_kl(rewards, gamma=gamma, max_beta=1000.0, iters=50)
    q = torch.softmax(beta * (rewards - rewards.max()), dim=0)
    kl = _kl_q_u(q).item()
    assert abs(kl - gamma) < 1e-3


def test_solve_beta_by_kl_respects_nonnegative_beta():
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float32)
    beta = solve_beta_by_kl(rewards, gamma=0.1)
    assert beta >= 0.0
