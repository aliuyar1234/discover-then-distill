import torch
from llm_mhc_sdft_tttd.model.mhc import sinkhorn_knopp


def test_sinkhorn_doubly_stochastic():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 4)
    P = sinkhorn_knopp(x, tmax=50)
    # rows and cols sum ~1
    row = P.sum(dim=-1)
    col = P.sum(dim=-2)
    assert torch.allclose(row, torch.ones_like(row), atol=1e-2, rtol=1e-2)
    assert torch.allclose(col, torch.ones_like(col), atol=1e-2, rtol=1e-2)
    # positivity
    assert (P >= 0).all()
