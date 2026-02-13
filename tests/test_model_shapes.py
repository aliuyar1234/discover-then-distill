import torch
from llm_mhc_sdft_tttd.config import ModelConfig, MHCConfig
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM


def test_forward_shapes():
    cfg = ModelConfig(
        vocab_size=1000,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_head=32,
        d_ff=256,
        max_seq_len=64,
        mhc=MHCConfig(n_streams=4),
    )
    model = MHCTransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size)
