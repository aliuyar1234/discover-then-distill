import torch


def test_generate_respects_prompt_lens_and_avoids_mid_pads():
    """Regression test for a subtle but critical bug:

    If prompts in a batch have different lengths and are right-padded, generation must
    start immediately after each sample's *true* prompt length, not after the padded
    max length.

    The failure mode (old code): for shorter prompts, the last token in the padded
    tensor is PAD, so the model generates after PAD and PAD tokens remain in the middle
    of the produced sequence.
    """

    from llm_mhc_sdft_tttd.config import ModelConfig, MHCConfig
    from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM

    torch.manual_seed(0)

    cfg = ModelConfig(
        vocab_size=50,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_ff=64,
        max_seq_len=64,
        mhc=MHCConfig(n_streams=2),
    )
    m = MHCTransformerLM(cfg)

    pad_id = 0
    # two prompts, different lengths
    p0 = torch.tensor([2, 5, 6, 7, 8], dtype=torch.long)  # len 5
    p1 = torch.tensor([2, 9, 10, 11, 12, 13, 14, 15], dtype=torch.long)  # len 8

    T = 8
    x = torch.full((2, T), pad_id, dtype=torch.long)
    x[0, : p0.numel()] = p0
    x[1, : p1.numel()] = p1

    # generate enough tokens to fill the original padding region for sample 0
    out, lens = m.generate(
        x,
        max_new_tokens=3,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=None,
        prompt_lens=[int(p0.numel()), int(p1.numel())],
        pad_token_id=pad_id,
        return_lens=True,
    )

    # prompts preserved
    assert torch.equal(out[0, : p0.numel()], p0)
    assert torch.equal(out[1, : p1.numel()], p1)

    # the padding region between len(p0) and len(p1) must be filled by generated tokens,
    # not left as PAD (i.e., no PAD tokens in the middle of sample 0's sequence).
    assert (out[0, p0.numel() : p1.numel()] != pad_id).all()

    assert lens[0] == int(p0.numel()) + 3
    assert lens[1] == int(p1.numel()) + 3
