import torch
import torch.nn.functional as F

from llm_mhc_sdft_tttd.config import MHCConfig, ModelConfig
from llm_mhc_sdft_tttd.model.lora import (
    apply_lora,
    load_lora,
    mark_only_lora_trainable,
    merge_lora_linears,
    save_lora,
)
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_ff=64,
        max_seq_len=32,
        mhc=MHCConfig(n_streams=2),
    )


def test_lora_save_load_and_merge_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_cfg()

    base = MHCTransformerLM(cfg)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    train_model = MHCTransformerLM(cfg)
    train_model.load_state_dict(base_state, strict=True)
    apply_lora(
        train_model,
        target_module_suffixes=("q_proj", "k_proj", "v_proj", "o_proj"),
        r=4,
        alpha=8,
        dropout=0.0,
    )
    mark_only_lora_trainable(train_model)

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    opt = torch.optim.AdamW([p for p in train_model.parameters() if p.requires_grad], lr=1e-2)
    logits = train_model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    opt.step()

    with torch.no_grad():
        logits_ref = train_model(x).detach().clone()

    lora_path = tmp_path / "adapter.pt"
    save_lora(train_model, str(lora_path))

    loaded_model = MHCTransformerLM(cfg)
    loaded_model.load_state_dict(base_state, strict=True)
    apply_lora(
        loaded_model,
        target_module_suffixes=("q_proj", "k_proj", "v_proj", "o_proj"),
        r=4,
        alpha=8,
        dropout=0.0,
    )
    load_lora(loaded_model, str(lora_path))

    with torch.no_grad():
        logits_loaded = loaded_model(x)

    assert torch.allclose(logits_ref, logits_loaded, atol=1e-6, rtol=1e-5)

    merge_lora_linears(loaded_model, unload=True)
    with torch.no_grad():
        logits_merged = loaded_model(x)
    assert torch.allclose(logits_ref, logits_merged, atol=1e-5, rtol=1e-4)
