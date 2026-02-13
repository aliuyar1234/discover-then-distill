import json
from pathlib import Path

import sentencepiece as spm
import torch

from llm_mhc_sdft_tttd.config import MHCConfig, ModelConfig, SDFTConfig, TTTDiscoverConfig
from llm_mhc_sdft_tttd.data.ttt_to_sdft import convert_discovery_logs_to_sdft_rows, write_sdft_jsonl
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM
from llm_mhc_sdft_tttd.training.sdft import train_sdft
from llm_mhc_sdft_tttd.training.ttt_discover import ToyStringMatchEnv, run_ttt_discover


def _train_tiny_tokenizer(tmp_path: Path) -> str:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "hello world",
                "toy ttt to sdft pipeline",
                "small tokenizer corpus",
                "self distillation continual learning",
            ]
        ),
        encoding="utf-8",
    )
    prefix = tmp_path / "spm_tiny"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=64,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    return str(prefix) + ".model"


def test_ttt_to_sdft_pipeline_completes(tmp_path):
    torch.manual_seed(0)

    tok_path = _train_tiny_tokenizer(tmp_path)

    cfg = ModelConfig(
        vocab_size=64,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_head=8,
        d_ff=64,
        max_seq_len=64,
        mhc=MHCConfig(n_streams=2),
    )
    model = MHCTransformerLM(cfg)

    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = base_dir / "ckpt_latest.pt"
    torch.save({"model": model.state_dict()}, base_ckpt)
    (base_dir / "model_config.json").write_text(cfg.to_json(), encoding="utf-8")

    ttt_out = tmp_path / "ttt"
    env = ToyStringMatchEnv(problem_description="Guess hidden target string.", target="a")
    ttt_cfg = TTTDiscoverConfig(
        out_dir=str(ttt_out),
        device="cpu",
        dtype="fp32",
        ttt_steps=1,
        rollouts_per_step=2,
        max_new_tokens=4,
        reuse_enabled=False,
        lora_rank=2,
        lora_alpha=4,
    )
    ttt_result = run_ttt_discover(
        base_ckpt=str(base_ckpt),
        tokenizer_path=tok_path,
        env=env,
        cfg=ttt_cfg,
    )
    assert "best_state" in ttt_result

    discovery_log = tmp_path / "discoveries.jsonl"
    with open(discovery_log, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "prompt": env.d,
                    "best_state": ttt_result["best_state"],
                    "best_reward": ttt_result["best_reward"],
                }
            )
            + "\n"
        )

    rows = convert_discovery_logs_to_sdft_rows([str(discovery_log)], min_reward=None)
    assert len(rows) == 1
    sdft_data = tmp_path / "from_ttt.jsonl"
    write_sdft_jsonl(str(sdft_data), rows)

    sdft_out = tmp_path / "sdft"
    sdft_cfg = SDFTConfig(
        total_steps=1,
        micro_batch_size=1,
        grad_accum_steps=1,
        max_new_tokens=4,
        device="cpu",
        dtype="fp32",
        save_every=1,
        log_every=1,
        gate_every=0,
        replay_ratio=0.0,
    )
    train_sdft(
        base_model_ckpt=str(base_ckpt),
        tokenizer_path=tok_path,
        sdft_data_path=str(sdft_data),
        out_dir=str(sdft_out),
        cfg=sdft_cfg,
    )
    assert (sdft_out / "sdft_latest.pt").exists()
