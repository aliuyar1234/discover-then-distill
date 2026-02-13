# Audit Report — LLM_MHC_SDFT_TTTD_Blueprint

This report documents the **strict audit** performed on the delivered blueprint zip, the **issues found**, and the **patches applied** to produce a consistent, runnable handoff for Codex.

## Scope of the audit

I audited the repository for:

1. **Import-time correctness** (no errors just by importing the package).
2. **End-to-end “smoke path” consistency** with the instructions in `docs/CODEX_HANDOFF.md`.
3. **Internal consistency** between:
   - `docs/CODEX_HANDOFF.md` code listings
   - actual code under `src/`
   - CLI scripts under `scripts/`
4. **Cross-platform robustness** (avoid relying on filesystem symlinks).
5. **Algorithmic correctness traps** that would cause silent drift (e.g., tokenization roundtrip mismatches).

## How the audit was executed

- Unzipped the provided archive.
- Ran the repository tests via:

```bash
PYTHONPATH=src pytest -q
```

- Performed static consistency checks by grepping for known pitfalls (dataclass defaults, collate_fn, checkpoint paths).
- Verified that the “smoke” phases in `docs/CODEX_HANDOFF.md` are achievable without manual edits.

## Issues found (and why they matter)

### 1) **Dataclass error in `ModelConfig` (hard import failure)**

**Problem:** `ModelConfig` used a mutable dataclass instance as a default (`mhc: MHCConfig = MHCConfig()`).

**Impact:** Importing the package raises:

> `ValueError: mutable default <class '...MHCConfig'> for field mhc is not allowed: use default_factory`

This blocks *every* run path (tests, scripts, training).

**Fix applied:**

- Updated `src/llm_mhc_sdft_tttd/config.py` to:

```python
mhc: MHCConfig = field(default_factory=MHCConfig)
```

- Updated the mirrored listing in `docs/CODEX_HANDOFF.md`.

---

### 2) **SDFT DataLoader would crash (missing `collate_fn`)**

**Problem:** `SDFTJsonlDataset` yields `SDFTExample` dataclass instances. PyTorch’s default collate cannot collate arbitrary dataclass objects.

**Impact:** Running `scripts/sdft.py` would crash on the first batch.

**Fix applied:**

- Added a top-level picklable function `identity_collate` in:

`src/llm_mhc_sdft_tttd/data/sdft_dataset.py`

- Updated `train_sdft` to pass:

```python
collate_fn=identity_collate
```

- Updated `docs/CODEX_HANDOFF.md` accordingly.
- Added a regression test `tests/test_sdft_dataloader.py`.

---

### 3) **TTT-Discover used decode→encode roundtrip for logprobs (silent mismatch)**

**Problem:** The TTT loop decoded sampled token ids to text, then re-encoded the text and used the re-encoded ids for computing `logp_theta`/`logp_base`.

**Impact:** `decode(encode(x))` is *not guaranteed* to be identity under SentencePiece (whitespace normalization, special tokens, etc.). This can cause silent objective drift where the optimizer updates on a different sequence than the one that produced the reward.

**Fix applied:**

- Updated `src/llm_mhc_sdft_tttd/training/ttt_discover.py` to compute logprobs directly on the token ids returned by `model.generate`:

```python
full_tensor = gen
logp_theta = sequence_logprob(model, full_tensor, ctx_lens)
logp_base  = sequence_logprob(base_model, full_tensor, ctx_lens)
```

- Updated the mirrored code in `docs/CODEX_HANDOFF.md`.

---

### 4) **SDFT step semantics were inconsistent + smoke doc was unachievable**

**Problem (a):** `SDFTConfig.total_steps` was effectively counting *micro-batches*, not optimizer steps.

**Problem (b):** Phase 7 smoke run sets `--steps 200`, but `SDFTConfig.save_every=500` and the code only saved on multiples of `save_every`, so no checkpoint would be produced.

**Impact:** The doc’s DoD check (`sdft_step_0000200.pt exists`) would fail.

**Fix applied:**

- Re-defined `total_steps`, `log_every`, `save_every` as **optimizer steps** (matching pretrain).
- Introduced `micro_step` counter for accumulation.
- Added a **final checkpoint write** and a stable `sdft_latest.pt` so short runs always produce artifacts.
- Updated config comments and the `docs/CODEX_HANDOFF.md` phase instructions.

---

### 5) **Pretrain smoke doc was unachievable (no checkpoint at 20 steps) + `ckpt_latest` was symlink-only**

**Problem (a):** Phase 5 uses `--steps 20` but `PretrainConfig.save_every=200`. No checkpoint would be produced.

**Problem (b):** `ckpt_latest.pt` was created only as a symlink (best-effort). On some systems (Windows, restricted FS), this yields **no `ckpt_latest.pt` file**, breaking follow-up commands.

**Impact:** The smoke run instructions would fail.

**Fix applied:**

- Made `save_checkpoint` always write a concrete `ckpt_latest.pt` file.
- Added a **final checkpoint write** at the end of training.
- Updated Phase 5 in `docs/CODEX_HANDOFF.md` to assert:
  - `ckpt_latest.pt` exists
  - `ckpt_step_0000020.pt` exists for `--steps 20`
- Updated code listing in `docs/CODEX_HANDOFF.md`.

---

### 6) **Missing `__init__.py` in subpackages (packaging / discovery risk)**

**Problem:** Subdirectories `data/`, `model/`, `training/`, and `eval/` lacked `__init__.py`.

**Impact:** Some packaging and tooling (and `setuptools.find_packages`) may omit those subpackages or degrade IDE/tooling behavior.

**Fix applied:** Added minimal `__init__.py` files to:

- `src/llm_mhc_sdft_tttd/data/__init__.py`
- `src/llm_mhc_sdft_tttd/model/__init__.py`
- `src/llm_mhc_sdft_tttd/training/__init__.py`
- `src/llm_mhc_sdft_tttd/eval/__init__.py`

---

### 7) **Generation was incorrect for right-padded prompts (SDFT would silently break)**

**Problem:** `MHCTransformerLM.generate()` previously assumed that `input_ids` had **no padding** and always sampled from the logits at the **last column** (`logits[:, -1]`).

When prompts in a batch have different lengths and are **right-padded** (as we do in SDFT and potentially in TTT), the last column can be the **PAD token** for shorter prompts. This causes two compounding errors:

1. The model generates **after PAD** instead of after the true prompt.
2. PAD tokens remain **in the middle** of the produced sequence (between prompt and generated tokens).

**Impact:** The SDFT loop becomes wrong even though it may *not crash*:

- the on-policy trajectories are sampled from the wrong context
- the student/teacher logits are aligned to a sequence that includes spurious PADs
- continual learning updates drift silently

**Fix applied:**

- Replaced `MHCTransformerLM.generate()` with a version that supports right-padded prompt batches via:
  - `prompt_lens` (true prompt lengths)
  - writing next tokens at each sample’s current length (not at the padded end)
  - optionally returning per-sample output lengths (`return_lens=True`)
  - forbidding PAD token generation by default

- Updated:
  - `training/sdft.py` to call `generate(..., prompt_lens=..., return_lens=True)` and slice generations using `gen_lens[i]`
  - `training/ttt_discover.py` to do the same and to compute logprobs only up to each sequence length

- Added regression test: `tests/test_generate_padding.py`.

---

### 8) **Repo usability depended on editable install (fragile in restricted environments)**

**Problem:** Several scripts (`scripts/pretrain.py`, `scripts/prepare_data.py`, `scripts/sdft.py`, `scripts/ttt_discover.py`) import `llm_mhc_sdft_tttd` directly. If `pip install -e .[dev]` fails or is blocked (some CI sandboxes, locked-down containers), these scripts would fail with `ModuleNotFoundError`.

**Impact:** Codex (or a human) could be blocked before even reaching the smoke tests.

**Fix applied:**

- Added `scripts/_bootstrap.py` which inserts `<repo_root>/src` into `sys.path`.
- Updated the scripts above to call:

```python
from _bootstrap import bootstrap
bootstrap()
```

- Added `tests/conftest.py` that adds `<repo_root>/src` to `sys.path` so `pytest -q` works even without editable install.

## Post-patch verification checklist

Codex (or a human) should validate the patched repository via:

1) Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]

# If editable install is unavailable, you can still proceed from source:
#   export PYTHONPATH=$PWD/src
# (scripts and tests are bootstrapped to work this way.)
```

2) Run tests:

```bash
pytest -q
```

3) Phase 5 pretrain smoke:

```bash
python scripts/pretrain.py \
  --train_bin data/packed/train.bin \
  --val_bin data/packed/val.bin \
  --out runs/pretrain_smoke \
  --model configs/model_mhc_120m.json \
  --steps 20 \
  --seq_len 64 \
  --micro_bs 2 \
  --grad_accum 1

ls -lah runs/pretrain_smoke/ckpt_latest.pt
ls -lah runs/pretrain_smoke/ckpt_step_0000020.pt
```

4) Phase 7 SDFT smoke:

```bash
python scripts/sdft.py \
  --ckpt runs/pretrain_smoke/ckpt_latest.pt \
  --tokenizer data/tokenizer/spm32k.model \
  --data data/sdft/demo.jsonl \
  --out runs/sdft_smoke \
  --steps 200

ls -lah runs/sdft_smoke/sdft_latest.pt
ls -lah runs/sdft_smoke/sdft_step_0000200.pt
```

5) TTT smoke:

```bash
python scripts/ttt_discover.py \
  --ckpt runs/sdft_smoke/sdft_latest.pt \
  --tokenizer data/tokenizer/spm32k.model \
  --out runs/ttt_smoke \
  --steps 10
```

## Notes / non-goals

- This blueprint remains a **research-grade** minimal implementation intended for a single workstation. It is designed for correctness and reproducibility rather than maximum throughput.
- The toy environment in TTT is deliberately simple; swap it out with a real environment / tool-usage harness as described in later phases.
