# Discover->Distill Runbook (Canonical)

This file is the execution contract for the paper direction.  
Goal: produce all artifacts needed for the Discover->Distill paper without ambiguity.

## Scope

- `DECISION=GO`
- Thesis: test-time LoRA adaptation (entropic objective + reuse) should beat compute-matched best-of-N, and SDFT consolidation should preserve gains with bounded retention loss.
- Compute budget target: ~24 GPU hours.

## Status Convention

- `[ ]` not started
- `[~]` in progress
- `[x]` complete

## Run Hygiene, Heartbeat, And Resume

- Long runs now emit structured tracker files under each run dir:
  - `_run_tracker/heartbeat.json` (live progress + ETA)
  - `_run_tracker/events.jsonl` (timeline)
  - `_run_tracker/artifacts.json` (required vs present artifacts)
- Live monitor:

```bash
python paper/scripts/run_status.py --root runs --watch 15
```

- Artifact completeness report:

```bash
python paper/scripts/artifact_tracker.py --root runs
```

- Pause safely: press `Ctrl+C` once and wait for a graceful pause checkpoint message.
- Resume commands:
  - pretrain: add `--resume auto`
  - SDFT: add `--resume auto`
  - discovery suite: add `--resume` (skips completed task dirs and resumes unfinished work)

Unified orchestration command (A->F sequential, checkpoint-aware resume):

```bash
python paper/scripts/run_phase.py --from-phase A --to-phase F --include-expected-ablation
```

Resume from a later phase if needed:

```bash
python paper/scripts/run_phase.py --from-phase C --to-phase F --include-expected-ablation
```

---

## Phase A: Setup And Data

### A1. Environment

- [ ] Create env and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
python -m pytest -q
```

Expected: tests pass.

### A2. Generate synthetic LM corpus

- [ ] Generate train/val text:

```bash
mkdir -p data/raw
python paper/scripts/gen_synthetic_corpus.py \
  --seed 0 \
  --train_lines 200000 \
  --val_lines 10000 \
  --train_out data/raw/train.txt \
  --val_out data/raw/val.txt
```

### A3. Tokenizer and packed bins

- [ ] Train tokenizer:

```bash
python scripts/train_tokenizer.py \
  --input data/raw/train.txt \
  --model_prefix data/tokenizer/spm32k \
  --vocab_size 32000 \
  --model_type bpe
```

- [ ] Pack bins:

```bash
python scripts/prepare_data.py \
  --tokenizer data/tokenizer/spm32k.model \
  --input data/raw/train.txt \
  --output data/packed/train.bin \
  --append_eos 1

python scripts/prepare_data.py \
  --tokenizer data/tokenizer/spm32k.model \
  --input data/raw/val.txt \
  --output data/packed/val.bin \
  --append_eos 1
```

---

## Phase B: Base Model

### B1. Pretrain base checkpoint (120M)

- [ ] Run pretraining:

```bash
python scripts/pretrain.py \
  --train_bin data/packed/train.bin \
  --val_bin data/packed/val.bin \
  --out runs/pretrain_120m_base \
  --model configs/model_mhc_120m.json \
  --steps 5000 \
  --seq_len 256 \
  --micro_bs 4 \
  --grad_accum 4 | tee runs/pretrain_120m_base/stdout.log
```

Required artifacts:

- `runs/pretrain_120m_base/ckpt_latest.pt`
- `runs/pretrain_120m_base/ckpt_step_0005000.pt`
- `runs/pretrain_120m_base/model_config.json`
- `runs/pretrain_120m_base/pretrain_config.json`
- `runs/pretrain_120m_base/metrics.jsonl`

---

## Phase C: Discovery Suite

### C1. Generate discovery tasks

- [ ] Create train/heldout discovery tasks:

```bash
python paper/scripts/gen_discovery_tasks.py \
  --seed 0 \
  --out_train data/discovery/train.jsonl \
  --out_heldout data/discovery/heldout.jsonl
```

### C2. Held-out evaluation suite (base vs TTT)

- [ ] Baseline best-of-N (seed 0,1):

```bash
for SEED in 0 1; do
  python paper/run_discovery_suite.py \
    --tasks data/discovery/heldout.jsonl \
    --ckpt runs/pretrain_120m_base/ckpt_latest.pt \
    --tokenizer data/tokenizer/spm32k.model \
    --out_root runs/suite/base_bestofn_seed${SEED} \
    --seed ${SEED} \
    --mode bestofn \
    --ttt_steps 30 --rollouts 32 --max_new_tokens 32 \
    | tee runs/suite/base_bestofn_seed${SEED}/stdout.log
done
```

- [ ] TTT main (seed 0,1):

```bash
for SEED in 0 1; do
  python paper/run_discovery_suite.py \
    --tasks data/discovery/heldout.jsonl \
    --ckpt runs/pretrain_120m_base/ckpt_latest.pt \
    --tokenizer data/tokenizer/spm32k.model \
    --out_root runs/suite/ttt_main_seed${SEED} \
    --seed ${SEED} \
    --mode ttt \
    --ttt_steps 30 --rollouts 32 --max_new_tokens 32 \
    --objective entropic \
    --reuse 1 --adaptive_beta 1 --kl_lambda 0.1 \
    | tee runs/suite/ttt_main_seed${SEED}/stdout.log
done
```

- [ ] Ablations:

```bash
# reuse off
python paper/run_discovery_suite.py ... --out_root runs/suite/ttt_reuse0_seed0 --mode ttt --reuse 0 --adaptive_beta 1 --kl_lambda 0.1
# adaptive beta off
python paper/run_discovery_suite.py ... --out_root runs/suite/ttt_adapt0_seed0 --mode ttt --reuse 1 --adaptive_beta 0 --kl_lambda 0.1
# KL off
python paper/run_discovery_suite.py ... --out_root runs/suite/ttt_kl0_seed0 --mode ttt --reuse 1 --adaptive_beta 1 --kl_lambda 0.0
# optional expected objective
python paper/run_discovery_suite.py ... --out_root runs/suite/ttt_expected_seed0 --mode ttt --objective expected --reuse 1 --adaptive_beta 1 --kl_lambda 0.1
```

Required artifact pattern per run root:

- `runs/suite/<run_name>/summary.jsonl`
- `runs/suite/<run_name>/<task_id>/result.json`
- for TTT modes: `runs/suite/<run_name>/<task_id>/metrics.jsonl` and `adapter_step_*/lora.pt`

---

## Phase D: Distillation (Discover->Distill)

### D1. Run TTT on train split for demo generation

- [ ] Generate TTT train runs:

```bash
python paper/run_discovery_suite.py \
  --tasks data/discovery/train.jsonl \
  --ckpt runs/pretrain_120m_base/ckpt_latest.pt \
  --tokenizer data/tokenizer/spm32k.model \
  --out_root runs/suite/ttt_train_seed0 \
  --seed 0 \
  --mode ttt \
  --ttt_steps 30 --rollouts 32 --max_new_tokens 32 \
  --objective entropic \
  --reuse 1 --adaptive_beta 1 --kl_lambda 0.1 \
  | tee runs/suite/ttt_train_seed0/stdout.log
```

### D2. Convert discovered solutions to SDFT dataset

- [ ] Convert with reward filtering:

```bash
python paper/scripts/build_sdft_from_suite.py \
  --suite_root runs/suite/ttt_train_seed0 \
  --tasks_jsonl data/discovery/train.jsonl \
  --out_jsonl data/sdft/from_ttt_train.jsonl \
  --min_reward 0.99
```

### D3. Run SDFT consolidation

- [ ] Consolidate:

```bash
python scripts/sdft.py \
  --ckpt runs/pretrain_120m_base/ckpt_latest.pt \
  --tokenizer data/tokenizer/spm32k.model \
  --data data/sdft/from_ttt_train.jsonl \
  --out runs/sdft_from_ttt_steps500 \
  --steps 500 \
  --gate_val_bin data/packed/val.bin \
  --gate_val_seq_len 256 \
  --gate_val_bs 4 \
  | tee runs/sdft_from_ttt_steps500/stdout.log
```

Required artifacts:

- `runs/sdft_from_ttt_steps500/sdft_latest.pt`
- `runs/sdft_from_ttt_steps500/sdft_step_0000500.pt`
- `runs/sdft_from_ttt_steps500/metrics.jsonl`
- `runs/sdft_from_ttt_steps500/sdft_eval_report.json`

---

## Phase E: Post-Consolidation Evaluation

### E1. Held-out best-of-N with consolidated checkpoint

- [ ] Run for seed 0,1:

```bash
for SEED in 0 1; do
  python paper/run_discovery_suite.py \
    --tasks data/discovery/heldout.jsonl \
    --ckpt runs/sdft_from_ttt_steps500/sdft_latest.pt \
    --tokenizer data/tokenizer/spm32k.model \
    --out_root runs/suite/sdft500_bestofn_seed${SEED} \
    --seed ${SEED} \
    --mode bestofn \
    --ttt_steps 30 --rollouts 32 --max_new_tokens 32 \
    | tee runs/suite/sdft500_bestofn_seed${SEED}/stdout.log
done
```

### E2. Retention (perplexity delta)

- [ ] Compute and save retention:

```bash
python paper/scripts/compute_retention_ppl.py \
  --base_ckpt runs/pretrain_120m_base/ckpt_latest.pt \
  --base_cfg runs/pretrain_120m_base/model_config.json \
  --updated_ckpt runs/sdft_from_ttt_steps500/sdft_latest.pt \
  --updated_cfg runs/sdft_from_ttt_steps500/model_config.json \
  --val_bin data/packed/val.bin \
  --seq_len 256 --batch_size 4 --max_batches 50 \
  --device cuda \
  --out runs/suite/retention_ppl.json
```

---

## Phase F: Figures And Manuscript

### F1. Generate figures

- [ ] Pipeline figure:

```bash
python paper/plot_pipeline_diagram.py
```

- [ ] Result figures:

```bash
python paper/plot_results.py --runs_dir runs/suite --fig_dir paper/figs
```

### F2. Fill paper evidence tables

- [ ] Update `paper/04_results_registry.md`
- [ ] Fill LaTeX results section/table values in `paper/latex/sections/05_results.tex`
- [ ] Add final references in `paper/latex/references.bib`

---

## Budget Guidance (~24 GPU Hours)

- Base pretrain: 6-10h
- Discovery suite (heldout + ablations): 8-12h
- SDFT consolidation variants: 2-6h
- Final evaluation + figures: 1-2h

---

## Claim-Safety Guardrails

Do not claim:

- SOTA on real-world external benchmarks.
- mHC causal effect without dedicated on/off architecture evidence.
- zero forgetting guarantees.

Keep claims scoped to this verifiable synthetic suite and reported budgets.
