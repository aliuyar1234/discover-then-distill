PYTHON ?= python

TOKENIZER ?= data/tokenizer/spm32k.model
TRAIN_RAW ?= data/raw/train.txt
VAL_RAW ?= data/raw/val.txt
TRAIN_BIN ?= data/packed/train.bin
VAL_BIN ?= data/packed/val.bin
MODEL_CFG ?= configs/model_mhc_120m.json

PRETRAIN_OUT ?= runs/pretrain_smoke
SDFT_DATA ?= data/sdft/demo.jsonl
SDFT_OUT ?= runs/sdft_smoke
TTT_OUT ?= runs/ttt_smoke

.PHONY: test tokenizer pack pretrain_smoke sdft_smoke ttt_smoke

test:
	$(PYTHON) -m pytest -q

tokenizer:
	$(PYTHON) scripts/train_tokenizer.py \
		--input $(TRAIN_RAW) \
		--model_prefix data/tokenizer/spm32k \
		--vocab_size 32000 \
		--model_type bpe

pack:
	$(PYTHON) scripts/prepare_data.py --tokenizer $(TOKENIZER) --input $(TRAIN_RAW) --output $(TRAIN_BIN) --append_eos 1
	$(PYTHON) scripts/prepare_data.py --tokenizer $(TOKENIZER) --input $(VAL_RAW) --output $(VAL_BIN) --append_eos 1

pretrain_smoke:
	$(PYTHON) scripts/pretrain.py \
		--train_bin $(TRAIN_BIN) \
		--val_bin $(VAL_BIN) \
		--out $(PRETRAIN_OUT) \
		--model $(MODEL_CFG) \
		--steps 20 \
		--seq_len 64 \
		--micro_bs 2 \
		--grad_accum 1 \
		--save_every 20 \
		--eval_every 20 \
		--log_every 1

sdft_smoke:
	$(PYTHON) scripts/sdft.py \
		--ckpt $(PRETRAIN_OUT)/ckpt_latest.pt \
		--tokenizer $(TOKENIZER) \
		--data $(SDFT_DATA) \
		--out $(SDFT_OUT) \
		--steps 20 \
		--micro_bs 1 \
		--grad_accum 1 \
		--max_new_tokens 64 \
		--gate_every 10 \
		--gate_val_bin $(VAL_BIN) \
		--gate_val_seq_len 64 \
		--gate_val_bs 2

ttt_smoke:
	$(PYTHON) scripts/ttt_discover.py \
		--ckpt $(SDFT_OUT)/sdft_latest.pt \
		--tokenizer $(TOKENIZER) \
		--out $(TTT_OUT) \
		--target HELLO \
		--steps 10 \
		--rollouts 8 \
		--max_new_tokens 64
