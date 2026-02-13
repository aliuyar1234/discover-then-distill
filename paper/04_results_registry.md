# Results Registry

Use one entry per run group.

## Template

```text
Run Group ID:
Date:
Code snapshot:
Command(s):
Config(s):
Dataset slice/version:
Key metrics:
- perplexity:
- probe score:
- TTT best reward:
- regression gate events:
Decision:
- keep / reject / rerun
Notes:
```

## Entries

Run Group ID: fast_paper_v1 (completed C->F)
Date: 2026-02-13
Code snapshot: local workspace (session-controlled fast profile)
Command(s): orchestrator C->F via `paper/scripts/session_control.ps1` profile `fast_paper_v1`
Config(s):
- ttt_steps=12
- rollouts=8
- max_new_tokens=16
- sdft_steps=300
- min_reward=0.95
Dataset slice/version:
- heldout: `data/discovery/heldout.jsonl` (120 tasks)
- train: `data/discovery/train.jsonl` (160 tasks)
Key metrics:
- discovery completion:
  - C2/C3 best-of-N: completed
  - C4/C5 main TTT: completed
  - C6/C7/C8/C9 ablations: completed
  - D1 train TTT: completed (160/160)
- D2 dataset conversion:
  - kept 38 / 160 train tasks at `min_reward=0.95`
- heldout mean best reward:
  - base best-of-N avg (seed0/seed1): 0.843595
  - post-SDFT best-of-N avg (seed0/seed1): 0.828814
  - gain (post-SDFT - base): -0.014781
- retention:
  - ppl_base: 161.741468
  - ppl_updated: 160.420874
  - delta_ppl: -1.320593
- figures:
  - `paper/figs_compute_v2/F1_pipeline.{pdf,png}`
  - `paper/figs_compute_v2/F2_reward_vs_steps.{pdf,png}`
  - `paper/figs_compute_v2/F3_gain_vs_retention.{pdf,png}`
  - `paper/figs_compute_v2/F4_execution_summary.{pdf,png}`
- manuscript artifact:
  - `paper/DiscoverThenDistill_ComputeMatched_v2_2026-02-13.pdf` (final polished build)
Decision:
- keep for paper artifacts and reporting; consider rerun/tuning if reward gain is required on fast profile
Notes:
- Overnight pause was user-requested after D1 completion.
- Final resume completed with `resume_count=2`; orchestrator ended `status=completed` at step `17/17`.
- Fixed Phase-F artifact mismatch by adding `--fig_dir` support to `paper/plot_pipeline_diagram.py` and passing it from `paper/scripts/run_phase.py`.
- Final paper polish pass improved figure readability, float flow, naming quality, and publication-facing PDF quality.
- Runtime diagnostics are reported as A->F composite accounting:
  - A/B from `runs/orchestrator/state.json`,
  - C/D/E/F from `runs/orchestrator_fast_v1/state.json`,
  - with explicit statement that the C-phase speed tweak does not alter A/B outputs.
