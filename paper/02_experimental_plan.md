# Experimental Plan

## Primary Questions

1. Does the integrated method outperform relevant baselines?
2. Which component contributes what (`mHC`, `SDFT`, `TTT`)?
3. Are gains stable under regression gates and replay?

## Experiment Matrix

| Exp ID | Objective | Setup | Metrics | Status | Notes |
|---|---|---|---|---|---|
| E00 | Sanity smoke | tiny config | run completion | planned | |
| E01 | Base pretrain quality | base LM only | ppl, probe score | planned | |
| E02 | +mHC effect | mHC vs no mHC | ppl, stability | planned | |
| E03 | +SDFT continual | forgetting/regression | ppl drift, probe drift | planned | |
| E04 | +TTT on toy | reward improvement | best reward | planned | |
| E05 | TTT->SDFT consolidation | post-consolidation quality | ppl, probes, reward retention | planned | |

## Mandatory Ablations

- Remove mHC, keep others fixed.
- Remove replay and/or gates in SDFT.
- Disable adaptive beta in TTT.
- Disable LoRA-only policy (analysis only; do not use for production policy).

## Reproducibility Requirements

- Fixed seeds per experiment.
- Config snapshots stored with each run.
- Exact command and code revision stored with results.
