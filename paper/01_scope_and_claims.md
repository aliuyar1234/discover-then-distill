# Scope And Claims

## Working Title

- Discover->Distill: Test-Time LoRA Discovery with Entropic Utility and Self-Distillation Consolidation

## Candidate Venues

- Primary: workshop track or arXiv-first release
- Backup: systems/continual-learning workshop venues

## Problem Statement

- We target the conflict between aggressive per-instance test-time adaptation and long-term checkpoint stability.
- Existing implementations often blur ephemeral test-time updates and persistent model updates.
- We need a reproducible single-GPU pipeline that supports both verified-reward discovery and continual consolidation.

## Central Hypothesis

- If we combine `mHC + SDFT + TTT-Discover` with strict separation (LoRA-only test-time updates, SDFT-only persistence), then:
- test-time adaptation should outperform compute-matched best-of-N sampling on verified reward,
- and consolidation should preserve part of those gains with limited retention loss.

## Intended Contributions

1. Method/system contribution: an explicit Discover->Distill policy with no-drift separation.
2. Implementation contribution: a minimal, end-to-end, reproducible codebase and artifact protocol.
3. Empirical contribution: reward dynamics + gain/retention tradeoff under bounded compute.

## Non-Claims

- No claim of state-of-the-art on broad external benchmarks.
- No claim that mHC is isolated as the causal source of TTT stability.
- No guarantee of zero forgetting from consolidation.

## Acceptance Criteria For Internal Go/No-Go

- Minimum empirical threshold: TTT main condition beats compute-matched best-of-N on held-out mean reward.
- Required ablations: reuse off, adaptive-beta off, KL shaping off, expected-vs-entropic objective.
- Required robustness checks: multi-seed (>=2) discovery runs + retention metric after consolidation.
