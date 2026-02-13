# Research Direction (Persistent)

Last updated: 2026-02-12

## Researcher

- Name: Ali Uyar
- Affiliation: Independent Researcher

## Decision

- `DECISION=GO`

## Core Thesis (Falsifiable)

Test-time LoRA adaptation with entropic-utility REINFORCE and archive reuse produces higher verified reward than compute-matched best-of-N sampling, and SDFT consolidation turns discovered solutions into persistent gains on future tasks with limited retention loss under a single-GPU budget.

## Paper Scope

- Primary direction: `Discover->Distill` (TTT discovery to SDFT consolidation).
- Paper format target: workshop/arXiv-style concise manuscript.
- Claims are limited to verifiable, compute-bounded synthetic discovery tasks unless broader evidence is added.

## Primary Contributions (Claim-Safe)

1. A minimal Discover->Distill pipeline with strict separation of ephemeral test-time adapters from persistent base updates.
2. A compute-bounded empirical protocol showing effects of entropic objective, adaptive beta, and archive reuse in TTT.
3. A consolidation analysis showing gain/retention tradeoff after SDFT distillation of discovered solutions.

## Risks To Control

- Toy environment credibility risk.
- Missing canonical per-step metrics logs for plotting.
- Over-claiming beyond the implemented and measured setup.

## Required Experimental Outputs

- Main: reward-vs-steps and final reward comparisons vs compute-matched best-of-N.
- Ablations: reuse off, adaptive-beta off, KL penalty off, expected-vs-entropic objective (optional but recommended).
- Consolidation: post-SDFT held-out gain and retention metric (perplexity delta and/or untouched family).

## Session Memory Rule

Each working session must append a status update to `paper/SESSION_LOG.md` with:

- what was changed
- what was run
- what remains next
