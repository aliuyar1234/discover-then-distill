# Paper Workspace

This folder contains the manuscript and experiment-facing artifacts for the current paper release.

## Canonical release

- PDF: `paper/DiscoverThenDistill_ComputeMatched_v2_2026-02-13.pdf`
- LaTeX source: `paper/latex/`
- Figures: `paper/figs_compute_v2/`

## Working context

Use these planning files in order when continuing iterations:

1. `01_scope_and_claims.md`
2. `02_experimental_plan.md`
3. `03_related_work_matrix.md`
4. `04_results_registry.md`
5. `05_writing_outline.md`
6. `06_submission_checklist.md`
7. `RESEARCH_DIRECTION.md` (persistent thesis/claim boundaries)
8. `SESSION_LOG.md` (session-by-session memory)
9. `RUNBOOK_DISCOVER_DISTILL.md` (canonical execution plan)
10. `NEXT_ACTIONS.md` (current operational queue)

Folder conventions (legacy workspace support):

- `figures/` for final figure assets used in the paper
- `tables/` for final table sources and exports
- `artifacts/` for paper-ready evaluation artifacts and frozen snapshots
- `appendix/` for appendix drafts and extra analyses

Current canonical manuscript workspace remains `paper/latex/`.

Operational monitoring helpers:

- `paper/scripts/run_status.py` - live heartbeat/progress/ETA view across tracked runs.
- `paper/scripts/artifact_tracker.py` - required artifact completeness report from run trackers.
- `paper/scripts/run_phase.py` - phase orchestrator (A->F) with artifact-aware resume.
