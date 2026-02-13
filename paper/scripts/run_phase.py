from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_mhc_sdft_tttd.tracking import RunTracker


PHASES = ["A", "B", "C", "D", "E", "F"]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def in_phase_range(phase: str, from_phase: str, to_phase: str) -> bool:
    a = PHASES.index(from_phase)
    b = PHASES.index(to_phase)
    x = PHASES.index(phase)
    return a <= x <= b


def abs_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


@dataclass
class StepDef:
    step_id: str
    phase: str
    description: str
    command: List[str]
    artifacts: List[str]
    log_path: str
    mirror_log_path: Optional[str] = None
    create_marker: bool = False


def build_steps(args: argparse.Namespace) -> List[StepDef]:
    py = args.python
    model_cfg = args.model_cfg
    tok_model = f"{args.tokenizer_prefix}.model"
    suite = args.suite_dir

    def step(
        step_id: str,
        phase: str,
        description: str,
        command: List[str],
        artifacts: List[str],
        mirror_log_path: Optional[str] = None,
        create_marker: bool = False,
    ) -> StepDef:
        return StepDef(
            step_id=step_id,
            phase=phase,
            description=description,
            command=command,
            artifacts=artifacts,
            log_path=str(Path(args.logs_dir) / f"{step_id}.log"),
            mirror_log_path=mirror_log_path,
            create_marker=create_marker,
        )

    steps: List[StepDef] = [
        step(
            "A1_env_pytest",
            "A",
            "Environment validation via test suite",
            [py, "-m", "pytest", "-q"],
            [str(Path(args.orchestrator_dir) / "markers" / "A1_env_pytest.ok")],
            create_marker=True,
        ),
        step(
            "A2_gen_corpus",
            "A",
            "Generate synthetic LM train/val corpus",
            [
                py,
                "paper/scripts/gen_synthetic_corpus.py",
                "--seed",
                str(args.seed),
                "--train_lines",
                str(args.train_lines),
                "--val_lines",
                str(args.val_lines),
                "--train_out",
                args.raw_train,
                "--val_out",
                args.raw_val,
            ],
            [args.raw_train, args.raw_val],
        ),
        step(
            "A3_train_tokenizer",
            "A",
            "Train SentencePiece tokenizer",
            [
                py,
                "scripts/train_tokenizer.py",
                "--input",
                args.raw_train,
                "--model_prefix",
                args.tokenizer_prefix,
                "--vocab_size",
                str(args.vocab_size),
                "--model_type",
                args.model_type,
            ],
            [f"{args.tokenizer_prefix}.model", f"{args.tokenizer_prefix}.vocab"],
        ),
        step(
            "A4_pack_train",
            "A",
            "Pack train tokens to binary format",
            [
                py,
                "scripts/prepare_data.py",
                "--tokenizer",
                tok_model,
                "--input",
                args.raw_train,
                "--output",
                args.packed_train,
                "--append_eos",
                "1",
            ],
            [args.packed_train, f"{args.packed_train}.json"],
        ),
        step(
            "A5_pack_val",
            "A",
            "Pack val tokens to binary format",
            [
                py,
                "scripts/prepare_data.py",
                "--tokenizer",
                tok_model,
                "--input",
                args.raw_val,
                "--output",
                args.packed_val,
                "--append_eos",
                "1",
            ],
            [args.packed_val, f"{args.packed_val}.json"],
        ),
        step(
            "B1_pretrain_120m",
            "B",
            "Pretrain base checkpoint",
            [
                py,
                "scripts/pretrain.py",
                "--train_bin",
                args.packed_train,
                "--val_bin",
                args.packed_val,
                "--out",
                args.pretrain_out,
                "--model",
                model_cfg,
                "--steps",
                str(args.pretrain_steps),
                "--seq_len",
                str(args.pretrain_seq_len),
                "--micro_bs",
                str(args.pretrain_micro_bs),
                "--grad_accum",
                str(args.pretrain_grad_accum),
                "--save_every",
                str(args.pretrain_save_every),
                "--eval_every",
                str(args.pretrain_eval_every),
                "--log_every",
                str(args.pretrain_log_every),
                "--device",
                args.device,
                "--resume",
                "auto",
            ],
            [
                f"{args.pretrain_out}/ckpt_latest.pt",
                f"{args.pretrain_out}/ckpt_step_{args.pretrain_steps:07d}.pt",
                f"{args.pretrain_out}/metrics.jsonl",
            ],
            mirror_log_path=f"{args.pretrain_out}/stdout.log",
        ),
        step(
            "C1_gen_discovery_tasks",
            "C",
            "Generate discovery train/heldout tasks",
            [
                py,
                "paper/scripts/gen_discovery_tasks.py",
                "--seed",
                str(args.seed),
                "--out_train",
                args.discovery_train,
                "--out_heldout",
                args.discovery_heldout,
            ],
            [args.discovery_train, args.discovery_heldout],
        ),
    ]

    def suite_step(
        step_id: str,
        description: str,
        out_root: str,
        mode: str,
        seed: int,
        objective: str = "entropic",
        reuse: int = 1,
        adaptive_beta: int = 1,
        kl_lambda: float = 0.1,
    ) -> StepDef:
        return step(
            step_id,
            "C",
            description,
            [
                py,
                "paper/run_discovery_suite.py",
                "--tasks",
                args.discovery_heldout,
                "--ckpt",
                f"{args.pretrain_out}/ckpt_latest.pt",
                "--tokenizer",
                tok_model,
                "--out_root",
                out_root,
                "--seed",
                str(seed),
                "--mode",
                mode,
                "--ttt_steps",
                str(args.ttt_steps),
                "--rollouts",
                str(args.rollouts),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--objective",
                objective,
                "--reuse",
                str(reuse),
                "--adaptive_beta",
                str(adaptive_beta),
                "--kl_lambda",
                str(kl_lambda),
                "--beta",
                str(args.beta),
                "--device",
                args.device,
                "--resume",
            ],
            [f"{out_root}/summary.jsonl"],
            mirror_log_path=f"{out_root}/stdout.log",
        )

    steps.extend(
        [
            suite_step(
                "C2_base_bestofn_seed0",
                "Heldout baseline best-of-N (seed 0)",
                f"{suite}/base_bestofn_seed0",
                mode="bestofn",
                seed=args.seed,
            ),
            suite_step(
                "C3_base_bestofn_seed1",
                "Heldout baseline best-of-N (seed 1)",
                f"{suite}/base_bestofn_seed1",
                mode="bestofn",
                seed=args.second_seed,
            ),
            suite_step(
                "C4_ttt_main_seed0",
                "Heldout TTT main condition (seed 0)",
                f"{suite}/ttt_main_seed0",
                mode="ttt",
                seed=args.seed,
                objective="entropic",
                reuse=1,
                adaptive_beta=1,
                kl_lambda=args.kl_lambda,
            ),
            suite_step(
                "C5_ttt_main_seed1",
                "Heldout TTT main condition (seed 1)",
                f"{suite}/ttt_main_seed1",
                mode="ttt",
                seed=args.second_seed,
                objective="entropic",
                reuse=1,
                adaptive_beta=1,
                kl_lambda=args.kl_lambda,
            ),
            suite_step(
                "C6_ttt_reuse0_seed0",
                "Ablation: reuse off",
                f"{suite}/ttt_reuse0_seed0",
                mode="ttt",
                seed=args.seed,
                objective="entropic",
                reuse=0,
                adaptive_beta=1,
                kl_lambda=args.kl_lambda,
            ),
            suite_step(
                "C7_ttt_adapt0_seed0",
                "Ablation: adaptive beta off",
                f"{suite}/ttt_adapt0_seed0",
                mode="ttt",
                seed=args.seed,
                objective="entropic",
                reuse=1,
                adaptive_beta=0,
                kl_lambda=args.kl_lambda,
            ),
            suite_step(
                "C8_ttt_kl0_seed0",
                "Ablation: KL shaping off",
                f"{suite}/ttt_kl0_seed0",
                mode="ttt",
                seed=args.seed,
                objective="entropic",
                reuse=1,
                adaptive_beta=1,
                kl_lambda=0.0,
            ),
        ]
    )

    if args.include_expected_ablation:
        steps.append(
            suite_step(
                "C9_ttt_expected_seed0",
                "Optional ablation: expected objective",
                f"{suite}/ttt_expected_seed0",
                mode="ttt",
                seed=args.seed,
                objective="expected",
                reuse=1,
                adaptive_beta=1,
                kl_lambda=args.kl_lambda,
            )
        )

    steps.extend(
        [
            step(
                "D1_ttt_train_seed0",
                "D",
                "Run TTT on train split for discovery->distill conversion",
                [
                    py,
                    "paper/run_discovery_suite.py",
                    "--tasks",
                    args.discovery_train,
                    "--ckpt",
                    f"{args.pretrain_out}/ckpt_latest.pt",
                    "--tokenizer",
                    tok_model,
                    "--out_root",
                    f"{suite}/ttt_train_seed0",
                    "--seed",
                    str(args.seed),
                    "--mode",
                    "ttt",
                    "--ttt_steps",
                    str(args.ttt_steps),
                    "--rollouts",
                    str(args.rollouts),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--temperature",
                    str(args.temperature),
                    "--top_p",
                    str(args.top_p),
                    "--objective",
                    "entropic",
                    "--reuse",
                    "1",
                    "--adaptive_beta",
                    "1",
                    "--kl_lambda",
                    str(args.kl_lambda),
                    "--beta",
                    str(args.beta),
                    "--device",
                    args.device,
                    "--resume",
                ],
                [f"{suite}/ttt_train_seed0/summary.jsonl"],
                mirror_log_path=f"{suite}/ttt_train_seed0/stdout.log",
            ),
            step(
                "D2_build_sdft_dataset",
                "D",
                "Convert discovered solutions to SDFT dataset",
                [
                    py,
                    "paper/scripts/build_sdft_from_suite.py",
                    "--suite_root",
                    f"{suite}/ttt_train_seed0",
                    "--tasks_jsonl",
                    args.discovery_train,
                    "--out_jsonl",
                    args.sdft_data,
                    "--min_reward",
                    str(args.min_reward),
                ],
                [args.sdft_data],
            ),
            step(
                "D3_sdft_consolidation",
                "D",
                "Run SDFT consolidation",
                [
                    py,
                    "scripts/sdft.py",
                    "--ckpt",
                    f"{args.pretrain_out}/ckpt_latest.pt",
                    "--tokenizer",
                    tok_model,
                    "--data",
                    args.sdft_data,
                    "--out",
                    args.sdft_out,
                    "--steps",
                    str(args.sdft_steps),
                    "--save_every",
                    str(args.sdft_save_every),
                    "--log_every",
                    str(args.sdft_log_every),
                    "--gate_val_bin",
                    args.packed_val,
                    "--gate_val_seq_len",
                    str(args.sdft_gate_seq_len),
                    "--gate_val_bs",
                    str(args.sdft_gate_batch_size),
                    "--device",
                    args.device,
                    "--resume",
                    "auto",
                ],
                [
                    f"{args.sdft_out}/sdft_latest.pt",
                    f"{args.sdft_out}/sdft_step_{args.sdft_steps:07d}.pt",
                    f"{args.sdft_out}/sdft_eval_report.json",
                ],
                mirror_log_path=f"{args.sdft_out}/stdout.log",
            ),
            step(
                "E1_sdft_bestofn_seed0",
                "E",
                "Heldout best-of-N with consolidated checkpoint (seed 0)",
                [
                    py,
                    "paper/run_discovery_suite.py",
                    "--tasks",
                    args.discovery_heldout,
                    "--ckpt",
                    f"{args.sdft_out}/sdft_latest.pt",
                    "--tokenizer",
                    tok_model,
                    "--out_root",
                    f"{suite}/sdft500_bestofn_seed0",
                    "--seed",
                    str(args.seed),
                    "--mode",
                    "bestofn",
                    "--ttt_steps",
                    str(args.ttt_steps),
                    "--rollouts",
                    str(args.rollouts),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--temperature",
                    str(args.temperature),
                    "--top_p",
                    str(args.top_p),
                    "--objective",
                    "entropic",
                    "--reuse",
                    "1",
                    "--adaptive_beta",
                    "1",
                    "--kl_lambda",
                    str(args.kl_lambda),
                    "--beta",
                    str(args.beta),
                    "--device",
                    args.device,
                    "--resume",
                ],
                [f"{suite}/sdft500_bestofn_seed0/summary.jsonl"],
                mirror_log_path=f"{suite}/sdft500_bestofn_seed0/stdout.log",
            ),
            step(
                "E2_sdft_bestofn_seed1",
                "E",
                "Heldout best-of-N with consolidated checkpoint (seed 1)",
                [
                    py,
                    "paper/run_discovery_suite.py",
                    "--tasks",
                    args.discovery_heldout,
                    "--ckpt",
                    f"{args.sdft_out}/sdft_latest.pt",
                    "--tokenizer",
                    tok_model,
                    "--out_root",
                    f"{suite}/sdft500_bestofn_seed1",
                    "--seed",
                    str(args.second_seed),
                    "--mode",
                    "bestofn",
                    "--ttt_steps",
                    str(args.ttt_steps),
                    "--rollouts",
                    str(args.rollouts),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--temperature",
                    str(args.temperature),
                    "--top_p",
                    str(args.top_p),
                    "--objective",
                    "entropic",
                    "--reuse",
                    "1",
                    "--adaptive_beta",
                    "1",
                    "--kl_lambda",
                    str(args.kl_lambda),
                    "--beta",
                    str(args.beta),
                    "--device",
                    args.device,
                    "--resume",
                ],
                [f"{suite}/sdft500_bestofn_seed1/summary.jsonl"],
                mirror_log_path=f"{suite}/sdft500_bestofn_seed1/stdout.log",
            ),
            step(
                "E3_retention",
                "E",
                "Compute retention perplexity delta",
                [
                    py,
                    "paper/scripts/compute_retention_ppl.py",
                    "--base_ckpt",
                    f"{args.pretrain_out}/ckpt_latest.pt",
                    "--base_cfg",
                    f"{args.pretrain_out}/model_config.json",
                    "--updated_ckpt",
                    f"{args.sdft_out}/sdft_latest.pt",
                    "--updated_cfg",
                    f"{args.sdft_out}/model_config.json",
                    "--val_bin",
                    args.packed_val,
                    "--seq_len",
                    str(args.retention_seq_len),
                    "--batch_size",
                    str(args.retention_batch_size),
                    "--max_batches",
                    str(args.retention_max_batches),
                    "--device",
                    args.device,
                    "--out",
                    f"{suite}/retention_ppl.json",
                ],
                [f"{suite}/retention_ppl.json"],
            ),
            step(
                "F1_plot_pipeline",
                "F",
                "Generate pipeline diagram figure",
                [py, "paper/plot_pipeline_diagram.py", "--fig_dir", args.fig_dir],
                [f"{args.fig_dir}/F1_pipeline.png", f"{args.fig_dir}/F1_pipeline.pdf"],
            ),
            step(
                "F2_plot_results",
                "F",
                "Generate reward/retention figures",
                [py, "paper/plot_results.py", "--runs_dir", suite, "--fig_dir", args.fig_dir],
                [f"{args.fig_dir}/F2_reward_vs_steps.png"],
            ),
        ]
    )

    return steps


def artifacts_exist(step: StepDef) -> bool:
    if not step.artifacts:
        return False
    for rel in step.artifacts:
        p = abs_path(rel)
        if not p.exists():
            return False
        if p.is_file():
            try:
                size = int(p.stat().st_size)
            except OSError:
                return False
            # Empty files are usually partial artifacts from interrupted runs.
            if size <= 0:
                return False
            # Discovery suite summaries must contain at least one completed row.
            if p.name == "summary.jsonl":
                has_row = False
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            has_row = True
                            break
                if not has_row:
                    return False
    return True


def run_command(step: StepDef, dry_run: bool) -> int:
    cmd_text = shlex.join(step.command)
    step_log = abs_path(step.log_path)
    ensure_parent(step_log)

    mirror_log: Optional[Path] = None
    if step.mirror_log_path:
        mirror_log = abs_path(step.mirror_log_path)
        ensure_parent(mirror_log)

    header = f"\n=== {iso_now()} | {step.step_id} | {cmd_text}\n"
    with open(step_log, "a", encoding="utf-8") as sf:
        sf.write(header)
        sf.flush()
        if mirror_log is not None and mirror_log != step_log:
            with open(mirror_log, "a", encoding="utf-8") as mf:
                mf.write(header)
                mf.flush()

        if dry_run:
            msg = f"[dry-run] {cmd_text}\n"
            sf.write(msg)
            sf.flush()
            if mirror_log is not None and mirror_log != step_log:
                with open(mirror_log, "a", encoding="utf-8") as mf:
                    mf.write(msg)
            return 0

        proc = subprocess.Popen(
            step.command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sf.write(line)
            sf.flush()
            if mirror_log is not None and mirror_log != step_log:
                with open(mirror_log, "a", encoding="utf-8") as mf:
                    mf.write(line)
        rc = proc.wait()

        footer = f"=== {iso_now()} | exit_code={rc} | {step.step_id}\n"
        sf.write(footer)
        sf.flush()
        if mirror_log is not None and mirror_log != step_log:
            with open(mirror_log, "a", encoding="utf-8") as mf:
                mf.write(footer)
                mf.flush()
        return rc


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Discover->Distill phases A-F with checkpoint-aware resume.")
    ap.add_argument("--from-phase", choices=PHASES, default="A")
    ap.add_argument("--to-phase", choices=PHASES, default="F")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Rerun steps even if required artifacts already exist.")
    ap.add_argument("--include-expected-ablation", action="store_true", help="Include optional expected-objective ablation.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--second-seed", type=int, default=1)
    ap.add_argument("--train-lines", type=int, default=200_000)
    ap.add_argument("--val-lines", type=int, default=10_000)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--pretrain-steps", type=int, default=5000)
    ap.add_argument("--pretrain-seq-len", type=int, default=256)
    ap.add_argument("--pretrain-micro-bs", type=int, default=4)
    ap.add_argument("--pretrain-grad-accum", type=int, default=4)
    ap.add_argument("--pretrain-log-every", type=int, default=10)
    ap.add_argument("--pretrain-eval-every", type=int, default=500)
    ap.add_argument("--pretrain-save-every", type=int, default=250)
    ap.add_argument("--ttt-steps", type=int, default=30)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--kl-lambda", type=float, default=0.1)
    ap.add_argument("--min-reward", type=float, default=0.99)
    ap.add_argument("--sdft-steps", type=int, default=500)
    ap.add_argument("--sdft-log-every", type=int, default=10)
    ap.add_argument("--sdft-save-every", type=int, default=100)
    ap.add_argument("--sdft-gate-seq-len", type=int, default=256)
    ap.add_argument("--sdft-gate-batch-size", type=int, default=4)
    ap.add_argument("--retention-seq-len", type=int, default=256)
    ap.add_argument("--retention-batch-size", type=int, default=4)
    ap.add_argument("--retention-max-batches", type=int, default=50)
    ap.add_argument("--model-cfg", default="configs/model_mhc_120m.json")
    ap.add_argument("--raw-train", default="data/raw/train.txt")
    ap.add_argument("--raw-val", default="data/raw/val.txt")
    ap.add_argument("--tokenizer-prefix", default="data/tokenizer/spm32k")
    ap.add_argument("--packed-train", default="data/packed/train.bin")
    ap.add_argument("--packed-val", default="data/packed/val.bin")
    ap.add_argument("--pretrain-out", default="runs/pretrain_120m_base")
    ap.add_argument("--suite-dir", default="runs/suite")
    ap.add_argument("--discovery-train", default="data/discovery/train.jsonl")
    ap.add_argument("--discovery-heldout", default="data/discovery/heldout.jsonl")
    ap.add_argument("--sdft-data", default="data/sdft/from_ttt_train.jsonl")
    ap.add_argument("--sdft-out", default="runs/sdft_from_ttt_steps500")
    ap.add_argument("--fig-dir", default="paper/figs")
    ap.add_argument("--orchestrator-dir", default="runs/orchestrator")
    ap.add_argument("--logs-dir", default="runs/orchestrator/logs")
    ap.add_argument("--state-path", default="runs/orchestrator/state.json")
    return ap


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    state_path = abs_path(args.state_path)
    state = load_json(state_path)
    if not state:
        state = {
            "created_at": iso_now(),
            "steps": {},
            "runs": [],
        }
    state.setdefault("steps", {})
    state.setdefault("runs", [])

    all_steps = build_steps(args)
    selected = [s for s in all_steps if in_phase_range(s.phase, args.from_phase, args.to_phase)]
    if not selected:
        print("No steps selected.")
        return

    tracker = RunTracker(
        out_dir=args.orchestrator_dir,
        run_type="phase_orchestrator",
        total_steps=len(selected),
        command=shlex.join([sys.executable, "paper/scripts/run_phase.py", *sys.argv[1:]]),
        resume=state_path.exists(),
    )
    tracker.register_artifact("state", str(state_path), required=True)

    run_row: Dict[str, Any] = {
        "started_at": iso_now(),
        "from_phase": args.from_phase,
        "to_phase": args.to_phase,
        "dry_run": args.dry_run,
        "force": args.force,
        "argv": sys.argv[1:],
    }
    state["runs"].append(run_row)
    save_json(state_path, state)

    done = 0
    try:
        for step in selected:
            step_state = state["steps"].setdefault(step.step_id, {})
            step_state["phase"] = step.phase
            step_state["description"] = step.description
            step_state["command"] = step.command
            step_state["command_text"] = shlex.join(step.command)
            step_state["artifacts"] = step.artifacts
            step_state["log_path"] = step.log_path
            step_state["mirror_log_path"] = step.mirror_log_path
            step_state["updated_at"] = iso_now()
            save_json(state_path, state)

            prev_status = str(step_state.get("status") or "")
            allow_skip_existing = prev_status in {"", "completed", "skipped_existing", "dry_run"}
            if not args.force and allow_skip_existing and artifacts_exist(step):
                step_state["status"] = "skipped_existing"
                step_state["updated_at"] = iso_now()
                save_json(state_path, state)
                done += 1
                tracker.event("step_skipped_existing", step_id=step.step_id)
                tracker.heartbeat(
                    step=done,
                    total_steps=len(selected),
                    message=f"Skipped existing: {step.step_id}",
                )
                continue

            step_state["status"] = "running"
            step_state["started_at"] = iso_now()
            step_state["attempts"] = int(step_state.get("attempts", 0)) + 1
            save_json(state_path, state)
            tracker.event("step_started", step_id=step.step_id, phase=step.phase)

            rc = run_command(step, dry_run=args.dry_run)
            step_state["exit_code"] = rc
            step_state["finished_at"] = iso_now()

            if rc != 0:
                step_state["status"] = "failed"
                save_json(state_path, state)
                tracker.finalize(
                    status="failed",
                    step=done,
                    total_steps=len(selected),
                    message=f"Step failed: {step.step_id} (exit={rc})",
                )
                raise SystemExit(rc)

            if args.dry_run:
                step_state["status"] = "dry_run"
                step_state["updated_at"] = iso_now()
                save_json(state_path, state)
                done += 1
                tracker.event("step_dry_run", step_id=step.step_id, phase=step.phase)
                tracker.heartbeat(
                    step=done,
                    total_steps=len(selected),
                    message=f"Dry-run: {step.step_id}",
                )
                continue

            if step.create_marker:
                for marker in step.artifacts:
                    marker_path = abs_path(marker)
                    ensure_parent(marker_path)
                    marker_path.write_text(
                        json.dumps({"step_id": step.step_id, "time": iso_now()}, indent=2),
                        encoding="utf-8",
                    )

            if step.artifacts and not artifacts_exist(step):
                step_state["status"] = "failed_missing_artifacts"
                step_state["missing_artifacts"] = [
                    p for p in step.artifacts if not abs_path(p).exists()
                ]
                save_json(state_path, state)
                tracker.finalize(
                    status="failed",
                    step=done,
                    total_steps=len(selected),
                    message=f"Step finished but artifacts missing: {step.step_id}",
                )
                raise SystemExit(2)

            step_state["status"] = "completed"
            step_state["updated_at"] = iso_now()
            save_json(state_path, state)

            done += 1
            tracker.event("step_completed", step_id=step.step_id, phase=step.phase)
            tracker.heartbeat(
                step=done,
                total_steps=len(selected),
                message=f"Completed: {step.step_id}",
            )

    except KeyboardInterrupt:
        tracker.finalize(
            status="paused",
            step=done,
            total_steps=len(selected),
            message="Orchestrator interrupted by user.",
        )
        raise
    except Exception as exc:
        run_row["finished_at"] = iso_now()
        run_row["status"] = "failed"
        run_row["error"] = f"{type(exc).__name__}: {exc}"
        save_json(state_path, state)
        tracker.finalize(
            status="failed",
            step=done,
            total_steps=len(selected),
            message=f"Orchestrator failed: {type(exc).__name__}: {exc}",
        )
        raise

    run_row["finished_at"] = iso_now()
    run_row["status"] = "completed"
    save_json(state_path, state)
    tracker.finalize(
        status="completed",
        step=done,
        total_steps=len(selected),
        message="Selected phases completed.",
    )
    print(f"Completed {done}/{len(selected)} steps.")
    print(f"State file: {state_path}")


if __name__ == "__main__":
    main()
