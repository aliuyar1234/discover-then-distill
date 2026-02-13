from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np
import torch

# bootstrap `<repo_root>/src`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_mhc_sdft_tttd.config import ModelConfig, TTTDiscoverConfig
from llm_mhc_sdft_tttd.data.tokenizer import SpmTokenizer
from llm_mhc_sdft_tttd.model.transformer import MHCTransformerLM
from llm_mhc_sdft_tttd.tracking import RunTracker
from llm_mhc_sdft_tttd.training.ttt_discover import ToyStringMatchEnv, run_ttt_discover


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_model_cfg(ckpt_path: str, model_cfg: str | None) -> ModelConfig:
    if model_cfg is not None:
        with open(model_cfg, "r", encoding="utf-8") as f:
            return ModelConfig.from_json(f.read())
    cfg_path = Path(ckpt_path).parent / "model_config.json"
    if not cfg_path.exists():
        raise ValueError(f"model_config.json not found next to checkpoint: {cfg_path}")
    return ModelConfig.from_json(cfg_path.read_text(encoding="utf-8"))


def load_model(ckpt_path: str, cfg: ModelConfig, device: str) -> MHCTransformerLM:
    model = MHCTransformerLM(cfg).to(device)
    payload = torch.load(ckpt_path, map_location="cpu")
    sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def best_of_n(
    model: MHCTransformerLM,
    tok: SpmTokenizer,
    problem: str,
    target: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    env = ToyStringMatchEnv(problem_description=problem, target=target)
    prompt = (
        "### Problem\n"
        f"{problem}\n\n"
        "### Current best solution (state)\n"
        "<empty>\n\n"
        "### Context (past attempts / hints)\n"
        "\n\n"
        "### Task\n"
        "Propose an improved solution. Output ONLY the solution (state) as plain text.\n"
    )

    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0).repeat(n, 1)
    ctx_lens = [len(ids)] * n

    gen, gen_lens = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_id(),
        min_new_tokens=1,
        prompt_lens=ctx_lens,
        pad_token_id=tok.pad_id(),
        return_lens=True,
        forbid_token_ids=[tok.pad_id()],
    )

    best_reward = float("-inf")
    best_state = ""
    for i in range(n):
        out = gen[i, ctx_lens[i] : int(gen_lens[i])].tolist()
        s = tok.decode(out).strip()
        r = float(env.reward(s))
        if r > best_reward:
            best_reward = r
            best_state = s
    return {"best_reward": float(best_reward), "best_state": best_state}


def read_tasks(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="jsonl with {task_id, problem, target}")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model_cfg", default=None)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["bestofn", "ttt"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ttt_steps", type=int, default=30)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    # ablations
    ap.add_argument("--objective", choices=["entropic", "expected"], default="entropic")
    ap.add_argument("--reuse", type=int, default=1)
    ap.add_argument("--adaptive_beta", type=int, default=1)
    ap.add_argument("--kl_lambda", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume run by skipping tasks that already have <out_root>/<task_id>/result.json.",
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    set_all_seeds(args.seed)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = read_tasks(args.tasks)
    summary_path = out_root / "summary.jsonl"
    tracker = RunTracker(
        out_dir=str(out_root),
        run_type="discovery_suite",
        total_steps=len(tasks),
        command=" ".join(sys.argv),
        resume=args.resume,
    )
    tracker.register_artifact("summary", str(summary_path), required=True)

    tok = SpmTokenizer(args.tokenizer)
    model_cfg = load_model_cfg(args.ckpt, args.model_cfg)
    base_model = None
    if args.mode == "bestofn":
        base_model = load_model(args.ckpt, model_cfg, device=device)

    done = 0
    mode = "a" if (args.resume and summary_path.exists()) else "w"
    written_task_ids = set()
    if args.resume and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as sf_old:
            for line in sf_old:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "task_id" in row:
                    written_task_ids.add(str(row["task_id"]))
    try:
        with open(summary_path, mode, encoding="utf-8") as sf:
            for t in tasks:
                tid = str(t["task_id"])
                problem = str(t["problem"])
                target = str(t["target"])

                run_dir = out_root / tid
                run_dir.mkdir(parents=True, exist_ok=True)
                result_file = run_dir / "result.json"

                if args.resume and result_file.exists():
                    if tid not in written_task_ids:
                        existing = json.loads(result_file.read_text(encoding="utf-8"))
                        sf.write(json.dumps({
                            "task_id": tid,
                            "mode": args.mode,
                            "seed": args.seed,
                            "ttt_steps": args.ttt_steps,
                            "rollouts": args.rollouts,
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "objective": args.objective,
                            "reuse": args.reuse,
                            "adaptive_beta": args.adaptive_beta,
                            "kl_lambda": args.kl_lambda,
                            "beta": args.beta,
                            "best_reward": float(existing["best_reward"]),
                            "recovered_from_result": True,
                        }) + "\n")
                        sf.flush()
                        written_task_ids.add(tid)
                    done += 1
                    tracker.heartbeat(
                        step=done,
                        total_steps=len(tasks),
                        message=f"Skipped completed task {tid}",
                    )
                    continue

                tracker.heartbeat(
                    step=done,
                    total_steps=len(tasks),
                    message=f"Running task {tid}",
                    metrics={"current_task": tid},
                    force=True,
                )

                if args.mode == "bestofn":
                    assert base_model is not None
                    # compute-match sample budget with TTT.
                    n = int(args.ttt_steps * args.rollouts)
                    result = best_of_n(
                        model=base_model,
                        tok=tok,
                        problem=problem,
                        target=target,
                        n=n,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
                else:
                    set_all_seeds(args.seed)
                    env = ToyStringMatchEnv(problem_description=problem, target=target)
                    cfg = TTTDiscoverConfig(
                        out_dir=str(run_dir),
                        device=device,
                        ttt_steps=args.ttt_steps,
                        rollouts_per_step=args.rollouts,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        objective=args.objective,
                        reuse_enabled=bool(args.reuse),
                        adaptive_beta=bool(args.adaptive_beta),
                        kl_penalty_lambda=float(args.kl_lambda),
                        beta=float(args.beta),
                    )
                    result = run_ttt_discover(
                        base_ckpt=args.ckpt,
                        tokenizer_path=args.tokenizer,
                        env=env,
                        cfg=cfg,
                        model_cfg=model_cfg,
                        resume=args.resume,
                        tracker_command=" ".join(sys.argv),
                    )

                sf.write(json.dumps({
                    "task_id": tid,
                    "mode": args.mode,
                    "seed": args.seed,
                    "ttt_steps": args.ttt_steps,
                    "rollouts": args.rollouts,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "objective": args.objective,
                    "reuse": args.reuse,
                    "adaptive_beta": args.adaptive_beta,
                    "kl_lambda": args.kl_lambda,
                    "beta": args.beta,
                    "best_reward": float(result["best_reward"]),
                }) + "\n")
                sf.flush()
                written_task_ids.add(tid)
                done += 1
                tracker.heartbeat(
                    step=done,
                    total_steps=len(tasks),
                    metrics={"last_task": tid, "best_reward": float(result["best_reward"])},
                )
    except KeyboardInterrupt:
        tracker.finalize(
            status="paused",
            step=done,
            total_steps=len(tasks),
            message="Discovery suite interrupted by user.",
        )
        raise
    except Exception as exc:
        tracker.finalize(
            status="failed",
            step=done,
            total_steps=len(tasks),
            message=f"{type(exc).__name__}: {exc}",
        )
        raise

    tracker.finalize(
        status="completed",
        step=done,
        total_steps=len(tasks),
        message="Discovery suite completed.",
    )


if __name__ == "__main__":
    main()
