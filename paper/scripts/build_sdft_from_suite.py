from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def read_tasks(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["task_id"])] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_root", required=True, help="e.g. runs/suite/ttt_train_seed0")
    ap.add_argument("--tasks_jsonl", required=True, help="e.g. data/discovery/train.jsonl")
    ap.add_argument("--out_jsonl", required=True, help="e.g. data/sdft/from_ttt_train.jsonl")
    ap.add_argument("--min_reward", type=float, default=0.99)
    args = ap.parse_args()

    suite_root = Path(args.suite_root)
    tasks = read_tasks(Path(args.tasks_jsonl))
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0
    with open(out_path, "w", encoding="utf-8") as wf:
        for task_id, task in tasks.items():
            seen += 1
            result_path = suite_root / task_id / "result.json"
            if not result_path.exists():
                continue
            r = json.loads(result_path.read_text(encoding="utf-8"))
            if float(r.get("best_reward", 0.0)) < args.min_reward:
                continue
            prompt = str(task["problem"])
            demo = str(r.get("best_state", "")).strip()
            wf.write(json.dumps({"prompt": prompt, "demonstration": demo}) + "\n")
            kept += 1

    print(
        json.dumps(
            {
                "suite_root": str(suite_root),
                "tasks_seen": seen,
                "kept": kept,
                "min_reward": args.min_reward,
                "out_jsonl": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
