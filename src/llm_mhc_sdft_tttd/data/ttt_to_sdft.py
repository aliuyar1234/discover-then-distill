from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _first_present(rec: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    with open(path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _normalize_record(
    rec: Dict[str, Any],
    default_prompt: Optional[str],
) -> Optional[Dict[str, Any]]:
    prompt = _first_present(rec, ("prompt", "problem_description", "problem"), default_prompt)
    demonstration = _first_present(rec, ("demonstration", "best_solution", "best_state", "solution"), None)
    reward = rec["reward"] if "reward" in rec else rec.get("best_reward")

    if prompt is None or demonstration is None:
        return None
    return {
        "prompt": str(prompt),
        "demonstration": str(demonstration),
        "reward": None if reward is None else float(reward),
    }


def convert_discovery_logs_to_sdft_rows(
    input_paths: Iterable[str],
    min_reward: Optional[float] = None,
    default_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for p in input_paths:
        path = Path(p)
        rows = _load_json_or_jsonl(path)
        for rec in rows:
            norm = _normalize_record(rec, default_prompt=default_prompt)
            if norm is None:
                continue
            r = norm["reward"]
            if min_reward is not None and r is not None and r < min_reward:
                continue
            out.append(
                {
                    "prompt": norm["prompt"],
                    "demonstration": norm["demonstration"],
                }
            )
    return out


def write_sdft_jsonl(path: str, rows: List[Dict[str, str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
