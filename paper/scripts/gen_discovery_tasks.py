from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path
from typing import Dict, List


def make_add(i: int) -> Dict[str, str]:
    a, b = random.randint(0, 9999), random.randint(0, 9999)
    return {
        "task_id": f"add_{i:04d}",
        "family": "add",
        "problem": f"Compute {a} + {b}. Output ONLY the integer.",
        "target": str(a + b),
    }


def make_sort(i: int) -> Dict[str, str]:
    xs = [random.randint(0, 99) for _ in range(8)]
    return {
        "task_id": f"sort_{i:04d}",
        "family": "sort",
        "problem": "Sort these integers in ascending order. Output ONLY numbers separated by spaces: " + " ".join(map(str, xs)),
        "target": " ".join(map(str, sorted(xs))),
    }


def make_reverse(i: int) -> Dict[str, str]:
    s = "".join(random.choice(string.ascii_lowercase) for _ in range(12))
    return {
        "task_id": f"rev_{i:04d}",
        "family": "rev",
        "problem": f"Reverse this string. Output ONLY the reversed string: {s}",
        "target": s[::-1],
    }


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_add", type=int, default=80)
    ap.add_argument("--train_sort", type=int, default=80)
    ap.add_argument("--held_add", type=int, default=40)
    ap.add_argument("--held_sort", type=int, default=40)
    ap.add_argument("--held_rev", type=int, default=40)
    ap.add_argument("--out_train", default="data/discovery/train.jsonl")
    ap.add_argument("--out_heldout", default="data/discovery/heldout.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)

    train = [make_add(i) for i in range(args.train_add)] + [
        make_sort(i) for i in range(args.train_sort)
    ]
    held = [make_add(i + args.train_add) for i in range(args.held_add)] + [
        make_sort(i + args.train_sort) for i in range(args.held_sort)
    ] + [make_reverse(i) for i in range(args.held_rev)]

    random.shuffle(train)
    random.shuffle(held)

    write_jsonl(Path(args.out_train), train)
    write_jsonl(Path(args.out_heldout), held)
    print(f"wrote {args.out_train} and {args.out_heldout}")


if __name__ == "__main__":
    main()
