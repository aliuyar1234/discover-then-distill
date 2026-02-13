from __future__ import annotations

import argparse
import random
import string
from pathlib import Path


def rand_words(n: int) -> str:
    alphabet = string.ascii_lowercase
    return " ".join(
        "".join(random.choice(alphabet) for _ in range(random.randint(3, 8)))
        for _ in range(n)
    )


def gen_line() -> str:
    r = random.random()
    if r < 0.35:
        a, b = random.randint(0, 999), random.randint(0, 999)
        return f"Q: What is {a}+{b}? A: {a+b}"
    if r < 0.55:
        xs = [random.randint(0, 99) for _ in range(6)]
        ys = " ".join(map(str, sorted(xs)))
        return f"Sort ascending: {' '.join(map(str, xs))} -> {ys}"
    if r < 0.70:
        return f"Text: {rand_words(8)}"
    if r < 0.85:
        s = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
        return f"Reverse '{s}' -> '{s[::-1]}'"
    return f"Note: {rand_words(12)}."


def write_lines(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(gen_line() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_lines", type=int, default=200_000)
    ap.add_argument("--val_lines", type=int, default=10_000)
    ap.add_argument("--train_out", default="data/raw/train.txt")
    ap.add_argument("--val_out", default="data/raw/val.txt")
    args = ap.parse_args()

    random.seed(args.seed)
    write_lines(Path(args.train_out), args.train_lines)
    write_lines(Path(args.val_out), args.val_lines)
    print(f"wrote {args.train_out}, {args.val_out}")


if __name__ == "__main__":
    main()
