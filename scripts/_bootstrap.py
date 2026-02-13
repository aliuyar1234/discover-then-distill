"""Bootstrap helper for running repo scripts without installing the package.

This repository is meant to be installed via:

    pip install -e .[dev]

However, editable installs can fail or be skipped in some environments (CI
sandboxes, locked-down containers, etc.). To keep the CLI scripts runnable
without relying on editable install state, this module can be imported from
any script under `scripts/`:

    from _bootstrap import bootstrap
    bootstrap()

It will add `<repo_root>/src` to `sys.path` if needed.

This is intentionally small and dependency-free.
"""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap() -> None:
    """Add `<repo_root>/src` to `sys.path` if not already present."""

    # scripts/_bootstrap.py -> scripts/ -> repo root
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if not src.exists():
        raise RuntimeError(
            f"Expected src directory at {src} (repo root: {repo_root}). "
            "Run scripts from the repo root, e.g. `python scripts/pretrain.py ...`."
        )
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
