"""Test bootstrap.

The repo is intended to be installed via `pip install -e .[dev]`.
However, to keep tests runnable in minimal environments (or when editable
installs are unavailable), we also add `<repo_root>/src` to PYTHONPATH.

This makes `pytest -q` work even if the package is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
