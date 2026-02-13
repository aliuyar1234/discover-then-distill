from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for art_path in root.rglob("_run_tracker/artifacts.json"):
        doc = safe_read_json(art_path)
        if not doc:
            continue
        artifacts = doc.get("artifacts", [])
        missing_required = []
        for a in artifacts:
            if not isinstance(a, dict):
                continue
            if bool(a.get("required")) and not bool(a.get("exists")):
                missing_required.append(a)
        rows.append(
            {
                "run_id": doc.get("run_id"),
                "path": str(art_path),
                "required_present": doc.get("required_present"),
                "required_total": doc.get("required_total"),
                "missing_required": missing_required,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Artifact completeness report for tracked runs.")
    ap.add_argument("--root", default="runs", help="Root directory to scan")
    ap.add_argument("--json", action="store_true", help="Print full JSON output")
    args = ap.parse_args()

    rows = collect(Path(args.root))
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print("No artifact manifests found.")
        return

    for row in rows:
        present = row.get("required_present")
        total = row.get("required_total")
        run_id = row.get("run_id")
        print(f"{run_id}: required {present}/{total}")
        missing = row.get("missing_required", [])
        if missing:
            for m in missing:
                print(f"  - MISSING: {m.get('name')} -> {m.get('path')}")


if __name__ == "__main__":
    main()

