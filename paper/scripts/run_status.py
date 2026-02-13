from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def fmt_secs(seconds: Any) -> str:
    if seconds is None:
        return "-"
    try:
        s = int(float(seconds))
    except Exception:
        return "-"
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_iso_ts(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        # Python's fromisoformat expects "+00:00" rather than trailing "Z".
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


def is_pid_running(pid: Any) -> bool | None:
    if not isinstance(pid, (int, float)):
        return None
    pid_int = int(pid)
    try:
        os.kill(pid_int, 0)
        return True
    except PermissionError:
        # Process exists but current user cannot signal it.
        return True
    except ProcessLookupError:
        return False
    except OSError:
        # Fall through to platform-specific fallback checks.
        pass

    # Windows fallback: tasklist is reliable even when os.kill(..., 0) is not.
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid_int}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            out = (proc.stdout or "").strip()
            if not out:
                return False
            if "No tasks are running" in out:
                return False
            return str(pid_int) in out
    except Exception:
        pass

    return None


def collect(root: Path) -> List[Dict[str, Any]]:
    now = time.time()
    rows: List[Dict[str, Any]] = []
    for hb_path in root.rglob("_run_tracker/heartbeat.json"):
        hb = safe_read_json(hb_path)
        if not hb:
            continue
        art = safe_read_json(hb_path.parent / "artifacts.json")
        meta = safe_read_json(hb_path.parent / "meta.json")

        status = str(hb.get("status") or "")
        hb_ts = parse_iso_ts(hb.get("time"))
        elapsed = hb.get("elapsed_seconds")
        if status == "running":
            if isinstance(elapsed, (int, float)) and hb_ts is not None:
                elapsed_live: Any = float(elapsed) + max(0.0, now - hb_ts)
            elif hb_ts is not None:
                elapsed_live = max(0.0, now - hb_ts)
            else:
                elapsed_live = elapsed
        else:
            elapsed_live = elapsed

        pid_alive = is_pid_running(meta.get("pid"))
        if status == "running" and pid_alive is False:
            status = "stale"

        rows.append(
            {
                "run_id": hb.get("run_id"),
                "run_type": hb.get("run_type"),
                "status": status,
                "step": hb.get("step"),
                "total_steps": hb.get("total_steps"),
                "progress_pct": hb.get("progress_pct"),
                "eta_seconds": hb.get("eta_seconds"),
                "elapsed_seconds": hb.get("elapsed_seconds"),
                "elapsed_seconds_live": elapsed_live,
                "heartbeat_age_seconds": (max(0.0, now - hb_ts) if hb_ts is not None else None),
                "time": hb.get("time"),
                "out_dir": hb.get("out_dir"),
                "pid": meta.get("pid"),
                "last_message": hb.get("last_message"),
                "required_present": art.get("required_present"),
                "required_total": art.get("required_total"),
            }
        )
    status_rank = {
        "running": 5,
        "stale": 4,
        "paused": 3,
        "failed": 2,
        "completed": 1,
    }
    rows.sort(
        key=lambda r: (status_rank.get(str(r.get("status")), 0), str(r.get("time"))),
        reverse=True,
    )
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No heartbeat files found.")
        return
    print("status      progress        eta      elapsed  hb_age   artifacts   run_type         out_dir")
    print("-" * 124)
    for r in rows:
        step = r.get("step")
        total = r.get("total_steps")
        if isinstance(step, (int, float)) and isinstance(total, (int, float)) and int(total) > 0:
            prog = f"{int(step):>5}/{int(total):<5} {float(r.get('progress_pct', 0.0)):6.2f}%"
        else:
            prog = "   -/ -      -   "
        req_have = r.get("required_present")
        req_total = r.get("required_total")
        if isinstance(req_have, int) and isinstance(req_total, int):
            art = f"{req_have}/{req_total}"
        else:
            art = "-/-"
        print(
            f"{str(r.get('status', '-')):<10}"
            f"{prog:<16}  "
            f"{fmt_secs(r.get('eta_seconds')):<8}  "
            f"{fmt_secs(r.get('elapsed_seconds_live', r.get('elapsed_seconds'))):<8}  "
            f"{fmt_secs(r.get('heartbeat_age_seconds')):<6}  "
            f"{art:<9}  "
            f"{str(r.get('run_type', '-')):<15}  "
            f"{str(r.get('out_dir', '-'))}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Live status view for tracked runs.")
    ap.add_argument("--root", default="runs", help="Root directory to scan for _run_tracker/heartbeat.json files.")
    ap.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds. 0 prints once.")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of table.")
    args = ap.parse_args()

    root = Path(args.root)
    if args.watch <= 0:
        rows = collect(root)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print_table(rows)
        return

    while True:
        rows = collect(root)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print("\x1bc", end="")
            print_table(rows)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
