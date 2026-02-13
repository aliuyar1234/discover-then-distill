from __future__ import annotations

import json
import os
import signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now_ts() -> float:
    return time.time()


def _iso_utc(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = _now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


class GracefulStopper:
    """Trap SIGINT/SIGTERM and request a graceful checkpoint-safe stop."""

    def __init__(self) -> None:
        self.stop_requested = False
        self._prev_sigint = None
        self._prev_sigterm = None

    def _handle(self, signum, frame) -> None:  # type: ignore[no-untyped-def]
        self.stop_requested = True
        print(f"[signal] received {signum}; will stop at the next safe checkpoint boundary.")

    def install(self) -> None:
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def uninstall(self) -> None:
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)


class RunTracker:
    """Structured run metadata, heartbeat, events, and artifact tracking."""

    def __init__(
        self,
        out_dir: str,
        run_type: str,
        total_steps: Optional[int] = None,
        command: Optional[str] = None,
        resume: bool = False,
        heartbeat_interval_sec: float = 15.0,
    ) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.run_type = run_type
        self.command = command
        self.heartbeat_interval_sec = float(max(1.0, heartbeat_interval_sec))

        self.tracker_dir = self.out_dir / "_run_tracker"
        self.tracker_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.tracker_dir / "meta.json"
        self.events_path = self.tracker_dir / "events.jsonl"
        self.heartbeat_path = self.tracker_dir / "heartbeat.json"
        self.artifacts_path = self.tracker_dir / "artifacts.json"

        # Fresh run: start with clean tracker files in this run directory.
        if not resume:
            for p in (self.events_path, self.heartbeat_path, self.artifacts_path, self.meta_path):
                if p.exists():
                    p.unlink()

        existing_meta = _read_json(self.meta_path) if resume else {}
        existing_heartbeat = _read_json(self.heartbeat_path) if resume else {}
        existing_artifacts = _read_json(self.artifacts_path) if resume else {}

        now = _now_ts()
        if resume and existing_meta:
            self.run_id = str(existing_meta.get("run_id") or f"{run_type}-{int(now)}-{os.getpid()}")
            self.started_ts = float(existing_meta.get("started_ts", now))
            self.resume_count = int(existing_meta.get("resume_count", 0)) + 1
        else:
            self.run_id = f"{run_type}-{int(now)}-{os.getpid()}"
            self.started_ts = now
            self.resume_count = 0

        hb_step = existing_heartbeat.get("step")
        self.current_step: Optional[int] = int(hb_step) if isinstance(hb_step, (int, float)) else None
        self.total_steps: Optional[int] = (
            int(total_steps)
            if total_steps is not None
            else (int(existing_heartbeat["total_steps"]) if isinstance(existing_heartbeat.get("total_steps"), (int, float)) else None)
        )

        self._last_write_ts = 0.0
        self._last_step_ts = now
        self._last_step_val = float(self.current_step) if self.current_step is not None else 0.0
        self._step_rate: Optional[float] = None

        self._artifacts: Dict[str, Dict[str, Any]] = {}
        for row in existing_artifacts.get("artifacts", []):
            if isinstance(row, dict) and "name" in row:
                self._artifacts[str(row["name"])] = row

        self._write_meta(status="running", now=now)
        self._append_global_index(status="running")
        self.event("resume" if resume else "start", total_steps=self.total_steps, command=self.command)
        self.heartbeat(step=self.current_step, total_steps=self.total_steps, status="running", force=True)

    def _write_meta(self, status: str, now: Optional[float] = None) -> None:
        if now is None:
            now = _now_ts()
        payload = {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "out_dir": str(self.out_dir),
            "status": status,
            "started_at": _iso_utc(self.started_ts),
            "started_ts": self.started_ts,
            "updated_at": _iso_utc(now),
            "updated_ts": now,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "resume_count": self.resume_count,
            "command": self.command,
            "total_steps": self.total_steps,
        }
        _write_json(self.meta_path, payload)

    def _append_global_index(self, status: str) -> None:
        try:
            root = Path("runs") / "_tracker"
            root.mkdir(parents=True, exist_ok=True)
            row = {
                "time": _iso_utc(),
                "run_id": self.run_id,
                "run_type": self.run_type,
                "out_dir": str(self.out_dir),
                "status": status,
            }
            with open(root / "runs_index.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass

    def event(self, name: str, **fields: Any) -> None:
        row = {
            "time": _iso_utc(),
            "run_id": self.run_id,
            "event": name,
            **fields,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def register_artifact(self, name: str, path: str, required: bool = True, description: str = "") -> None:
        p = Path(path)
        if not p.is_absolute():
            p = (self.out_dir / p).resolve()
        self._artifacts[name] = {
            "name": name,
            "path": str(p),
            "required": bool(required),
            "description": description,
        }
        self.refresh_artifacts()

    def refresh_artifacts(self) -> None:
        rows = []
        for name in sorted(self._artifacts.keys()):
            row = dict(self._artifacts[name])
            p = Path(row["path"])
            exists = p.exists()
            row["exists"] = exists
            row["size_bytes"] = int(p.stat().st_size) if exists else None
            row["modified_at"] = _iso_utc(p.stat().st_mtime) if exists else None
            rows.append(row)
            self._artifacts[name] = row
        payload = {
            "run_id": self.run_id,
            "updated_at": _iso_utc(),
            "artifacts": rows,
            "required_total": sum(1 for r in rows if r.get("required")),
            "required_present": sum(1 for r in rows if r.get("required") and r.get("exists")),
        }
        _write_json(self.artifacts_path, payload)

    def heartbeat(
        self,
        *,
        step: Optional[int],
        total_steps: Optional[int] = None,
        status: str = "running",
        message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        now = _now_ts()
        if total_steps is not None:
            self.total_steps = int(total_steps)
        if step is not None:
            self.current_step = int(step)
            step_val = float(step)
            d_step = step_val - self._last_step_val
            d_t = now - self._last_step_ts
            if d_step > 0 and d_t > 1e-6:
                self._step_rate = d_step / d_t
                self._last_step_val = step_val
                self._last_step_ts = now

        if not force and (now - self._last_write_ts) < self.heartbeat_interval_sec and status == "running":
            return _read_json(self.heartbeat_path)

        elapsed = max(0.0, now - self.started_ts)
        progress_pct = None
        eta_seconds = None
        if self.current_step is not None and self.total_steps and self.total_steps > 0:
            progress_pct = 100.0 * float(self.current_step) / float(self.total_steps)
            remaining = max(0, int(self.total_steps) - int(self.current_step))
            if self._step_rate is not None and self._step_rate > 1e-6:
                eta_seconds = float(remaining) / self._step_rate
            elif self.current_step > 0:
                eta_seconds = elapsed * (float(remaining) / float(self.current_step))

        payload: Dict[str, Any] = {
            "time": _iso_utc(now),
            "run_id": self.run_id,
            "run_type": self.run_type,
            "status": status,
            "out_dir": str(self.out_dir),
            "step": self.current_step,
            "total_steps": self.total_steps,
            "progress_pct": progress_pct,
            "eta_seconds": eta_seconds,
            "elapsed_seconds": elapsed,
            "step_rate": self._step_rate,
            "last_message": message,
            "metrics": metrics or {},
            "resume_count": self.resume_count,
        }
        _write_json(self.heartbeat_path, payload)
        self._last_write_ts = now
        self._write_meta(status=status, now=now)
        return payload

    def finalize(
        self,
        *,
        status: str,
        step: Optional[int],
        total_steps: Optional[int] = None,
        message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.refresh_artifacts()
        self.heartbeat(
            step=step,
            total_steps=total_steps,
            status=status,
            message=message,
            metrics=metrics,
            force=True,
        )
        self.event("finalize", status=status, message=message, step=step, total_steps=total_steps)
        self._append_global_index(status=status)
