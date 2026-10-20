"""Per-run, per-step logs for the daily pipeline.

While `run_nightly` runs, every log record in the process is also written to
`data/runs/logs/<run_id>.jsonl`, tagged with the pipeline step that was running. The
Schedule tab reads these back so clicking a step shows that step's log.

Runs from before this existed have no .jsonl, so `read_step_logs` falls back to slicing
`logs/jobs.log` by the step's time window.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from settings import PROJECT_ROOT

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def log_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "runs" / "logs"


def log_path(data_dir: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[\w.-]+", run_id or ""):
        raise ValueError(f"bad run id {run_id!r}")
    return log_dir(data_dir) / f"{run_id}.jsonl"


class StepLogHandler(logging.Handler):
    """Appends each record as one JSON line, tagged with the current `step`."""

    def __init__(self, path: Path, level: int = logging.INFO):
        super().__init__(level)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.step: str | None = None
        self._fh = path.open("a", encoding="utf-8", buffering=1)  # line-buffered: readable while running

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg += "\n" + logging.Formatter().formatException(record.exc_info)
            line = {"t": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                    "level": record.levelname, "logger": record.name, "step": self.step, "msg": msg}
            with self.lock:
                self._fh.write(json.dumps(line, default=str) + "\n")
        except Exception:  # never let logging break the pipeline
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


class capture:
    """Context manager: attach a StepLogHandler to the root logger for the duration of a run."""

    def __init__(self, data_dir: Path, run_id: str):
        self.handler = StepLogHandler(log_path(data_dir, run_id))
        self._old_level: int | None = None

    def __enter__(self) -> StepLogHandler:
        root = logging.getLogger()
        self._raised: list[logging.Handler] = []
        if root.level > logging.INFO:
            # let INFO records reach our handler without making existing handlers chattier
            self._old_level = root.level
            for h in root.handlers:
                if h.level == logging.NOTSET:
                    h.setLevel(root.level)
                    self._raised.append(h)
            root.setLevel(logging.INFO)
        root.addHandler(self.handler)
        return self.handler

    def __exit__(self, *exc) -> None:
        root = logging.getLogger()
        root.removeHandler(self.handler)
        self.handler.close()
        if self._old_level is not None:
            root.setLevel(self._old_level)
            for h in self._raised:
                h.setLevel(logging.NOTSET)


# ── reading ──────────────────────────────────────────────────────────────

# Used only for old runs without a .jsonl and with no recorded step times.
_STEP_LOGGERS = {
    "ingest": ("sources", "source."),
    "tune": ("training.trainer", "optuna"),
    "backtest": ("backtest",),
    "train": ("prediction", "training.trainer"),
    "predict": ("prediction",),
}
_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) (\w+)\s+(\S+) \| (.*)$")


def read_step_logs(data_dir: Path, run: dict, step: str | None = None, after: int = 0,
                   min_level: str = "INFO", limit: int = 2000, logs_root: Path | None = None) -> dict:
    """Log lines for one run (optionally one step). `after` is the number of matching lines the
    caller already has, so a running step can be followed by polling with the returned `next`."""
    floor = LEVELS.get(min_level.upper(), 20)
    path = log_path(data_dir, run["run_id"])
    if path.exists():
        lines, source = _read_jsonl(path), "run"
        if step:  # "recommend" is part of predict
            lines = [x for x in lines if x.get("step") == step or (step == "predict" and x.get("step") == "recommend")]
    else:
        lines, source = _from_jobs_log(run, step, logs_root or PROJECT_ROOT / "logs"), "jobs.log"
    lines = [x for x in lines if LEVELS.get(x.get("level", "INFO"), 20) >= floor]
    total = len(lines)
    out = lines[after:after + limit] if after < total else []
    return {"run_id": run["run_id"], "step": step, "source": source, "lines": out,
            "next": after + len(out), "total": total}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:  # a line being written right now
                continue
    return out


def _utc(iso: str | None) -> datetime | None:
    if not iso:
        return None
    t = datetime.fromisoformat(iso)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _step_window(run: dict, step: str | None) -> tuple[datetime | None, datetime | None]:
    start, end = _utc(run.get("started")), _utc(run.get("finished"))
    if not step:
        return start, end
    steps = [s for s in run.get("steps", []) if s.get("started")]
    for i, s in enumerate(steps):
        if s["step"] == step:
            nxt = next((x for x in steps[i + 1:] if x["step"] != "recommend"), None)
            return _utc(s["started"]), _utc(nxt["started"]) if nxt else end
    return None, None  # step times unknown


def _from_jobs_log(run: dict, step: str | None, logs_root: Path) -> list[dict]:
    lo, hi = _step_window(run, step)
    by_logger = False
    if lo is None:  # no step times recorded: whole run, filtered by which modules log for that step
        lo, hi = _utc(run.get("started")), _utc(run.get("finished"))
        by_logger = bool(step)
    if lo is None:
        return []
    files = [p for p in (logs_root / "jobs.log.1", logs_root / "jobs.log") if p.exists()]
    out: list[dict] = []
    for p in files:
        with p.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.rstrip("\n")
                m = _LINE.match(raw)
                if not m:
                    if out and out[-1].get("_in"):  # traceback continuation
                        out[-1]["msg"] += "\n" + raw
                    continue
                t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").astimezone(timezone.utc)  # log times are local
                t = t.replace(microsecond=int(m[2]) * 1000)
                inside = t >= lo and (hi is None or t <= hi)
                if inside and by_logger:
                    inside = m[4].startswith(_STEP_LOGGERS.get(step, ("",)))
                if out:
                    out[-1].pop("_in", None)
                if inside:
                    out.append({"t": t.isoformat(timespec="milliseconds"), "level": m[3], "logger": m[4],
                                "step": step, "msg": m[5], "_in": True})
    if out:
        out[-1].pop("_in", None)
    return out
