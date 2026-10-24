"""Runs paper-trading cycles for the dashboard: on a timer (when switched on in the Trading tab)
and on demand. Separate from the JobRunner so a long nightly pipeline doesn't hold up trading;
each cycle runs in its own short-lived process (model code stays out of the web server)."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

CYCLE_TIMEOUT = 15 * 60  # seconds; a cycle is normally a few seconds
FAILURE_BACKOFF = timedelta(minutes=5)  # wait this long after a failed cycle before retrying


def _cycle_entry(raw_settings: dict, settings_path: str, args: dict) -> dict:
    from pathlib import Path

    from settings import Settings
    from trading.paper import run_cycle

    settings = Settings(raw_settings, Path(settings_path) if settings_path else None)
    return run_cycle(settings, force_target=args.get("force_target"), reason=args.get("reason", ""))


class TradingRunner:
    def __init__(self, get_settings):
        self.get_settings = get_settings
        self._lock = threading.Lock()
        self.current: dict | None = None
        self.last: dict | None = None
        self._last_attempt: datetime | None = None
        self._last_failed = False

    def busy(self) -> bool:
        return self.current is not None

    def start(self, trigger: str = "manual", force_target: float | None = None, reason: str = "") -> dict:
        with self._lock:
            if self.current is not None:
                raise RuntimeError("a trading cycle is already running")
            self.current = {"trigger": trigger, "started": _now(), "status": "running"}
            self._last_attempt = datetime.now(timezone.utc)
            args = {"force_target": force_target, "reason": reason}
            threading.Thread(target=self._run, args=(dict(self.current), args), daemon=True,
                             name="trading-cycle").start()
            return dict(self.current)

    def _run(self, info: dict, args: dict) -> None:
        from compute import run_isolated

        s = self.get_settings()
        try:
            res = run_isolated(_cycle_entry, s.raw, str(s.path or ""), args, label="paper trade cycle",
                               timeout=CYCLE_TIMEOUT)
            info.update(status="ok", result=res)
            self._last_failed = False
        except Exception as exc:
            log.warning("paper trade cycle failed: %s", str(exc).split("\n")[0])
            info.update(status="error", error=str(exc).split("\n")[0])
            self._last_failed = True
        info["finished"] = _now()
        with self._lock:
            self.last, self.current = info, None

    # called by the scheduler loop
    def due(self, state: dict, now: datetime | None = None) -> bool:
        if not state.get("enabled") or self.busy():
            return False
        now = now or datetime.now(timezone.utc)
        if now.weekday() >= 5:  # gas CFDs don't trade at weekends
            return False
        if self._last_failed and self._last_attempt and now - self._last_attempt < FAILURE_BACKOFF:
            return False  # don't hammer IG after a failure
        last = state.get("last_cycle")
        if not last:
            return True
        return now >= datetime.fromisoformat(last) + timedelta(minutes=int(state.get("interval_minutes") or 60))

    def next_cycle(self, state: dict) -> str | None:
        if not state.get("enabled"):
            return None
        last = state.get("last_cycle")
        if not last:
            return _now()
        nxt = datetime.fromisoformat(last) + timedelta(minutes=int(state.get("interval_minutes") or 60))
        while nxt.weekday() >= 5:
            nxt = (nxt + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        return nxt.isoformat(timespec="seconds")

    def tick(self, now: datetime | None = None) -> bool:
        from trading.paper import load_state

        try:
            state = load_state(self.get_settings())
        except Exception:
            return False
        if self.due(state, now):
            self.start("schedule")
            return True
        return False

    def snapshot(self) -> dict:
        return {"current": self.current, "last": self.last}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
