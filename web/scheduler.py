"""Daily pipeline schedule, run by the dashboard server.

The schedule (time of day, which steps, on/off) is kept in <data_dir>/schedule.json so it
survives restarts. A background thread checks every 20 seconds; when the time arrives it
starts the "nightly" job through the same JobRunner the buttons use. If another job is
running at that moment it waits for it to finish (up to MAX_WAIT).
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

STEPS = ("ingest", "tune", "backtest", "train", "predict", "strategy")
DEFAULT = {"enabled": False, "time": "02:30", "steps": list(STEPS), "backtest_days": None, "tune_days": None,
           "last_slot": None, "last_started": None, "last_job_id": None,
           # Separate from the daily pipeline: top up the data (actuals, weather forecast) through
           # the day so charts and predictions use the latest prices and demand.
           "refresh_enabled": True, "refresh_minutes": 30, "last_refresh": None}
TICK_SECONDS = 20
MAX_WAIT = timedelta(hours=3)


class Scheduler:
    def __init__(self, runner, get_settings, trading=None):
        self.runner = runner
        self.trading = trading  # TradingRunner: paper-trading cycles on their own timer
        self.get_settings = get_settings
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── persistence ──────────────────────────────────────────────────────
    @property
    def path(self) -> Path:
        return Path(self.get_settings().data_dir) / "schedule.json"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.get_settings().timezone)

    def load(self) -> dict:
        try:
            return {**DEFAULT, **json.loads(self.path.read_text())}
        except (FileNotFoundError, json.JSONDecodeError):
            return dict(DEFAULT)

    def _save(self, cfg: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2))
        tmp.replace(self.path)

    # ── time maths ───────────────────────────────────────────────────────
    def _slot(self, cfg: dict, day) -> datetime:
        hh, mm = (int(x) for x in cfg["time"].split(":"))
        return datetime(day.year, day.month, day.day, hh, mm, tzinfo=self.tz)

    def next_run(self, cfg: dict | None = None, now: datetime | None = None) -> datetime | None:
        cfg = cfg or self.load()
        if not cfg["enabled"]:
            return None
        now = now or datetime.now(self.tz)
        today = self._slot(cfg, now.date())
        if cfg.get("last_slot") != today.isoformat() and now < today + MAX_WAIT:
            return max(today, now) if now >= today else today
        return self._slot(cfg, (now + timedelta(days=1)).date())

    # ── public API ───────────────────────────────────────────────────────
    def update(self, changes: dict) -> dict:
        with self._lock:
            cfg = self.load()
            if "time" in changes:
                hh, mm = (int(x) for x in str(changes["time"]).split(":")[:2])
                if not (0 <= hh < 24 and 0 <= mm < 60):
                    raise ValueError("time must be HH:MM")
                cfg["time"] = f"{hh:02d}:{mm:02d}"
            if "steps" in changes:
                steps = [s for s in STEPS if s in (changes["steps"] or [])]
                if not steps:
                    raise ValueError("pick at least one step")
                cfg["steps"] = steps
            if "backtest_days" in changes:
                v = changes["backtest_days"]
                cfg["backtest_days"] = int(v) if v not in (None, "") else None
            if "tune_days" in changes:
                v = changes["tune_days"]
                if v not in (None, "") and not 7 <= int(v) <= 2000:
                    raise ValueError("tune days must be between 7 and 2000")
                cfg["tune_days"] = int(v) if v not in (None, "") else None
            if "enabled" in changes:
                cfg["enabled"] = bool(changes["enabled"])
            if "refresh_enabled" in changes:
                cfg["refresh_enabled"] = bool(changes["refresh_enabled"])
            if "refresh_minutes" in changes:
                minutes = int(changes["refresh_minutes"])
                if not 5 <= minutes <= 24 * 60:
                    raise ValueError("refresh interval must be 5-1440 minutes")
                cfg["refresh_minutes"] = minutes
            if changes.get("enabled") or "time" in changes:
                # Starting (or moving the time) after today's slot has passed shouldn't fire at once:
                # the first run is the next occurrence of the time.
                now = datetime.now(self.tz)
                today = self._slot(cfg, now.date())
                if now >= today:
                    cfg["last_slot"] = today.isoformat()
                elif cfg.get("last_slot") == today.isoformat():
                    cfg["last_slot"] = None
            self._save(cfg)
            return cfg

    def job_args(self, cfg: dict, trigger: str) -> dict:
        return {"steps": cfg["steps"], "backtest_days": cfg.get("backtest_days"),
                "tune_days": cfg.get("tune_days"), "trigger": trigger}

    def fetch_now(self) -> dict:
        """Download the latest data right away; the background refresh timer restarts from now."""
        with self._lock:
            job = self.runner.start("ingest", {"trigger": "fetch"}, self.get_settings())
            cfg = self.load()
            cfg["last_refresh"] = datetime.now(self.tz).isoformat(timespec="seconds")
            self._save(cfg)
            return job

    def run_now(self) -> dict:
        cfg = self.load()
        return self.runner.start("nightly", self.job_args(cfg, "manual"), self.get_settings())

    # ── background loop ──────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(TICK_SECONDS):
            try:
                self.tick()
            except Exception:
                log.exception("scheduler tick failed")
            if self.trading is not None:
                try:
                    self.trading.tick()
                except Exception:
                    log.exception("trading tick failed")

    def next_refresh(self, cfg: dict | None = None, now: datetime | None = None) -> datetime | None:
        cfg = cfg or self.load()
        if not cfg.get("refresh_enabled"):
            return None
        last = cfg.get("last_refresh")
        if not last:
            return now or datetime.now(self.tz)
        return datetime.fromisoformat(last) + timedelta(minutes=int(cfg.get("refresh_minutes") or 30))

    def tick(self, now: datetime | None = None) -> bool:
        """Start the daily pipeline if it's due, else a data refresh if that's due.
        Returns True if it started a job."""
        with self._lock:
            cfg = self.load()
            now = now or datetime.now(self.tz)
            if self._tick_daily(cfg, now):
                return True
            return self._tick_refresh(cfg, now)

    def _tick_daily(self, cfg: dict, now: datetime) -> bool:
        if not cfg["enabled"]:
            return False
        slot = self._slot(cfg, now.date())
        if now < slot or cfg.get("last_slot") == slot.isoformat():
            return False
        if now - slot > MAX_WAIT:  # server was off / busy for too long: skip today's run
            log.warning("scheduled run for %s missed; skipping to tomorrow", slot)
            cfg["last_slot"] = slot.isoformat()
            self._save(cfg)
            return False
        if self.runner.busy():
            return False  # wait for the current job; try again next tick
        job = self.runner.start("nightly", self.job_args(cfg, "schedule"), self.get_settings())
        cfg["last_slot"] = slot.isoformat()
        cfg["last_started"] = now.isoformat(timespec="seconds")
        cfg["last_job_id"] = job.get("id")
        # the pipeline's own ingest counts as a refresh
        if "ingest" in cfg["steps"]:
            cfg["last_refresh"] = now.isoformat(timespec="seconds")
        self._save(cfg)
        log.info("scheduled pipeline started (%s)", ", ".join(cfg["steps"]))
        return True

    def _tick_refresh(self, cfg: dict, now: datetime) -> bool:
        if not cfg.get("refresh_enabled"):
            return False
        due = self.next_refresh(cfg, now)
        if due is not None and now < due:
            return False
        if self.runner.busy():
            return False
        self.runner.start("ingest", {"trigger": "refresh"}, self.get_settings())
        cfg["last_refresh"] = now.isoformat(timespec="seconds")
        self._save(cfg)
        return True
