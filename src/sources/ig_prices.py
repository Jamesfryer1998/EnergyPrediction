"""Hourly bid/ask candles for the traded market (UK natural gas by default) from IG's demo API."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from storage import TS
from .base import DataSource, SourceFatal, SourceNotImplemented, UpdateResult

log = logging.getLogger(__name__)


REFRESH_AFTER_HOUR = timedelta(minutes=2)  # wait this long after the hour for the candle to close
ALLOWANCE_BACKOFF = timedelta(hours=6)  # don't ask again this soon after "allowance used up"


def _utcnow() -> datetime:  # naive UTC; a function so tests can move the clock
    return datetime.now(timezone.utc).replace(tzinfo=None)


class IGGasPrices(DataSource):
    """IG CFD quotes for the paper-trading market: hourly bid/ask OHLC, mid and spread.

    IG limits requests per minute and candles per week, so this source makes ONE request per
    run instead of walking week-by-week windows:
      * first run: the latest `trading.history_points` candles (whatever history IG has);
      * afterwards: everything since the last stored candle (minus a few hours for revisions),
        at most once per clock hour (a couple of minutes after the hour, once the candle has
        closed). Without that limit a 1-minute trading interval re-downloaded the last few hours
        every cycle while the market was shut, and used up the weekly allowance overnight.
      * if IG says the weekly allowance is used up, it stops asking for ALLOWANCE_BACKOFF.
    Only one download runs at a time on this machine (a data refresh, a backtest and a trading
    cycle can all ask at once); the others skip and use what's stored."""

    name = "ig_gas"
    description = "IG demo: hourly bid/ask candles for the paper-trading market (UK natural gas)"
    category = "trading"
    frequency = "1h"
    revision_lookback = timedelta(hours=3)

    def __init__(self, *args, client=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from trading.ig import IGClient, IGNotConfigured

            try:
                self._client = IGClient()
            except IGNotConfigured as exc:
                raise SourceNotImplemented(str(exc)) from None
        return self._client

    @property
    def cfg(self) -> dict:
        return self.settings.get("trading", {}) or {}

    def update(self, start=None, end=None, full: bool = False) -> UpdateResult:
        from trading.ig import IGAllowanceExceeded, IGError, IGLoginError, IGRateLimited
        from trading.market import resolve_market, trading_dir

        res = UpdateResult(self.name, self.region)
        lock = _try_lock(trading_dir(self.settings) / "ig_download.lock")
        if lock is None:
            res.status, res.message = "skipped", "another IG download is running"
            return res
        try:
            last = None if full else self.store.last_timestamp(self.name, self.region)
            now = _utcnow()
            if last is not None:
                wait = self._refresh_wait(now)
                if wait:
                    res.status, res.message = "skipped", wait
                    return res
            resolution = self.cfg.get("resolution", "HOUR")
            market = resolve_market(self.settings, self.client)
            if last is None:
                points = int(self.cfg.get("history_points", 4000))
                self._save_refresh_state(last_request=now.isoformat())
                raw = self.client.prices(market["epic"], resolution=resolution, max_points=points)
                res.start = raw["timestamp"].min().date().isoformat() if len(raw) else None
            else:
                if now - last.tz_convert("UTC").tz_localize(None) < timedelta(minutes=50):
                    res.status, res.message = "skipped", "up to date"
                    return res
                since = (last - self.revision_lookback).tz_convert("UTC").tz_localize(None)
                self._save_refresh_state(last_request=now.isoformat())
                raw = self.client.prices(market["epic"], since, now, resolution)
                res.start = since.date().isoformat()
            res.end = now.date().isoformat()
            res.windows = 1
            df = self.validate(self.transform(raw))
            if not df.empty:
                res.rows = self.store.write(self.name, self.region, df, self.key_columns)
            self.log.info("%s: %d candles (%s)", market["epic"], len(df),
                          "history" if last is None else f"since {res.start}")
            allowance = getattr(self.client, "last_allowance", None)
            if allowance:
                self.log.info("IG history allowance: %s of %s candles left this week",
                              allowance.get("remainingAllowance"), allowance.get("totalAllowance"))
        except SourceNotImplemented as exc:
            res.status, res.message = "not_implemented", str(exc)
        except IGAllowanceExceeded as exc:
            self._save_refresh_state(blocked_until=(_utcnow() + ALLOWANCE_BACKOFF).isoformat())
            res.status, res.message = "error", (f"{exc}; stored data is still used; not asking again for "
                                                f"{ALLOWANCE_BACKOFF.total_seconds() / 3600:.0f}h")
            self.log.warning(res.message)
        except (IGLoginError, IGRateLimited, IGError, SourceFatal) as exc:
            res.status, res.message = "error", str(exc)
            self.log.error("IG prices failed: %s", exc)
        finally:
            _unlock(lock)
        return res

    # ── how often we ask IG ──────────────────────────────────────────────
    def _refresh_path(self):
        from trading.market import trading_dir

        return trading_dir(self.settings) / "ig_refresh.json"

    def _refresh_state(self) -> dict:
        try:
            return json.loads(self._refresh_path().read_text())
        except (OSError, ValueError):
            return {}

    def _save_refresh_state(self, **changes) -> None:
        state = {**self._refresh_state(), **changes}
        try:
            self._refresh_path().write_text(json.dumps(state))
        except OSError as exc:
            self.log.debug("could not save IG refresh state: %s", exc)

    def _refresh_wait(self, now: datetime) -> str | None:
        """Why this incremental refresh should wait, or None if it's due."""
        state = self._refresh_state()
        blocked = state.get("blocked_until")
        if blocked and now < datetime.fromisoformat(blocked):
            return f"IG price-history allowance used up; trying again after {blocked[11:16]} UTC"
        last_req = state.get("last_request")
        if last_req:
            nxt = datetime.fromisoformat(last_req).replace(minute=0, second=0, microsecond=0) \
                + timedelta(hours=1) + REFRESH_AFTER_HOUR
            if now < nxt:
                return f"already refreshed this hour; next at {nxt:%H:%M} UTC"
        return None

    # not used: update() above talks to IG directly (one request per run)
    def fetch(self, start: date, end: date) -> pd.DataFrame:
        from datetime import time

        from trading.market import resolve_market

        market = resolve_market(self.settings, self.client)
        return self.client.prices(market["epic"], datetime.combine(start, time()),
                                  datetime.combine(end + timedelta(days=1), time()),
                                  self.cfg.get("resolution", "HOUR"))

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=[TS])
        return raw.rename(columns={"timestamp": TS})


def _try_lock(path):
    """Non-blocking exclusive lock; None if someone else holds it."""
    try:
        import fcntl

        fh = open(path, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return None
        return fh
    except ImportError:  # no fcntl (Windows): no cross-process lock
        return open(path, "a+")


def _unlock(fh) -> None:
    try:
        import fcntl

        fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception:
        pass
    fh.close()
