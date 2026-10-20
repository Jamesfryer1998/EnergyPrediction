"""Hourly bid/ask candles for the traded market (UK natural gas by default) from IG's demo API."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from storage import TS
from .base import DataSource, SourceFatal, SourceNotImplemented, UpdateResult

log = logging.getLogger(__name__)


class IGGasPrices(DataSource):
    """IG CFD quotes for the paper-trading market: hourly bid/ask OHLC, mid and spread.

    IG limits requests per minute and candles per week, so this source makes ONE request per
    run instead of walking week-by-week windows:
      * first run: the latest `trading.history_points` candles (whatever history IG has);
      * afterwards: everything since the last stored candle (minus a few hours for revisions).
    Only one download runs at a time on this machine (a data refresh, a backtest and a trading
    cycle can all ask at once); the others skip and use what's stored."""

    name = "ig_gas"
    description = "IG demo: hourly bid/ask candles for the paper-trading market (UK natural gas)"
    category = "trading"
    frequency = "1h"
    revision_lookback = timedelta(hours=6)

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
            resolution = self.cfg.get("resolution", "HOUR")
            market = resolve_market(self.settings, self.client)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if last is None:
                points = int(self.cfg.get("history_points", 4000))
                raw = self.client.prices(market["epic"], resolution=resolution, max_points=points)
                res.start = raw["timestamp"].min().date().isoformat() if len(raw) else None
            else:
                if now - last.tz_convert("UTC").tz_localize(None) < timedelta(minutes=50):
                    res.status, res.message = "skipped", "up to date"
                    return res
                since = (last - self.revision_lookback).tz_convert("UTC").tz_localize(None)
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
            res.status, res.message = "error", f"{exc}; stored data is still used"
            self.log.warning(res.message)
        except (IGLoginError, IGRateLimited, IGError, SourceFatal) as exc:
            res.status, res.message = "error", str(exc)
            self.log.error("IG prices failed: %s", exc)
        finally:
            _unlock(lock)
        return res

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
