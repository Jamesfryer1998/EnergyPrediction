"""Abstract base class every data source inherits from.

A subclass declares a few class attributes and implements two methods:

    class MySource(DataSource):
        name = "my_source"                 # storage key + config key
        description = "What this is"
        frequency = "30min"                # native resolution after transform()
        max_days_per_request = 7           # API window limit -> update() chunks for you

        def fetch(self, start, end) -> pd.DataFrame | list | dict:   # raw download
        def transform(self, raw) -> pd.DataFrame:                  # tidy: timestamp + value cols

Everything else (incremental updates, chunking, retries, validation, de-duplication,
storage and loading) is handled here, so every source behaves identically.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, ClassVar

import pandas as pd
import requests

from settings import Settings, get_settings
from storage import TS, ParquetStore

log = logging.getLogger(__name__)

REGISTRY: dict[str, type[DataSource]] = {}


class SourceNotImplemented(NotImplementedError):
    """Raised by stub sources that don't have an automated feed yet."""


class SourceFatal(RuntimeError):
    """A failure that will hit every remaining window too (bad login, rate limit): stop this
    source's run instead of retrying window after window."""


@dataclass
class UpdateResult:
    source: str
    region: str
    rows: int = 0
    windows: int = 0
    start: str | None = None
    end: str | None = None
    status: str = "ok"          # ok | skipped | not_implemented | error
    message: str = ""


class DataSource(ABC):
    # ── declared by subclasses ───────────────────────────────────────────
    name: ClassVar[str]
    description: ClassVar[str] = ""
    category: ClassVar[str] = "other"   # prices | weather | demand | grid | events | commodities
    frequency: ClassVar[str] = "30min"
    max_days_per_request: ClassVar[int] = 7
    key_columns: ClassVar[tuple[str, ...]] = (TS,)
    regions: ClassVar[tuple[str, ...]] = ("GB",)
    implemented: ClassVar[bool] = True
    # How far back to re-download on each incremental run (late revisions / settlement runs)
    revision_lookback: ClassVar[timedelta] = timedelta(days=2)
    # Data the source publishes ahead of time (e.g. forecasts) can extend past "now"
    allows_future: ClassVar[bool] = False

    # API windows fetched concurrently (big speed-up on backfills; incremental runs are 1-2 windows)
    parallel_windows: ClassVar[int] = 4
    http_timeout: ClassVar[float] = 30.0
    http_retries: ClassVar[int] = 4

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None) or cls.__dict__.get("abstract", False):
            return  # intermediate base class, not a concrete source
        if not getattr(cls, "name", None):
            raise TypeError(f"{cls.__name__} must define a class attribute 'name'")
        if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
            raise TypeError(f"duplicate data source name '{cls.name}'")
        REGISTRY[cls.name] = cls

    def __init__(
        self,
        region: str = "GB",
        store: ParquetStore | None = None,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ):
        if region not in self.regions:
            raise ValueError(f"{self.name} does not support region {region!r} (supports {self.regions})")
        self.region = region
        self.settings = settings or get_settings()
        self.store = store or ParquetStore(self.settings.data_dir)
        self._session = session
        self._local = threading.local()
        self.log = logging.getLogger(f"source.{self.name}")

    # ── subclass contract ────────────────────────────────────────────────
    @abstractmethod
    def fetch(self, start: date, end: date) -> Any:
        """Download raw data for [start, end] inclusive of both dates (UTC calendar days)."""

    @abstractmethod
    def transform(self, raw: Any) -> pd.DataFrame:
        """Turn raw data into a tidy frame: tz-aware UTC 'timestamp' + numeric value columns."""

    # ── config helpers ───────────────────────────────────────────────────
    @property
    def config(self) -> dict:
        return self.settings.source_cfg(self.name)

    @property
    def region_config(self) -> dict:
        return self.settings.region(self.region)

    @property
    def session(self) -> requests.Session:
        """One HTTP session per thread (windows and weather points are fetched concurrently)."""
        if self._session is not None:
            return self._session
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = self._local.session = requests.Session()
            sess.headers["User-Agent"] = "EnergyPrediction/0.1"
        return sess

    # ── HTTP helper with retry/backoff ───────────────────────────────────
    def http_request(self, url: str, params: dict | None = None, method: str = "GET",
                     json_body: Any = None) -> requests.Response:
        """HTTP with retry/backoff on connection errors, 429 and 5xx. 4xx raises immediately."""
        last_exc: Exception | None = None
        for attempt in range(self.http_retries):
            try:
                resp = self.session.request(method, url, params=params, json=json_body, timeout=self.http_timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} from {url}", response=resp)
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_exc = exc
                wait = 2**attempt
                self.log.warning("%s %s failed (%s); retry %d in %ss", method, url, exc, attempt + 1, wait)
                time.sleep(wait)
        raise RuntimeError(f"{self.name}: giving up on {url}") from last_exc

    def http_get_json(self, url: str, params: dict | None = None) -> Any:
        return self.http_request(url, params).json()

    def http_get_text(self, url: str, params: dict | None = None) -> str:
        return self.http_request(url, params).text

    def http_get_bytes(self, url: str, params: dict | None = None) -> bytes:
        return self.http_request(url, params).content

    # ── validation ───────────────────────────────────────────────────────
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[TS])
        if TS not in df.columns:
            raise ValueError(f"{self.name}.transform() must return a '{TS}' column")
        df = df.copy()
        df[TS] = pd.to_datetime(df[TS], utc=True)
        df = df.dropna(subset=[TS])
        missing = [c for c in self.key_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: missing key columns {missing}")
        if not self.allows_future:
            df = df[df[TS] <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)]
        df = df.drop_duplicates(subset=list(self.key_columns), keep="last")
        return df.sort_values(list(self.key_columns)).reset_index(drop=True)

    # ── windows ──────────────────────────────────────────────────────────
    def windows(self, start: date, end: date) -> Iterator[tuple[date, date]]:
        """Split [start, end] (inclusive) into chunks the API accepts."""
        step = max(1, int(self.max_days_per_request))
        cur = start
        while cur <= end:
            chunk_end = min(end, cur + timedelta(days=step - 1))
            yield cur, chunk_end
            cur = chunk_end + timedelta(days=1)

    def default_start(self) -> date:
        return date.fromisoformat(str(self.settings.get("history_start", "2023-01-01")))

    def resume_start(self) -> date:
        last = self.store.last_timestamp(self.name, self.region)
        if last is None:
            return self.default_start()
        return (last - self.revision_lookback).date()

    # ── public API ───────────────────────────────────────────────────────
    def update(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        full: bool = False,
    ) -> UpdateResult:
        """Incrementally download, validate and store. Safe to re-run."""
        res = UpdateResult(self.name, self.region)
        if not self.implemented:
            res.status, res.message = "not_implemented", self.not_implemented_reason()
            return res
        start_d = _as_date(start) or (self.default_start() if full else self.resume_start())
        end_d = _as_date(end) or datetime.now(timezone.utc).date()
        if start_d > end_d:
            res.status, res.message = "skipped", "up to date"
            return res
        res.start, res.end = start_d.isoformat(), end_d.isoformat()
        windows = list(self.windows(start_d, end_d))
        workers = max(1, min(int(self.parallel_windows), len(windows)))
        batch_size = max(8, workers * 4)  # windows per store write (fewer partition rewrites)

        fatal: list[str] = []

        def get(window):
            if fatal:  # an earlier window hit a fatal error: don't call the API again
                raise SourceFatal(fatal[0])
            w_start, w_end = window
            try:
                return self.validate(self.transform(self.fetch(w_start, w_end)))
            except SourceFatal as exc:
                fatal.append(str(exc))
                raise

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"ingest-{self.name}") as pool:
            for b in range(0, len(windows), batch_size):
                batch = windows[b : b + batch_size]
                frames = []
                for (w_start, w_end), fut in zip(batch, [pool.submit(get, w) for w in batch]):
                    try:
                        df = fut.result()
                    except SourceNotImplemented as exc:
                        res.status, res.message = "not_implemented", str(exc)
                        return res
                    except SourceFatal as exc:
                        if res.status != "error" or not res.message.startswith("stopped"):
                            self.log.error("%s %s..%s failed, stopping this run: %s", self.region, w_start, w_end, exc)
                        res.status, res.message = "error", f"stopped at {w_start}: {exc}"
                        continue
                    except Exception as exc:  # keep going; one bad window shouldn't kill the run
                        self.log.error("%s window %s..%s failed: %s", self.region, w_start, w_end, exc)
                        res.status, res.message = "error", f"{w_start}..{w_end}: {exc}"
                        continue
                    res.windows += 1
                    if not df.empty:
                        frames.append(df)
                if frames:
                    merged = pd.concat(frames, ignore_index=True)
                    frames = []
                    res.rows += self.store.write(self.name, self.region, merged, self.key_columns)
                    self.log.info("%s %s..%s: %d rows (%d windows)", self.region, batch[0][0], batch[-1][1],
                                  len(merged), len(batch))
                if fatal:
                    break
        return res

    def load(self, start=None, end=None, columns=None) -> pd.DataFrame:
        return self.store.read(self.name, self.region, start, end, columns)

    def not_implemented_reason(self) -> str:
        return f"{self.name} has no automated feed yet"

    @classmethod
    def info(cls) -> dict:
        return {
            "name": cls.name,
            "class": cls.__name__,
            "description": cls.description,
            "category": cls.category,
            "frequency": cls.frequency,
            "implemented": cls.implemented,
            "regions": list(cls.regions),
        }


class ManualCsvSource(DataSource):
    """Base for sources without a free API: reads data/manual/<name>.csv if present.

    The CSV needs a timestamp/date column plus value columns. Subclasses may override
    ``fetch`` with a real API later without touching anything else.
    """

    abstract = True  # not registered itself; concrete subclasses are
    implemented = True
    max_days_per_request = 100_000  # one window: the whole file
    csv_hint: ClassVar[str] = ""

    @property
    def csv_path(self):
        return self.settings.data_dir / "manual" / f"{self.name}.csv"

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise SourceNotImplemented(
                f"{self.name}: no automated feed; drop a CSV at {self.csv_path}. {self.csv_hint}"
            )
        df = pd.read_csv(self.csv_path)
        ts_col = next((c for c in ("timestamp", "date", "datetime", "time") if c in df.columns), None)
        if ts_col is None:
            raise ValueError(f"{self.csv_path} needs a timestamp/date column")
        df = df.rename(columns={ts_col: TS})
        df[TS] = pd.to_datetime(df[TS], utc=True)
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        return df[(df[TS] >= lo) & (df[TS] < hi)]

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return raw

    def resume_start(self) -> date:
        return self.default_start()  # files are small; always re-read in full


def _as_date(value) -> date | None:
    if value is None or isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))
