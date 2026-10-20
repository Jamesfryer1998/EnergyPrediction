"""Parquet + DuckDB storage.

Layout:  <data_dir>/store/<dataset>/region=<region>/<YYYY-MM>.parquet

Every dataset is a time series with a tz-aware UTC ``timestamp`` column. Writes are
upserts: new rows replace existing rows with the same key columns, so re-downloading
a window (e.g. after Elexon revises a settlement) is always safe.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TS = "timestamp"


class ParquetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root) / "store"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── paths ────────────────────────────────────────────────────────────
    def dataset_dir(self, dataset: str, region: str) -> Path:
        return self.root / dataset / f"region={region}"

    def datasets(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def regions(self, dataset: str) -> list[str]:
        d = self.root / dataset
        return sorted(p.name.split("=", 1)[1] for p in d.glob("region=*")) if d.exists() else []

    # ── write ────────────────────────────────────────────────────────────
    def write(
        self,
        dataset: str,
        region: str,
        df: pd.DataFrame,
        key_cols: Sequence[str] = (TS,),
    ) -> int:
        """Upsert rows. Returns number of rows written."""
        if df is None or df.empty:
            return 0
        if TS not in df.columns:
            raise ValueError(f"{dataset}: dataframe has no '{TS}' column")
        df = df.copy()
        df[TS] = _to_utc(df[TS])
        key_cols = list(key_cols)
        out_dir = self.dataset_dir(dataset, region)
        out_dir.mkdir(parents=True, exist_ok=True)

        months = df[TS].dt.strftime("%Y-%m")
        with self._lock:
            for month, part in df.groupby(months, sort=True):
                path = out_dir / f"{month}.parquet"
                if path.exists():
                    existing = pd.read_parquet(path)
                    existing[TS] = _to_utc(existing[TS])
                    part = pd.concat([existing, part], ignore_index=True)
                part = (
                    part.drop_duplicates(subset=key_cols, keep="last")
                    .sort_values(key_cols)
                    .reset_index(drop=True)
                )
                tmp = path.with_suffix(".tmp")
                part.to_parquet(tmp, index=False)
                tmp.replace(path)
        log.debug("wrote %d rows to %s/%s", len(df), dataset, region)
        return len(df)

    # ── read ─────────────────────────────────────────────────────────────
    def read(
        self,
        dataset: str,
        region: str,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Rows with start <= timestamp < end (either bound optional)."""
        d = self.dataset_dir(dataset, region)
        files = sorted(d.glob("*.parquet")) if d.exists() else []
        start_ts = _ts(start)
        end_ts = _ts(end)
        if start_ts is not None:
            files = [f for f in files if f.stem >= start_ts.strftime("%Y-%m")]
        if end_ts is not None:
            files = [f for f in files if f.stem <= end_ts.strftime("%Y-%m")]
        if not files:
            return pd.DataFrame(columns=[TS, *(c for c in (columns or []) if c != TS)])
        cols = None if columns is None else list(dict.fromkeys([TS, *columns]))
        frames = [pd.read_parquet(f, columns=cols) for f in files]
        df = pd.concat(frames, ignore_index=True)
        df[TS] = _to_utc(df[TS])
        if start_ts is not None:
            df = df[df[TS] >= start_ts]
        if end_ts is not None:
            df = df[df[TS] < end_ts]
        return df.sort_values(TS).reset_index(drop=True)

    def last_timestamp(self, dataset: str, region: str) -> pd.Timestamp | None:
        d = self.dataset_dir(dataset, region)
        files = sorted(d.glob("*.parquet")) if d.exists() else []
        for f in reversed(files):
            ts = pd.read_parquet(f, columns=[TS])[TS]
            if len(ts):
                return _to_utc(ts).max()
        return None

    def first_timestamp(self, dataset: str, region: str) -> pd.Timestamp | None:
        d = self.dataset_dir(dataset, region)
        files = sorted(d.glob("*.parquet")) if d.exists() else []
        for f in files:
            ts = pd.read_parquet(f, columns=[TS])[TS]
            if len(ts):
                return _to_utc(ts).min()
        return None

    def summary(self, region: str) -> list[dict]:
        out = []
        for ds in self.datasets():
            if region not in self.regions(ds):
                continue
            files = list(self.dataset_dir(ds, region).glob("*.parquet"))
            out.append(
                {
                    "dataset": ds,
                    "first": _iso(self.first_timestamp(ds, region)),
                    "last": _iso(self.last_timestamp(ds, region)),
                    "files": len(files),
                    "bytes": sum(f.stat().st_size for f in files),
                }
            )
        return out

    # ── ad-hoc SQL ───────────────────────────────────────────────────────
    def sql(self, query: str) -> pd.DataFrame:
        """Run DuckDB SQL. Reference datasets as views named after them, e.g.
        ``store.sql("select avg(price_mid) from prices_mid where region='GB'")``.
        """
        import duckdb

        con = duckdb.connect()
        try:
            for ds in self.datasets():
                glob = (self.root / ds / "region=*" / "*.parquet").as_posix()
                con.execute(
                    f"create view {ds} as select * from read_parquet('{glob}', hive_partitioning=true)"
                )
            return con.execute(query).df()
        finally:
            con.close()


def _ts(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _to_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, utc=True)
    return s.dt.as_unit("ns") if hasattr(s.dt, "as_unit") else s


def _iso(ts: pd.Timestamp | None) -> str | None:
    return ts.isoformat() if ts is not None else None
