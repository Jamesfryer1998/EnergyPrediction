"""Join every stored source onto one half-hourly grid and derive model features.

Leakage rule: a column is either
  * known ahead (weather, calendar, carbon-intensity *forecast*) -> used at time t, or
  * realised (prices, demand, generation, ...) -> only used via lags >= features.min_lag.
With min_lag = 96 (2 days) every lag is already known when tomorrow is predicted
from this morning's data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from settings import Settings, get_settings
from storage import TS, ParquetStore

log = logging.getLogger(__name__)

FREQ = "30min"
# source -> columns usable at time t (None = all columns)
KNOWN_AHEAD: dict[str, list[str] | None] = {
    "weather": None,
    "grid_carbon_intensity": ["ci_forecast"],
}
# realised sources whose columns get lag features (beyond the targets themselves)
LAGGED_EXOG: dict[str, list[str]] = {
    "grid_generation": ["gen_wind", "gen_ccgt", "gen_total", "renewable_share", "interconnector_net"],
    "prices_system": ["net_imbalance_volume"],
    "grid_carbon_intensity": ["ci_actual"],
}
DAILY_SOURCES = ["gas_prices", "carbon_allowances", "certificates", "certificate_prices"]


@dataclass
class FeatureSet:
    frame: pd.DataFrame                      # indexed by UTC timestamp
    feature_cols: list[str]
    target_cols: dict[str, str] = field(default_factory=dict)  # target name -> column

    def xy(self, target: str, start=None, end=None, dropna: bool = True):
        col = self.target_cols[target]
        df = self.frame
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index < pd.Timestamp(end)]
        if dropna:
            df = df[df[col].notna()]
        return df[self.feature_cols], df[col]


class FeatureBuilder:
    def __init__(self, store: ParquetStore | None = None, settings: Settings | None = None, region: str = "GB"):
        self.settings = settings or get_settings()
        self.store = store or ParquetStore(self.settings.data_dir)
        self.region = region
        fcfg = self.settings.get("features", {}) or {}
        self.min_lag = int(fcfg.get("min_lag", 96))
        self.lags = sorted({int(lag) for lag in fcfg.get("lags", [96, 336]) if int(lag) >= self.min_lag})
        self.rolling = [int(w) for w in fcfg.get("rolling_windows", [48, 336])]
        if any(int(lag) < self.min_lag for lag in fcfg.get("lags", [])):
            log.warning("dropping lags shorter than min_lag=%d to avoid leakage", self.min_lag)

    # ── loading ──────────────────────────────────────────────────────────
    def _load(self, name: str, start, end) -> pd.DataFrame:
        df = self.store.read(name, self.region, start, end)
        if df.empty:
            return df
        df = df.drop_duplicates(TS, keep="last").set_index(TS).sort_index()
        return df.select_dtypes(include=[np.number, "bool"])

    def _to_grid(self, df: pd.DataFrame, grid: pd.DatetimeIndex, interpolate: bool) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=grid)
        union = df.index.union(grid)
        out = df.reindex(union)
        out = out.interpolate(method="time", limit=4, limit_area="inside") if interpolate else out
        return out.reindex(grid)

    # ── main entry ───────────────────────────────────────────────────────
    def build(self, start, end, targets: dict[str, dict] | None = None) -> FeatureSet:
        targets = targets or self.settings.targets
        start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
        max_back = max(self.lags + [self.min_lag + w for w in self.rolling] + [self.min_lag])
        load_start = start - pd.Timedelta(minutes=30 * (max_back + 48))
        grid = pd.date_range(start.floor(FREQ), end.ceil(FREQ), freq=FREQ, inclusive="left", tz="UTC")
        full_grid = pd.date_range(load_start.floor(FREQ), end.ceil(FREQ), freq=FREQ, inclusive="left", tz="UTC")
        frame = pd.DataFrame(index=full_grid)
        feature_cols: list[str] = []
        target_cols: dict[str, str] = {}

        # targets (+ their lags)
        lag_series: dict[str, pd.Series] = {}
        for tname, tcfg in targets.items():
            src, col = tcfg["source"], tcfg["column"]
            df = self._load(src, load_start, end)
            if col not in df:
                log.warning("target %s: %s.%s has no data", tname, src, col)
                frame[col] = np.nan
            else:
                frame[col] = self._to_grid(df[[col]], full_grid, interpolate=False)[col]
            target_cols[tname] = col
            lag_series[col] = frame[col]

        # realised exogenous -> lags only
        for src, cols in LAGGED_EXOG.items():
            df = self._load(src, load_start, end)
            for c in cols:
                if c in df and c not in lag_series:
                    lag_series[c] = self._to_grid(df[[c]], full_grid, interpolate=False)[c]

        lag_feats = {}
        for c, s in lag_series.items():
            for lag in self.lags:
                lag_feats[f"{c}__lag{lag}"] = s.shift(lag)
            base = s.shift(self.min_lag)
            for w in self.rolling:
                lag_feats[f"{c}__roll{w}_mean"] = base.rolling(w, min_periods=max(1, w // 2)).mean()
            lag_feats[f"{c}__roll48_std"] = base.rolling(48, min_periods=24).std()
        lag_df = pd.DataFrame(lag_feats, index=full_grid)
        frame = pd.concat([frame, lag_df], axis=1)
        feature_cols += list(lag_df.columns)

        # known-ahead exogenous
        for src, cols in KNOWN_AHEAD.items():
            df = self._load(src, load_start, end)
            if df.empty:
                continue
            if cols is not None:
                df = df[[c for c in cols if c in df]]
            # keep national means + per-point temperature/wind to limit width
            if src == "weather":
                keep = [c for c in df if c.startswith("wx_mean__") or c.endswith(("temperature_2m", "wind_speed_10m"))]
                df = df[keep]
            g = self._to_grid(df, full_grid, interpolate=True).add_prefix(f"{src}__")
            frame = pd.concat([frame, g], axis=1)
            feature_cols += list(g.columns)

        # Daily commodities, lagged two days: gas SAP for day D is only published on D+1, and a
        # prediction for D is made on D-1, so D-2 is the latest value reliably known.
        for src in DAILY_SOURCES:
            df = self._load(src, load_start - pd.Timedelta(days=60), end)
            if df.empty:
                continue
            daily = df.resample("1D").last().shift(2)
            g = daily.reindex(daily.index.union(full_grid)).ffill().reindex(full_grid).add_prefix(f"{src}__")
            frame = pd.concat([frame, g], axis=1)
            feature_cols += list(g.columns)

        cal = self._calendar(full_grid, load_start, end)
        frame = pd.concat([frame, cal], axis=1)
        feature_cols += list(cal.columns)

        frame = frame.loc[grid]
        frame.index.name = TS
        return FeatureSet(frame=frame, feature_cols=feature_cols, target_cols=target_cols)

    def _calendar(self, idx: pd.DatetimeIndex, start, end) -> pd.DataFrame:
        local = idx.tz_convert(self.settings.timezone)
        sp = local.hour * 2 + local.minute // 30
        doy = local.dayofyear
        out = pd.DataFrame(
            {
                "cal__sp": sp,
                "cal__hour": local.hour,
                "cal__dow": local.dayofweek,
                "cal__month": local.month,
                "cal__is_weekend": (local.dayofweek >= 5).astype(int),
                "cal__doy_sin": np.sin(2 * np.pi * doy / 365.25),
                "cal__doy_cos": np.cos(2 * np.pi * doy / 365.25),
                "cal__sp_sin": np.sin(2 * np.pi * sp / 48),
                "cal__sp_cos": np.cos(2 * np.pi * sp / 48),
                "cal__xmas_period": ((local.month == 12) & (local.day >= 23) | (local.month == 1) & (local.day <= 2)).astype(int),
            },
            index=idx,
        )
        hol = self.store.read("calendar_events", self.region)
        local_dates = pd.Index(local.date)
        if not hol.empty:
            hol_dates = set(pd.to_datetime(hol[TS], utc=True).dt.tz_convert(self.settings.timezone).dt.date)
            out["cal__bank_holiday"] = local_dates.isin(hol_dates).astype(int)
        else:
            out["cal__bank_holiday"] = 0
        # Named storms are announced a day or two ahead, so the flag can be used on the day itself.
        storms = self.store.read("weather_events", self.region)
        if not storms.empty:
            storm_days = set(pd.to_datetime(storms[TS], utc=True).dt.tz_convert(self.settings.timezone).dt.date)
            out["evt__named_storm"] = local_dates.isin(storm_days).astype(int)
        return out
