"""Weather from Open-Meteo (free, no key).

* Archive API (ERA5 reanalysis) for history — lags real time by ~5 days.
* Forecast API for the last few days and the next 2 days, so tomorrow's prediction
  has weather inputs. When the archive catches up it overwrites the forecast rows.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from storage import TS
from .base import DataSource

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 6
DEFAULT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "precipitation",
]


class OpenMeteoWeather(DataSource):
    name = "weather"
    description = "Hourly weather at configured points + national averages (Open-Meteo ERA5 + forecast)"
    category = "weather"
    frequency = "1h"
    max_days_per_request = 90
    parallel_windows = 2  # x4 points each; keeps well inside Open-Meteo's free rate limit
    allows_future = True
    revision_lookback = timedelta(days=ARCHIVE_LAG_DAYS + 1)

    @property
    def points(self) -> dict[str, list[float]]:
        return self.region_config.get("weather_points", {})

    @property
    def variables(self) -> list[str]:
        return self.config.get("variables", DEFAULT_VARS)

    def update(self, start=None, end=None, full=False):
        # Always pull through tomorrow+1 so next-day predictions have weather inputs.
        if end is None:
            end = datetime.now(timezone.utc).date() + timedelta(days=2)
        return super().update(start, end, full)

    def fetch(self, start: date, end: date) -> dict[str, list[dict]]:
        from concurrent.futures import ThreadPoolExecutor

        cutoff = datetime.now(timezone.utc).date() - timedelta(days=ARCHIVE_LAG_DAYS)

        def one(item):
            point, (lat, lon) = item
            chunks = []
            if start <= cutoff:
                chunks.append(self._call(ARCHIVE, lat, lon, start, min(end, cutoff)))
            if end > cutoff:
                chunks.append(self._call(FORECAST, lat, lon, max(start, cutoff + timedelta(days=1)), end))
            return point, chunks

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="weather") as pool:  # points in parallel
            return dict(pool.map(one, self.points.items()))

    def _call(self, url: str, lat: float, lon: float, start: date, end: date) -> dict:
        return self.http_get_json(
            url,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": ",".join(self.variables),
                "timezone": "UTC",
                "wind_speed_unit": "ms",
            },
        )

    def transform(self, raw: dict[str, list[dict]]) -> pd.DataFrame:
        frames = []
        for point, chunks in raw.items():
            parts = []
            for payload in chunks:
                hourly = (payload or {}).get("hourly") or {}
                if not hourly.get("time"):
                    continue
                part = pd.DataFrame(hourly)
                part[TS] = pd.to_datetime(part.pop("time"), utc=True)
                parts.append(part)
            if not parts:
                continue
            df = pd.concat(parts).drop_duplicates(TS, keep="last").set_index(TS)
            frames.append(df.add_prefix(f"{point}__"))
        if not frames:
            return pd.DataFrame(columns=[TS])
        wide = pd.concat(frames, axis=1).sort_index()
        for var in self.variables:
            cols = [c for c in wide.columns if c.endswith(f"__{var}")]
            if cols:
                wide[f"wx_mean__{var}"] = wide[cols].mean(axis=1)
        return wide.reset_index()
