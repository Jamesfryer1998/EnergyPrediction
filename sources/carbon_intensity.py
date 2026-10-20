"""NESO Carbon Intensity API (free, no key). https://carbon-intensity.github.io/api-definitions/"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from storage import TS
from .base import DataSource

API = "https://api.carbonintensity.org.uk"


class CarbonIntensity(DataSource):
    name = "grid_carbon_intensity"
    description = "GB grid carbon intensity (gCO2/kWh), forecast and actual, half-hourly"
    category = "grid"
    max_days_per_request = 14
    allows_future = True  # API publishes ~48h of forecast

    def fetch(self, start: date, end: date) -> list[dict]:
        url = f"{API}/intensity/{start.isoformat()}T00:00Z/{(end + timedelta(days=1)).isoformat()}T00:00Z"
        payload = self.http_get_json(url)
        return (payload or {}).get("data", [])

    def transform(self, raw: list[dict]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(columns=[TS])
        rows = [
            {
                TS: r["from"],
                "ci_forecast": (r.get("intensity") or {}).get("forecast"),
                "ci_actual": (r.get("intensity") or {}).get("actual"),
            }
            for r in raw
        ]
        df = pd.DataFrame(rows)
        df[TS] = pd.to_datetime(df[TS], utc=True)
        for c in ("ci_forecast", "ci_actual"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
