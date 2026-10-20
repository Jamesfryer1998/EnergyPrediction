"""Elexon Insights Solution (BMRS) API — free, no key. https://bmrs.elexon.co.uk/api-documentation"""
from __future__ import annotations

from datetime import date, timedelta
import pandas as pd

from storage import TS
from .base import DataSource

BMRS = "https://data.elexon.co.uk/bmrs/api/v1"


def num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column, or all-NaN if the API didn't return it."""
    if col not in df:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


class ElexonSource(DataSource):
    abstract = True
    regions = ("GB",)

    def get_data(self, path: str, params: dict) -> list[dict]:
        payload = self.http_get_json(f"{BMRS}{path}", {**params, "format": "json"})
        if isinstance(payload, dict):
            return payload.get("data", []) or []
        return payload or []

    @staticmethod
    def frame(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df[TS] = pd.to_datetime(df["startTime"], utc=True)
        return df


class ElexonMarketIndexPrices(ElexonSource):
    """Market Index Data (MID): the GB short-term wholesale electricity reference price."""

    name = "prices_mid"
    description = "GB wholesale electricity Market Index Price (APX/EPEX MID), £/MWh, half-hourly"
    category = "prices"
    max_days_per_request = 7
    provider = "APXMIDP"  # N2EXMIDP publishes zero volumes most periods

    def fetch(self, start: date, end: date) -> list[dict]:
        return self.get_data(
            "/balancing/pricing/market-index",
            {
                "from": f"{start.isoformat()}T00:00Z",
                "to": f"{(end + timedelta(days=1)).isoformat()}T00:00Z",
                "dataProviders": self.provider,
            },
        )

    def transform(self, raw: list[dict]) -> pd.DataFrame:
        df = self.frame(raw)
        if df.empty:
            return df
        if "dataProvider" in df:
            df = df[df["dataProvider"] == self.provider]
        return pd.DataFrame(
            {
                TS: df[TS],
                "price_mid": num(df, "price"),
                "volume_mid": num(df, "volume"),
                "settlement_period": num(df, "settlementPeriod"),
            }
        )


class ElexonSystemPrices(ElexonSource):
    """Imbalance (cash-out) prices — single price since Nov 2015, so SSP == SBP."""

    name = "prices_system"
    description = "GB system sell/buy (imbalance) price and net imbalance volume, half-hourly"
    category = "prices"
    max_days_per_request = 1  # endpoint takes one settlement date
    parallel_windows = 8      # ...so fetch several days at once

    def fetch(self, start: date, end: date) -> list[dict]:
        rows: list[dict] = []
        d = start
        while d <= end:
            rows += self.get_data(f"/balancing/settlement/system-prices/{d.isoformat()}", {})
            d += timedelta(days=1)
        return rows

    def transform(self, raw: list[dict]) -> pd.DataFrame:
        df = self.frame(raw)
        if df.empty:
            return df
        return pd.DataFrame(
            {
                TS: df[TS],
                "system_sell_price": num(df, "systemSellPrice"),
                "system_buy_price": num(df, "systemBuyPrice"),
                "net_imbalance_volume": num(df, "netImbalanceVolume"),
            }
        )


class ElexonDemandOutturn(ElexonSource):
    """Initial demand outturn (INDO / ITSDO)."""

    name = "demand"
    description = "GB national demand outturn (INDO) and transmission system demand (ITSDO), MW"
    category = "demand"
    max_days_per_request = 7

    def fetch(self, start: date, end: date) -> list[dict]:
        return self.get_data(
            "/demand/outturn",
            {"settlementDateFrom": start.isoformat(), "settlementDateTo": end.isoformat()},
        )

    def transform(self, raw: list[dict]) -> pd.DataFrame:
        df = self.frame(raw)
        if df.empty:
            return df
        return pd.DataFrame(
            {
                TS: df[TS],
                "demand_indo": num(df, "initialDemandOutturn"),
                "demand_itsdo": num(df, "initialTransmissionSystemDemandOutturn"),
            }
        )


class ElexonGenerationByFuel(ElexonSource):
    """FUELHH: half-hourly generation outturn by fuel type, pivoted to one column per fuel."""

    name = "grid_generation"
    description = "GB generation by fuel type (wind, CCGT, nuclear, interconnectors…), MW, half-hourly"
    category = "grid"
    max_days_per_request = 7

    def fetch(self, start: date, end: date) -> list[dict]:
        return self.get_data(
            "/datasets/FUELHH",
            {"settlementDateFrom": start.isoformat(), "settlementDateTo": end.isoformat()},
        )

    def transform(self, raw: list[dict]) -> pd.DataFrame:
        df = self.frame(raw)
        if df.empty:
            return df
        df["generation"] = num(df, "generation")
        wide = df.pivot_table(index=TS, columns="fuelType", values="generation", aggfunc="last")
        wide.columns = [f"gen_{str(c).lower()}" for c in wide.columns]
        wide = wide.reset_index()
        gen_cols = [c for c in wide.columns if c.startswith("gen_") and not c.startswith("gen_int")]
        ic_cols = [c for c in wide.columns if c.startswith("gen_int")]
        wide["gen_total"] = wide[gen_cols].sum(axis=1, min_count=1)
        wide["interconnector_net"] = wide[ic_cols].sum(axis=1, min_count=1) if ic_cols else pd.NA
        renew = [c for c in ("gen_wind", "gen_npshyd", "gen_biomass") if c in wide]
        if renew:
            wide["renewable_share"] = wide[renew].sum(axis=1) / wide["gen_total"]
        return wide
