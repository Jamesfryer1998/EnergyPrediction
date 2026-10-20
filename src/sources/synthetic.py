"""Synthetic GB-like data so the pipeline and dashboard can run offline (`demo` command, tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from storage import TS, ParquetStore


def generate(days: int = 400, end: pd.Timestamp | None = None, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end = (end or pd.Timestamp.now(tz="UTC")).floor("D")
    # two extra days at the end stand in for weather forecasts; realised series are cut at `end`
    idx = pd.date_range(end - pd.Timedelta(days=days), end + pd.Timedelta(days=2), freq="30min",
                        inclusive="left", tz="UTC")
    n = len(idx)
    local = idx.tz_convert("Europe/London")
    hour = local.hour + local.minute / 60
    doy = local.dayofyear.to_numpy()
    weekend = (local.dayofweek >= 5).astype(float)

    season = np.cos(2 * np.pi * (doy - 15) / 365)  # +1 mid-Jan, -1 mid-July
    temp = 10 - 7 * season + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.5, n)
    wind = np.clip(6 + 2 * season + np.cumsum(rng.normal(0, 0.15, n)) % 6 - 3 + rng.normal(0, 1, n), 0, None)
    solar = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * (500 - 250 * season) * rng.uniform(0.4, 1, n)

    daily_shape = 0.75 + 0.25 * np.sin(np.pi * np.clip(hour - 6, 0, 16) / 16) + 0.08 * np.exp(-((hour - 18) ** 2) / 2)
    demand = 28000 * daily_shape * (1 + 0.18 * season) * (1 - 0.08 * weekend) - 250 * (temp - 12)
    demand += rng.normal(0, 500, n)
    wind_gen = np.clip(wind**3 * 25, 0, 18000)
    solar_gen = solar * 25
    residual = demand - wind_gen - solar_gen
    price = 20 + residual / 300 + rng.normal(0, 6, n) + 10 * season
    system = price + rng.normal(0, 25, n)

    frames = {
        "demand": pd.DataFrame({TS: idx, "demand_itsdo": demand, "demand_indo": demand * 0.93}),
        "prices_mid": pd.DataFrame({TS: idx, "price_mid": price, "volume_mid": rng.uniform(500, 3000, n)}),
        "prices_system": pd.DataFrame(
            {TS: idx, "system_sell_price": system, "system_buy_price": system, "net_imbalance_volume": rng.normal(0, 300, n)}
        ),
        "grid_generation": pd.DataFrame(
            {TS: idx, "gen_wind": wind_gen, "gen_ccgt": np.clip(residual - 5000, 0, None), "gen_nuclear": 5000.0,
             "gen_total": demand, "renewable_share": (wind_gen + solar_gen) / demand}
        ),
    }
    realised = idx < end
    for name in list(frames):
        frames[name] = frames[name][realised].reset_index(drop=True)
    hourly = idx[::2]
    frames["weather"] = pd.DataFrame(
        {TS: hourly, "wx_mean__temperature_2m": temp[::2], "wx_mean__wind_speed_10m": wind[::2],
         "wx_mean__shortwave_radiation": solar[::2], "wx_mean__cloud_cover": rng.uniform(20, 100, len(hourly))}
    )
    hol_days = pd.to_datetime(["2025-12-25", "2025-12-26", "2026-01-01", "2026-04-03", "2026-04-06",
                               "2026-05-04", "2026-05-25", "2026-08-31", "2026-12-25"]).tz_localize("Europe/London")
    frames["calendar_events"] = pd.DataFrame(
        {TS: hol_days.tz_convert("UTC"), "is_bank_holiday": 1, "holiday_name": "Bank holiday"}
    )
    frames["ig_gas"] = synthetic_gas_candles(days, end, seed)
    return frames


def synthetic_gas_candles(days: int = 400, end: pd.Timestamp | None = None, seed: int = 7) -> pd.DataFrame:
    """Hourly bid/ask candles for a UK-gas-like CFD (pence/therm), weekdays 07:00-17:00 UTC.
    Mean-reverting with momentum bursts, so a model has something (but not too much) to find."""
    rng = np.random.default_rng(seed + 1)
    end = (end or pd.Timestamp.now(tz="UTC")).floor("h")
    idx = pd.date_range(end - pd.Timedelta(days=days), end, freq="1h", inclusive="left", tz="UTC")
    idx = idx[(idx.dayofweek < 5) & (idx.hour >= 7) & (idx.hour < 17)]
    n = len(idx)
    doy = idx.dayofyear.to_numpy()
    level = 85 + 20 * np.cos(2 * np.pi * (doy - 15) / 365)
    x = np.zeros(n)
    for i in range(1, n):  # OU deviation from the seasonal level, plus some short-term momentum
        mom = 0.25 * (x[i - 1] - x[i - 2]) if i > 1 else 0.0
        x[i] = x[i - 1] + 0.04 * (0 - x[i - 1]) + mom + rng.normal(0, 0.9)
    mid = level + x
    spread = np.clip(rng.normal(0.5, 0.08, n), 0.3, None)
    noise = np.abs(rng.normal(0, 0.4, (2, n)))
    op = np.r_[mid[0], mid[:-1]]
    df = pd.DataFrame({TS: idx, "volume": rng.integers(50, 500, n).astype(float)})
    for side, sign in (("bid", -0.5), ("ask", 0.5)):
        df[f"{side}_open"] = op + sign * spread
        df[f"{side}_close"] = mid + sign * spread
        df[f"{side}_high"] = np.maximum(op, mid) + noise[0] + sign * spread
        df[f"{side}_low"] = np.minimum(op, mid) - noise[1] + sign * spread
    df["mid_close"] = mid
    df["spread"] = spread
    return df


def populate(store: ParquetStore, region: str = "GB", days: int = 400) -> dict[str, int]:
    return {name: store.write(name, region, df) for name, df in generate(days).items()}
