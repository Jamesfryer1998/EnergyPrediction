"""Calendar and weather events."""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timezone

import pandas as pd

from storage import TS
from .base import DataSource

BANK_HOLIDAYS = "https://www.gov.uk/bank-holidays.json"


class UKBankHolidays(DataSource):
    name = "calendar_events"
    description = "UK bank holidays (gov.uk), one row per holiday"
    category = "events"
    frequency = "1D"
    max_days_per_request = 100_000
    allows_future = True  # gov.uk publishes ~1 year ahead

    def resume_start(self) -> date:
        return self.default_start()

    def fetch(self, start: date, end: date) -> dict:
        return self.http_get_json(BANK_HOLIDAYS)

    def transform(self, raw: dict) -> pd.DataFrame:
        division = self.region_config.get("holiday_division", "england-and-wales")
        events = (raw or {}).get(division, {}).get("events", [])
        if not events:
            return pd.DataFrame(columns=[TS])
        df = pd.DataFrame(events)
        local = pd.to_datetime(df["date"]).dt.tz_localize(self.settings.timezone)
        return pd.DataFrame(
            {TS: local.dt.tz_convert("UTC"), "is_bank_holiday": 1, "holiday_name": df["title"]}
        )


# ── named storms ────────────────────────────────────────────────────────────

STORM_CENTRE = "https://weather.metoffice.gov.uk/warnings-and-advice/uk-storm-centre"
FIRST_SEASON = 2015  # storm naming started in 2015/16


def season_urls(start_year: int) -> list[str]:
    """The Met Office has used both 'uk-storm-season-2024-25' and 'uk-storm-season-202526'."""
    yy = str(start_year + 1)[-2:]
    return [f"{STORM_CENTRE}/uk-storm-season-{start_year}-{yy}", f"{STORM_CENTRE}/uk-storm-season-{start_year}{yy}"]


def parse_impact_dates(text: str) -> list[date]:
    """'20-21 October 2024', '27 November 2024', '30 November - 1 December 2024',
    '31 December 2024 - 2 January 2025', '3-4 Oct 2025' -> every day in the range."""
    t = re.sub(r"[–—]", "-", str(text)).replace(" to ", "-")
    t = re.sub(r"\(.*?\)|\*", "", t).strip()
    parts = [p.strip() for p in t.split("-") if p.strip()]
    if not parts:
        return []
    end = pd.to_datetime(parts[-1], dayfirst=True, errors="coerce")
    if pd.isna(end):
        return []
    start = end
    if len(parts) > 1:
        left = parts[0]
        if left.isdigit():
            start = end.replace(day=int(left))
        else:
            cand = pd.to_datetime(left, dayfirst=True, errors="coerce")
            if pd.isna(cand) or not re.search(r"\d{4}", left):  # '30 November' -> borrow the year
                cand = pd.to_datetime(f"{left} {end.year}", dayfirst=True, errors="coerce")
                if pd.notna(cand) and cand > end:
                    cand = cand.replace(year=end.year - 1)
            start = cand if pd.notna(cand) else end
    if start > end or (end - start).days > 14:
        return [end.date()]
    return [d.date() for d in pd.date_range(start, end, freq="D")]


class WeatherEvents(DataSource):
    """Named storms affecting the UK (Met Office storm centre), one row per impact day.
    Storms are named a day or two before landfall, so tomorrow's storms are known when
    predicting — a useful signal for wind output, demand and price spikes."""

    name = "weather_events"
    description = "Named storms affecting the UK, per impact day (Met Office storm centre)"
    category = "events"
    frequency = "1D"
    max_days_per_request = 100_000
    key_columns = (TS, "storm_name")
    allows_future = True

    def resume_start(self) -> date:
        # small pages; refetch the current and previous season each time
        today = datetime.now(timezone.utc).date()
        return max(self.default_start(), date(today.year - 1, 9, 1))

    def fetch(self, start: date, end: date) -> dict[int, str]:
        first = max(FIRST_SEASON, (start.year - 1) if start.month < 9 else start.year)
        last = end.year if end.month >= 9 else end.year - 1
        pages: dict[int, str] = {}
        for season in range(first, last + 1):
            for url in season_urls(season):
                try:
                    pages[season] = self.http_get_text(url)
                    break
                except Exception as exc:  # 404 for the other URL style, or no page yet
                    self.log.debug("storm season %s: %s (%s)", season, url, exc)
        if not pages:
            raise RuntimeError(f"no Met Office storm season pages found for {first}-{last}")
        return pages

    @staticmethod
    def parse_season(html: str) -> pd.DataFrame:
        rows = []
        for t in pd.read_html(io.StringIO(html)):
            cols = [str(c) for c in t.columns]
            name_col = next((c for c in cols if c.strip().lower() == "name"), None)
            impact_col = next((c for c in cols if "impact" in c.lower()), None)
            named_col = next((c for c in cols if "named" in c.lower() and "by" not in c.lower()), None)
            if not (name_col and impact_col):
                continue
            for _, r in t.iterrows():
                name = re.sub(r"\s+", " ", str(r[name_col])).strip()
                if not name or name.lower() == "nan":
                    continue
                named = pd.to_datetime(str(r[named_col]), dayfirst=True, errors="coerce") if named_col else pd.NaT
                for d in parse_impact_dates(r[impact_col]):
                    rows.append({"day": d, "storm_name": name, "named_on": named})
        return pd.DataFrame(rows)

    def transform(self, raw: dict[int, str]) -> pd.DataFrame:
        frames = [self.parse_season(html) for html in raw.values()]
        df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
        if df.empty:
            return pd.DataFrame(columns=[TS, "storm_name"])
        local = pd.to_datetime(df["day"]).dt.tz_localize(self.settings.timezone)
        return pd.DataFrame({
            TS: local.dt.tz_convert("UTC"),
            "storm_name": df["storm_name"],
            "named_storm": 1,
            "named_on": df["named_on"].dt.strftime("%Y-%m-%d").fillna(""),
        })


__all__ = ["UKBankHolidays", "WeatherEvents", "parse_impact_dates", "season_urls"]
