"""Gas, carbon allowance and renewable-certificate prices.

Gas      — GB System Average Price (SAP) of gas, daily. Tries National Gas Transmission's
           operational-data REST API first; if the SAP series can't be found there, falls back
           to the ONS "System Average Price (SAP) of gas" spreadsheets (sourced from National Gas,
           updated roughly weekly).
Carbon   — EU allowances: EEX primary-auction results (daily on auction days, EUR/t).
           UK allowances: GOV.UK UK ETS monthly average carbon price (GBP/t).
           EUR is converted to GBP with ECB reference rates so the two can be compared.
Certs    — REGO prices aren't published by Ofgem (only certificate volumes are), so this
           stays a manual CSV of auction/broker prices.

None of these sources are fetched by this sandboxed build environment; use
``python src/main.py probe <name>`` to test a source against the live sites.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from storage import TS
from .base import DataSource, ManualCsvSource, SourceNotImplemented

KWH_PER_THERM = 29.3071

# ── helpers ───────────────────────────────────────────────────────────────


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def find_header_table(raw: pd.DataFrame, must: tuple[str, ...]) -> pd.DataFrame | None:
    """Spreadsheets often have notes above the table. Find the first row whose cells contain
    every keyword in `must` (case-insensitive substring), use it as the header and return the
    rows below it."""
    for i in range(min(len(raw), 60)):
        cells = [_norm(c) for c in raw.iloc[i].tolist()]
        if all(any(k in c for c in cells) for k in must):
            body = raw.iloc[i + 1 :].copy()
            body.columns = [str(c).strip() for c in raw.iloc[i].tolist()]
            return body.dropna(how="all")
    return None


def pick_column(columns, include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> str | None:
    for c in columns:
        n = _norm(c)
        if any(k in n for k in include) and not any(k in n for k in exclude):
            return c
    return None


def smart_dates(values) -> pd.Series:
    """Spreadsheet dates arrive as real datetimes, ISO strings ('2026-09-01') or UK strings
    ('01/09/2026'). dayfirst=True would mangle ISO strings, so only use it for the others."""
    s = pd.Series(values)

    def one(v):
        if isinstance(v, (pd.Timestamp, datetime, date)):
            return pd.Timestamp(v)
        txt = str(v).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}", txt):
            return pd.to_datetime(txt[:10], errors="coerce")
        return pd.to_datetime(txt, dayfirst=True, errors="coerce")

    return pd.to_datetime(s.map(one), errors="coerce")


def local_day_to_utc(dates: pd.Series, tz: str) -> pd.Series:
    d = smart_dates(dates)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize().dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


# ── gas ───────────────────────────────────────────────────────────────────

NG_API = "https://api.nationalgas.com/operationaldata/v1/publications"
ONS_SAP_PAGE = ("https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/"
                "systemaveragepricesapofgas")
SAP_NAME = re.compile(r"system average price", re.I)


_DATE_WORD = re.compile(r"\b(date|gas day|applicable for|applicable date)\b")
_SAP_WORDS = ("sap", "system average", "actual day")
_NOT_DAILY = ("7", "seven", "rolling", "average of", "30", "month")


def _sap_from_grid(grid: pd.DataFrame) -> pd.DataFrame | None:
    """Try every header-like row in a sheet; return the best (date, sap) table or None."""
    best = None
    cell = lambda v: "" if pd.isna(v) else _norm(v)  # noqa: E731
    for i in range(min(len(grid), 200)):
        # headers can span two rows, e.g. National Gas's "SAP, Actual Day" over "Value (p/kWh)"
        above = [cell(v) for v in grid.iloc[i - 1].tolist()] if i else [""] * grid.shape[1]
        header = [f"{a} {cell(v)}".strip() for a, v in zip(above, grid.iloc[i].tolist())]
        date_idx = [j for j, h in enumerate(header) if _DATE_WORD.search(h)]
        if not date_idx:
            continue
        price_idx = ([j for j, h in enumerate(header) if any(k in h for k in _SAP_WORDS) and not any(k in h for k in _NOT_DAILY)]
                     or [j for j, h in enumerate(header) if ("price" in h or "pence" in h or "p/kwh" in h)
                         and not any(k in h for k in _NOT_DAILY)])
        price_idx = [j for j in price_idx if j not in date_idx]
        if not price_idx:
            continue
        body = grid.iloc[i + 1:]
        dates = smart_dates(body.iloc[:, date_idx[0]].tolist())
        price = pd.to_numeric(body.iloc[:, price_idx[0]], errors="coerce").to_numpy()
        out = pd.DataFrame({"date": dates.to_numpy(), "sap": price}).dropna()
        out = out[(out["sap"] > 0) & (out["date"].dt.year >= 1990)]
        if len(out) >= 3 and (best is None or len(out) > len(best)):
            out = out.reset_index(drop=True)
            out.attrs["unit"] = "p/therm" if "therm" in header[price_idx[0]] else "p/kWh"
            best = out
    return best


def _describe_workbook(sheets: dict) -> str:
    """Short summary of a workbook for error messages, so a changed layout can be fixed."""
    parts = []
    for name, raw in sheets.items():
        rows = []
        for i in range(min(len(raw), 40)):
            cells = [str(c)[:40] for c in raw.iloc[i].tolist() if pd.notna(c)]
            if len(cells) >= 2:  # header/data-like rows only
                rows.append(f"r{i}: " + " | ".join(cells[:5]))
            if len(rows) >= 3:
                break
        parts.append(f"[{name} {raw.shape[0]}x{raw.shape[1]}] " + " / ".join(rows))
    return "; ".join(parts)[:1500]


class GasPrices(DataSource):
    """GB System Average Price of gas (the average price of gas traded on the On-the-day
    Commodity Market) — the standard daily GB gas reference price, close to NBP day-ahead."""

    name = "gas_prices"
    description = "GB gas System Average Price (SAP), p/kWh and p/therm, daily (National Gas / ONS)"
    category = "commodities"
    frequency = "1D"
    max_days_per_request = 100_000  # one pass; the fetch splits API calls itself
    revision_lookback = timedelta(days=10)
    api_chunk_days = 31

    def fetch(self, start: date, end: date) -> dict:
        errors = []
        try:
            pub = self._find_sap_publication()
            if pub:
                return {"kind": "national_gas", "name": pub[1], "rows": self._fetch_national_gas(pub[0], start, end)}
            errors.append("National Gas catalogue has no 'System Average Price' publication")
        except Exception as exc:  # API down / changed: fall back
            errors.append(f"National Gas API: {exc}")
        self.log.info("gas: %s; using ONS spreadsheets", "; ".join(errors))
        return {"kind": "ons", "files": self._fetch_ons_files(start, end), "notes": errors}

    # National Gas ---------------------------------------------------------
    def _find_sap_publication(self) -> tuple[str, str] | None:
        catalogue = self.http_get_json(f"{NG_API}/catalogue")
        found: list[tuple[str, str]] = []

        def walk(node):
            if isinstance(node, dict):
                name, pid = node.get("name") or node.get("publicationName"), node.get("publicationId")
                if pid and name and SAP_NAME.search(str(name)):
                    found.append((pid, str(name)))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(catalogue)
        # prefer the plain daily SAP over e.g. 7-day averages
        found.sort(key=lambda p: (("7" in p[1]) or ("average of" in p[1].lower()), len(p[1])))
        return found[0] if found else None

    def _fetch_national_gas(self, pid: str, start: date, end: date) -> list[dict]:
        rows: list[dict] = []
        cur = start
        while cur <= end:
            to = min(end, cur + timedelta(days=self.api_chunk_days - 1))
            body = {"fromDate": cur.isoformat(), "toDate": to.isoformat(), "publicationIds": [pid], "latestValue": "N"}
            resp = self.http_request(f"{NG_API}/gasday", method="POST", json_body=body).json()
            for pub in resp if isinstance(resp, list) else [resp]:
                rows += pub.get("publications", []) or []
            cur = to + timedelta(days=1)
        return rows

    # ONS --------------------------------------------------------------------
    def _fetch_ons_files(self, start: date, end: date) -> list[bytes]:
        html = self.http_get_text(ONS_SAP_PAGE)
        links = sorted(set(re.findall(r'href="([^"]*systemaveragepricesapofgas/(\d{4})/[^"]+\.xlsx)"', html)),
                       key=lambda t: t[1])
        if not links:
            raise RuntimeError("no xlsx links found on the ONS SAP dataset page (layout changed?)")
        # each yearly edition may hold several years; read the ones covering the window
        wanted = [href for href, year in links if int(year) >= start.year - 1] or [links[-1][0]]
        files = []
        for href in wanted:
            url = href if href.startswith("http") else f"https://www.ons.gov.uk{href}"
            files.append(self.http_get_bytes(url))
        return files

    @staticmethod
    def parse_ons_workbook(content: bytes) -> pd.DataFrame:
        """Daily SAP from an ONS workbook. ONS sheets have a cover, contents and notes, and
        notes rows above each table (which can contain words like "updated"), so every row
        that looks like a header is tried, and the candidate with the most valid
        (date, price) rows wins. Tables laid out with dates across the columns are handled too."""
        import warnings

        with warnings.catch_warnings():  # ONS workbooks have no default style; harmless openpyxl warning
            warnings.simplefilter("ignore", UserWarning)
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
        best: pd.DataFrame | None = None
        for _, raw in sheets.items():
            for grid in (raw, raw.T.reset_index(drop=True)):
                cand = _sap_from_grid(grid)
                if cand is not None and (best is None or len(cand) > len(best)):
                    best = cand
        if best is None:
            raise RuntimeError("couldn't find a date + SAP column in the ONS workbook. Layout: " + _describe_workbook(sheets))
        return best

    def transform(self, raw: dict) -> pd.DataFrame:
        tz = self.settings.timezone
        if raw["kind"] == "national_gas":
            df = pd.DataFrame(raw["rows"])
            if df.empty:
                return pd.DataFrame(columns=[TS])
            if "applicableAt" in df:
                df = df.sort_values("applicableAt")
            df = df.drop_duplicates("applicableFor", keep="last")
            pkwh = pd.to_numeric(df["value"], errors="coerce")
            ts = local_day_to_utc(df["applicableFor"], tz)
        else:
            frames = []
            for content in raw["files"]:
                part = self.parse_ons_workbook(content)
                factor = 1 / KWH_PER_THERM if part.attrs.get("unit") == "p/therm" else 1.0
                frames.append(pd.DataFrame({"date": part["date"], "pkwh": part["sap"] * factor}))
            df = pd.concat(frames, ignore_index=True)
            ts = local_day_to_utc(df["date"], tz)
            pkwh = df["pkwh"]
        out = pd.DataFrame({TS: ts.to_numpy(), "gas_sap_pkwh": pkwh.to_numpy()}).dropna()
        out["gas_sap_ptherm"] = out["gas_sap_pkwh"] * KWH_PER_THERM
        out["gas_sap_7d_avg_pkwh"] = out.sort_values(TS)["gas_sap_pkwh"].rolling(7, min_periods=1).mean()
        return out.drop_duplicates(TS, keep="last")


# ── carbon ────────────────────────────────────────────────────────────────

EEX_REPORT = "https://public.eex-group.com/eex/eua-auction-report/emission-spot-primary-market-auction-report-{year}-data.xlsx"
UKETS_PAGE = ("https://www.gov.uk/government/publications/taking-part-in-the-uk-emissions-trading-scheme-markets/"
              "cost-containment-mechanism-ccm-trigger-prices-and-average-monthly-prices-full-table")
ECB_GBP_PER_EUR = "https://data-api.ecb.europa.eu/service/data/EXR/D.GBP.EUR.SP00.A"
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


class CarbonAllowances(DataSource):
    """EU and UK carbon allowance prices side by side (in GBP) so they can be compared:
    - EUA: EEX primary auction clearing prices (official; auction days only)
    - UKA: GOV.UK UK ETS monthly average carbon price (official; monthly)
    Each sub-source is optional: if one site fails the others are still stored."""

    name = "carbon_allowances"
    description = "EU (EEX auctions) and UK (GOV.UK monthly average) carbon allowance prices, EUR/t and GBP/t"
    category = "commodities"
    frequency = "1D"
    max_days_per_request = 100_000
    revision_lookback = timedelta(days=60)  # UK monthly averages are published ~6 weeks after month start

    def fetch(self, start: date, end: date) -> dict:
        out: dict = {"errors": {}}
        eex = {}
        for y in range(max(start.year, 2020), end.year + 1):
            try:
                eex[y] = self.http_get_bytes(EEX_REPORT.format(year=y))
            except Exception as exc:  # e.g. this year's report not published yet in early January
                out["errors"][f"eua {y}"] = str(exc)
        if eex:
            out["eex"] = eex
        try:
            out["ukets_html"] = self.http_get_text(UKETS_PAGE)
        except Exception as exc:
            out["errors"]["uka"] = str(exc)
        try:
            out["fx_csv"] = self.http_get_text(ECB_GBP_PER_EUR, {
                "format": "csvdata", "startPeriod": (start - timedelta(days=40)).isoformat(), "endPeriod": end.isoformat()})
        except Exception as exc:
            out["errors"]["fx"] = str(exc)
        if "eex" not in out and "ukets_html" not in out:
            raise RuntimeError(f"all carbon sources failed: {out['errors']}")
        for k, v in out["errors"].items():
            self.log.warning("carbon %s failed: %s", k, v)
        return out

    # parsers (static so they can be unit-tested with saved files) ---------
    @staticmethod
    def parse_eex(content: bytes) -> pd.DataFrame:
        raw = pd.read_excel(io.BytesIO(content), header=None)
        table = find_header_table(raw, ("date", "price"))
        if table is None:
            raise RuntimeError("EEX report: header row with Date and Price not found")
        date_col = pick_column(table.columns, ("date",))
        price_col = pick_column(table.columns, ("auction price", "price"), ("minimum", "maximum", "min", "max"))
        name_col = pick_column(table.columns, ("auction name", "contract", "product"))
        vol_col = pick_column(table.columns, ("auction volume", "volume"), ("bid", "submitted"))
        df = pd.DataFrame({"date": smart_dates(table[date_col]).to_numpy(),
                           "price": pd.to_numeric(table[price_col], errors="coerce")})
        df["volume"] = pd.to_numeric(table[vol_col], errors="coerce").to_numpy() if vol_col else 1.0
        if name_col:  # keep standard EUAs; drop aviation (EUAA) and ETS2 auctions
            names = table[name_col].astype(str).str.lower().to_numpy()
            df = df[~pd.Series(names, index=df.index).str.contains("euaa|aviation|ets2|ets 2", regex=True)]
        df = df.dropna(subset=["date", "price"])
        df["volume"] = df["volume"].fillna(1.0)
        return (df.groupby(df["date"].dt.normalize())
                  .apply(lambda g: np.average(g["price"], weights=g["volume"]) if g["volume"].sum() > 0 else g["price"].mean(),
                         include_groups=False)
                  .rename("eua_auction_eur").reset_index())

    @staticmethod
    def parse_ukets(html: str) -> pd.DataFrame:
        """Rebuild the monthly UKA average price series from the CCM tables: each row lists
        the months of its monitoring period and their average prices."""
        tables = pd.read_html(io.StringIO(html))
        prices: dict[pd.Timestamp, float] = {}
        for t in tables:
            cols = [str(c) for c in t.columns]
            mon_col = next((c for c in cols if _norm(c) == "month"), None)
            period_col = next((c for c in cols if "monitoring period" in _norm(c)), None)
            avg_cols = [c for c in cols if re.match(r"month \d+ average price", _norm(c))]
            if not (mon_col and period_col and avg_cols):
                continue
            avg_cols.sort(key=lambda c: int(re.search(r"\d+", c).group()))
            for _, row in t.iterrows():
                pub = pd.to_datetime(str(row[mon_col]).strip(), format="%b-%y", errors="coerce")
                if pd.isna(pub):
                    continue
                months = [m.strip()[:3].lower() for m in str(row[period_col]).split(",") if m.strip()]
                n = len(months)
                for k, (m, c) in enumerate(zip(months, avg_cols)):
                    month_start = (pub - pd.DateOffset(months=n - k)).normalize()
                    if MONTHS.get(m) != month_start.month:
                        continue  # table/period mismatch; skip rather than guess
                    val = pd.to_numeric(str(row[c]).replace("£", "").replace(",", "").strip(), errors="coerce")
                    if pd.notna(val):
                        prices[month_start] = float(val)
        if not prices:
            raise RuntimeError("UK ETS page: no monthly average prices found (layout changed?)")
        return pd.DataFrame({"month": list(prices), "uka_avg_gbp": list(prices.values())}).sort_values("month")

    @staticmethod
    def parse_ecb(csv_text: str) -> pd.Series:
        df = pd.read_csv(io.StringIO(csv_text))
        s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(),
                      index=pd.to_datetime(df["TIME_PERIOD"]), name="gbp_per_eur").dropna()
        return s.sort_index()

    def transform(self, raw: dict) -> pd.DataFrame:
        tz = self.settings.timezone
        fx = None
        if raw.get("fx_csv"):
            try:
                fx = self.parse_ecb(raw["fx_csv"])
            except Exception as exc:
                self.log.warning("ECB FX parse failed: %s", exc)
        frames = []
        eua_daily = pd.DataFrame()
        if raw.get("eex"):
            eua_daily = pd.concat([self.parse_eex(c) for c in raw["eex"].values()], ignore_index=True)
            eua_daily = eua_daily.drop_duplicates("date", keep="last").sort_values("date")
            if fx is not None and len(fx):
                rate = fx.reindex(fx.index.union(eua_daily["date"])).ffill().reindex(eua_daily["date"]).to_numpy()
                eua_daily["eua_auction_gbp"] = eua_daily["eua_auction_eur"] * rate
            # auction results are public the same day; timestamp at local end of day to be safe
            frames.append(pd.DataFrame({TS: local_day_to_utc(eua_daily["date"], tz) + pd.Timedelta(hours=18),
                                        **{c: eua_daily[c].to_numpy() for c in eua_daily.columns if c != "date"}}))
        if raw.get("ukets_html"):
            uka = self.parse_ukets(raw["ukets_html"])
            # compare like with like: EUA monthly average in GBP for the same month
            if len(eua_daily) and "eua_auction_gbp" in eua_daily:
                m = eua_daily.groupby(eua_daily["date"].dt.to_period("M"))["eua_auction_gbp"].mean()
                uka["eua_month_avg_gbp"] = uka["month"].dt.to_period("M").map(m).astype(float)
                uka["uka_minus_eua_gbp"] = uka["uka_avg_gbp"] - uka["eua_month_avg_gbp"]
            # A month's average is only known after it ends (GOV.UK updates around the 12th),
            # so stamp it at the 15th of the following month to avoid look-ahead.
            avail = uka["month"] + pd.DateOffset(months=1) + pd.Timedelta(days=14)
            frames.append(pd.DataFrame({TS: local_day_to_utc(avail, tz),
                                        **{c: uka[c].to_numpy() for c in uka.columns if c != "month"},
                                        "uka_month": uka["month"].dt.strftime("%Y-%m").to_numpy()}))
        if not frames:
            return pd.DataFrame(columns=[TS])
        out = pd.concat(frames, ignore_index=True).dropna(subset=[TS])
        return out.groupby(TS, as_index=False).first()


class CertificatePrices(ManualCsvSource):
    name = "certificate_prices"
    description = "REGO certificate prices (manual CSV: Ofgem publishes volumes, not prices)"
    category = "commodities"
    frequency = "irregular"
    csv_hint = ("Columns: date, rego_price (GBP/MWh). Ofgem's Renewable Electricity Register publishes REGO "
                "volumes but not prices; prices come from auction results (e.g. e-power auctions) and broker "
                "reports. Add one row per auction/report.")


__all__ = ["CarbonAllowances", "CertificatePrices", "GasPrices", "SourceNotImplemented"]
