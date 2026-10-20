"""Renewable certificate volumes from Ofgem's Renewable Electricity Register (RER).

Ofgem doesn't publish certificate *prices*, but its public reports show how many certificates are
issued each month: REGOs (Renewable Energy Guarantees of Origin) and ROCs (Renewables Obligation
Certificates), by technology. That's a monthly measure of how much renewable generation was
certified, which is useful context for prices.

Only the small summary reports ("Certificates by Technology and Month of Issue") are downloaded.
They're parsed in memory and only monthly totals are stored: no raw files are kept. The full
certificate-level reports are very large and not needed.
"""
from __future__ import annotations

import io
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

from storage import TS
from .base import DataSource
from .commodities import _describe_workbook, _norm, smart_dates

DASHBOARD = "https://rer.ofgem.gov.uk/Reports/Dashboard"
REPORT_TITLES = {  # scheme -> report title on the dashboard
    "rego": "Certificates by Technology and Month of Issue (REGO)",
    "ro": "Certificates by Technology and Month of Issue (RO)",
}
# Links as of Oct 2026, used if the dashboard can't be read. Override under sources.certificates.reports.
DEFAULT_LINKS = {
    "rego": "https://ofgemcloud.sharepoint.com/:x:/s/RERExternalReportsLive/"
            "IQAsaEB6IRhiTq6y9zIr6WAUAXp-Zehv7gdeKOrY81veG64?download=1",
    "ro": "https://ofgemcloud.sharepoint.com/:x:/s/RERExternalReportsLive/"
          "IQD5pFgdlzoFTIeqLbsgBn0GAVWcF_a6Rf_Slje2j2jQmkg?download=1",
}
TECH_GROUPS = (  # (group, keywords) — first match wins
    ("wind", ("wind",)),
    ("solar", ("solar", "photovoltaic", "pv")),
    ("hydro", ("hydro", "tidal", "wave")),
    ("bio", ("bio", "landfill", "sewage", "anaerobic", "waste", "wood", "straw", "fuelled", "gas")),
)
GROUPS = ("wind", "solar", "hydro", "bio", "other")


def tech_group(name) -> str:
    n = _norm(name)
    for group, keys in TECH_GROUPS:
        if any(k in n for k in keys):
            return group
    return "other"


class OfgemCertificates(DataSource):
    name = "certificates"
    description = "REGO and ROC certificates issued per month by technology (Ofgem Renewable Electricity Register)"
    category = "commodities"
    frequency = "1M"
    max_days_per_request = 100_000  # the reports always hold the full history: one download per run
    revision_lookback = timedelta(days=120)  # late issuance revises recent months

    def fetch(self, start: date, end: date) -> dict:
        links = self._links()
        out, errors = {}, []
        for scheme in ("rego", "ro"):
            try:
                out[scheme] = self.http_get_bytes(links[scheme])
            except Exception as exc:  # one report failing shouldn't lose the other
                errors.append(f"{scheme}: {exc}")
                self.log.warning("certificates: %s report download failed: %s", scheme.upper(), exc)
        if not out:
            raise RuntimeError("; ".join(errors))
        return out

    def _links(self) -> dict:
        """Report links from the dashboard (they can change), falling back to config/defaults."""
        links = {**DEFAULT_LINKS, **(self.config.get("reports") or {})}
        try:
            html = self.http_get_text(DASHBOARD)
            for scheme, title in REPORT_TITLES.items():
                found = find_report_link(html, title)
                if found:
                    links[scheme] = found
        except Exception as exc:
            self.log.info("certificates: couldn't read the RER dashboard (%s); using stored links", exc)
        return links

    def transform(self, raw: dict) -> pd.DataFrame:
        frames = []
        for scheme, content in raw.items():
            monthly = parse_month_report(content)  # month (Timestamp) x group -> certificates
            monthly = monthly.reindex(columns=list(GROUPS), fill_value=0.0)
            monthly.insert(0, "total", monthly.sum(axis=1))
            frames.append(monthly.add_prefix(f"{scheme}_issued_"))
        if not frames:
            return pd.DataFrame(columns=[TS])
        wide = pd.concat(frames, axis=1).sort_index()
        # Certificates issued in month M are only all known once M has ended: stamp each month on
        # the 1st of the next month (London time), so features never see a month early. The
        # current, incomplete month lands in the future and is dropped by validate().
        stamp = (wide.index + pd.offsets.MonthBegin(1)).tz_localize(self.settings.timezone).tz_convert("UTC")
        wide.index = stamp
        wide.index.name = TS
        return wide.reset_index()


def find_report_link(html: str, title: str) -> str | None:
    """The download link in the dashboard row whose text matches `title`."""
    t = re.escape(title).replace(r"\ ", r"\s+")
    m = re.search(t + r"(.{0,3000}?)href=\"([^\"]+)\"", html, re.S | re.I)
    if not m:
        return None
    return m.group(2).replace("&amp;", "&")


def _read_tables(content: bytes) -> dict[str, pd.DataFrame]:
    import warnings

    if content[:2] == b"PK":  # xlsx
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    text = content.decode("utf-8-sig", errors="replace")
    if text.lstrip().startswith("<"):
        raise ValueError("got a web page instead of a report (the download link may have changed)")
    return {"csv": pd.read_csv(io.StringIO(text), header=None, dtype=str, sep=None, engine="python")}


_MONTH_HDR = re.compile(r"month|issue|period|date")
_TECH_HDR = re.compile(r"technolog|tech group|generation type|fuel")
_COUNT_HDR = re.compile(r"certificate|count|number|no\.|volume|quantity|total|mwh")


def parse_month_report(content: bytes) -> pd.DataFrame:
    """'Certificates by Technology and Month of Issue' -> monthly totals per technology group.
    Handles a long table (month, technology, [country], count) or a wide one (technology rows,
    month columns), with notes above the header."""
    sheets = _read_tables(content)
    best = None
    for raw in sheets.values():
        for cand in (_long_table(raw), _wide_table(raw)):
            if cand is not None and (best is None or len(cand) > len(best)):
                best = cand
    if best is None or best.empty:
        raise RuntimeError("couldn't find month / technology / certificate columns in the report. Layout: "
                           + _describe_workbook(sheets))
    return best


def _cell(v) -> str:
    return "" if pd.isna(v) else _norm(v)


def _long_table(raw: pd.DataFrame) -> pd.DataFrame | None:
    for i in range(min(len(raw), 60)):
        hdr = [_cell(v) for v in raw.iloc[i].tolist()]
        tech = next((j for j, h in enumerate(hdr) if _TECH_HDR.search(h)), None)
        month = next((j for j, h in enumerate(hdr) if _MONTH_HDR.search(h) and j != tech), None)
        if tech is None or month is None:
            continue
        counts = [j for j, h in enumerate(hdr) if _COUNT_HDR.search(h) and j not in (tech, month)]
        body = raw.iloc[i + 1:]
        if not counts:  # fall back to the last mostly-numeric column
            numeric = [j for j in range(raw.shape[1]) if j not in (tech, month)
                       and pd.to_numeric(body.iloc[:, j], errors="coerce").notna().mean() > 0.8]
            counts = numeric[-1:]
        if not counts:
            continue
        months = _months(body.iloc[:, month])
        n = pd.to_numeric(body.iloc[:, counts[-1]].astype(str).str.replace(",", ""), errors="coerce")
        df = pd.DataFrame({"month": months.to_numpy(), "group": [tech_group(t) for t in body.iloc[:, tech]],
                           "n": n.to_numpy()}).dropna()
        if len(df) >= 3:
            return df.pivot_table(index="month", columns="group", values="n", aggfunc="sum").sort_index()
    return None


def _wide_table(raw: pd.DataFrame) -> pd.DataFrame | None:
    for i in range(min(len(raw), 60)):
        row = raw.iloc[i].tolist()
        months = _months(pd.Series(row))
        cols = [j for j, m in enumerate(months) if pd.notna(m)]
        if len(cols) < 3:
            continue
        hdr = [_cell(v) for v in row]
        tech = next((j for j, h in enumerate(hdr) if _TECH_HDR.search(h)), 0)
        body = raw.iloc[i + 1:]
        recs = []
        for _, r in body.iterrows():
            name = r.iloc[tech]
            if pd.isna(name) or "total" in _cell(name):
                continue
            for j in cols:
                v = pd.to_numeric(str(r.iloc[j]).replace(",", ""), errors="coerce")
                if pd.notna(v):
                    recs.append((months[j], tech_group(name), float(v)))
        if len(recs) >= 3:
            df = pd.DataFrame(recs, columns=["month", "group", "n"])
            return df.pivot_table(index="month", columns="group", values="n", aggfunc="sum").sort_index()
    return None


def _months(values: pd.Series) -> pd.Series:
    """Month labels ('Jan-2024', '2024-01', 'January 2024', dates) -> first-of-month timestamps."""
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NaT
        if isinstance(v, (int, float, np.integer, np.floating)):  # counts, not months
            return pd.NaT
        txt = str(v).strip()
        if not re.search(r"\d{4}|\d{1,2}[/-]\d{2}", txt) or len(txt) > 40:
            return pd.NaT
        try:
            t = smart_dates([v]).iloc[0]
            if pd.isna(t):
                t = pd.to_datetime(txt, format="mixed", errors="coerce")
        except (ValueError, TypeError):
            return pd.NaT
        if pd.isna(t) or not 2000 <= t.year <= 2100:
            return pd.NaT
        return pd.Timestamp(t.year, t.month, 1)
    return pd.Series([one(v) for v in values], index=values.index)
