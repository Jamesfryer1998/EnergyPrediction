"""Base-class contract, incremental updates and parsing of each live source."""
from datetime import date, timedelta

import pandas as pd
import pytest
from helpers import make_env

from sources import REGISTRY, DataSource, build_source, list_sources
from sources.base import SourceNotImplemented
from storage import TS


class FakeSource(DataSource):
    name = "_fake_test_source"
    max_days_per_request = 3
    calls: list = []

    def fetch(self, start, end):
        FakeSource.calls.append((start, end))
        idx = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
                            freq="30min", inclusive="left")
        return [{"t": t.isoformat(), "v": i} for i, t in enumerate(idx)]

    def transform(self, raw):
        df = pd.DataFrame(raw)
        return pd.DataFrame({TS: pd.to_datetime(df["t"], utc=True), "v": df["v"]})


def test_registry_has_every_stream():
    names = set(REGISTRY)
    for expected in ["prices_mid", "prices_system", "demand", "grid_generation", "grid_carbon_intensity",
                     "weather", "calendar_events", "weather_events",
                     "gas_prices", "carbon_allowances", "certificates", "certificate_prices"]:
        assert expected in names
    assert all("mode" in s for s in list_sources())


def test_subclass_must_define_name():
    with pytest.raises(TypeError):
        class Nameless(DataSource):
            def fetch(self, start, end):
                return None

            def transform(self, raw):
                return raw


def test_windows_are_chunked_and_cover_range(tmp_path):
    settings, store = make_env(tmp_path)
    src = FakeSource(store=store, settings=settings)
    w = list(src.windows(date(2026, 1, 1), date(2026, 1, 8)))
    assert w[0] == (date(2026, 1, 1), date(2026, 1, 3))
    assert w[-1][1] == date(2026, 1, 8)
    assert all(b - a <= timedelta(days=2) for a, b in w)


def test_update_is_incremental_and_idempotent(tmp_path):
    settings, store = make_env(tmp_path)
    src = FakeSource(store=store, settings=settings)
    FakeSource.calls = []
    r1 = src.update("2026-01-01", "2026-01-06")
    assert r1.status == "ok" and r1.windows == 2
    n1 = len(src.load())
    src.update("2026-01-05", "2026-01-06")  # overlap -> upsert, no duplicates
    assert len(src.load()) == n1 == 6 * 48
    # resume starts at last timestamp minus the revision lookback
    assert src.resume_start() == date(2026, 1, 4)


def test_validate_rejects_missing_timestamp(tmp_path):
    settings, store = make_env(tmp_path)
    src = FakeSource(store=store, settings=settings)
    with pytest.raises(ValueError):
        src.validate(pd.DataFrame({"v": [1]}))


def test_manual_csv_source_reports_missing_file(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("certificate_prices", store=store, settings=settings)
    with pytest.raises(SourceNotImplemented):
        src.fetch(date(2026, 1, 1), date(2026, 1, 2))
    assert src.update("2026-01-01", "2026-01-02").status == "not_implemented"


def test_manual_csv_source_reads_file(tmp_path):
    settings, store = make_env(tmp_path)
    (settings.data_dir / "manual").mkdir(parents=True)
    (settings.data_dir / "manual" / "certificate_prices.csv").write_text("date,rego_price\n2026-01-01,8.5\n2026-01-02,8.7\n")
    src = build_source("certificate_prices", store=store, settings=settings)
    assert src.update("2026-01-01", "2026-01-05").rows == 2


# ── parsing of real API payload shapes ──────────────────────────────────
def test_elexon_mid_transform(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("prices_mid", store=store, settings=settings)
    raw = [
        {"startTime": "2026-01-01T00:00:00Z", "dataProvider": "APXMIDP", "settlementDate": "2026-01-01",
         "settlementPeriod": 1, "price": 75.5, "volume": 1200.0},
        {"startTime": "2026-01-01T00:00:00Z", "dataProvider": "N2EXMIDP", "settlementDate": "2026-01-01",
         "settlementPeriod": 1, "price": 0.0, "volume": 0.0},
    ]
    df = src.transform(raw)
    assert len(df) == 1 and df["price_mid"].iloc[0] == 75.5


def test_elexon_fuelhh_pivot(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("grid_generation", store=store, settings=settings)
    raw = [{"startTime": "2026-01-01T00:00:00Z", "fuelType": f, "generation": g}
           for f, g in [("WIND", 8000), ("CCGT", 10000), ("NUCLEAR", 4000), ("INTFR", 1500)]]
    df = src.transform(raw)
    row = df.iloc[0]
    assert row["gen_wind"] == 8000 and row["gen_total"] == 22000 and row["interconnector_net"] == 1500


def test_open_meteo_transform(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("weather", store=store, settings=settings)
    payload = {"hourly": {"time": ["2026-01-01T00:00", "2026-01-01T01:00"], "temperature_2m": [4.0, 3.5]}}
    df = src.transform({"london": [payload], "leeds": [payload]})
    assert "london__temperature_2m" in df and df["wx_mean__temperature_2m"].tolist() == [4.0, 3.5]


def test_bank_holidays_transform(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("calendar_events", store=store, settings=settings)
    raw = {"england-and-wales": {"events": [{"title": "Christmas Day", "date": "2026-12-25"}]}}
    df = src.transform(raw)
    assert df[TS].iloc[0] == pd.Timestamp("2026-12-25T00:00Z")  # UK winter = UTC


def test_carbon_intensity_transform(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("grid_carbon_intensity", store=store, settings=settings)
    df = src.transform([{"from": "2026-01-01T00:00Z", "to": "2026-01-01T00:30Z",
                         "intensity": {"forecast": 150, "actual": 160, "index": "moderate"}}])
    assert df["ci_actual"].iloc[0] == 160


# ── gas / carbon / storms (fixtures mimic the documented formats of each site) ───────────
import io  # noqa: E402

from sources.events import parse_impact_dates  # noqa: E402


def _xlsx(rows: list[list]) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, header=False)
    return buf.getvalue()


def test_storm_impact_date_ranges():
    assert parse_impact_dates("20-21 October 2024") == [date(2024, 10, 20), date(2024, 10, 21)]
    assert parse_impact_dates("27 November 2024") == [date(2024, 11, 27)]
    assert parse_impact_dates("30 November - 1 December 2024") == [date(2024, 11, 30), date(2024, 12, 1)]
    assert parse_impact_dates("31 December 2024 – 1 January 2025") == [date(2024, 12, 31), date(2025, 1, 1)]
    assert parse_impact_dates("3-4 Oct 2025") == [date(2025, 10, 3), date(2025, 10, 4)]
    assert parse_impact_dates("") == []


def test_weather_events_parse_met_office_table(tmp_path):
    settings, store = make_env(tmp_path)
    html = """<html><body><h2>Named storms</h2><table>
      <tr><th>Name</th><th>Date named</th><th>Date of impact on UK and/or Ireland and/or Netherlands</th></tr>
      <tr><td><a href="x.pdf">Ashley</a></td><td>18 October 2024</td><td>20-21 October 2024</td></tr>
      <tr><td><a href="y.pdf">Bert</a></td><td>21 November 2024</td><td>22-25 November 2024</td></tr>
      <tr><td>Conall</td><td>26 November 2024</td><td>27 November 2024</td></tr>
    </table></body></html>"""
    src = build_source("weather_events", store=store, settings=settings)
    df = src.validate(src.transform({2024: html}))
    assert len(df) == 2 + 4 + 1 and set(df["storm_name"]) == {"Ashley", "Bert", "Conall"}
    assert df[df.storm_name == "Ashley"]["timestamp"].min() == pd.Timestamp("2024-10-19T23:00Z")  # 20 Oct, BST
    store.write("weather_events", "GB", df, src.key_columns)
    assert len(store.read("weather_events", "GB")) == 7


def test_weather_events_tries_both_url_styles(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("weather_events", store=store, settings=settings)
    seen = []

    def get(url, params=None):
        seen.append(url)
        if url.endswith("202526"):
            return "<table><tr><th>Name</th><th>Date of impact</th></tr><tr><td>Amy</td><td>3-4 October 2025</td></tr></table>"
        raise RuntimeError("404")

    src.http_get_text = get
    pages = src.fetch(date(2025, 9, 1), date(2026, 3, 1))
    assert list(pages) == [2025] and any(u.endswith("2025-26") for u in seen)
    assert list(src.transform(pages)["storm_name"]) == ["Amy", "Amy"]


def test_gas_national_gas_transform(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("gas_prices", store=store, settings=settings)
    rows = [{"applicableFor": "2026-10-01", "value": "2.95", "applicableAt": "2026-10-02T08:00:00+00:00"},
            {"applicableFor": "2026-10-01", "value": "3.00", "applicableAt": "2026-10-02T09:00:00+00:00"},  # revision
            {"applicableFor": "2026-10-02", "value": "3.10", "applicableAt": "2026-10-03T08:00:00+00:00"}]
    df = src.transform({"kind": "national_gas", "name": "SAP", "rows": rows})
    assert list(df["gas_sap_pkwh"]) == [3.00, 3.10]
    assert abs(df["gas_sap_ptherm"].iloc[0] - 3.00 * 29.3071) < 1e-9


def test_gas_ons_workbook_and_fallback(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("gas_prices", store=store, settings=settings)
    book = _xlsx([["System Average Price (SAP) of gas", None, None],
                  ["Source: National Gas Transmission", None, None],
                  [None, None, None],
                  ["Date", "SAP (p/kWh)", "7-day average SAP (p/kWh)"],
                  ["2026-09-29", 2.8, 2.7], ["2026-09-30", 2.9, 2.75], ["2026-10-01", 3.0, 2.8]])
    page = ('<a href="/file?uri=/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas/'
            '2026/systemaveragepriceofgasdataset081026.xlsx">xlsx</a>')
    # National Gas catalogue has no SAP entry -> falls back to ONS
    src.http_get_json = lambda url, params=None: [{"name": "Allocations, Energy", "publicationId": "PUBOBJ1"}]
    src.http_get_text = lambda url, params=None: page
    src.http_get_bytes = lambda url, params=None: book
    raw = src.fetch(date(2026, 9, 1), date(2026, 10, 2))
    assert raw["kind"] == "ons" and raw["notes"]
    df = src.validate(src.transform(raw))
    assert list(df["gas_sap_pkwh"]) == [2.8, 2.9, 3.0]


def _xlsx_sheets(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(xw, sheet_name=name, index=False, header=False)
    return buf.getvalue()


def test_gas_ons_workbook_with_cover_notes_and_wide_tables():
    from datetime import datetime

    from sources.commodities import GasPrices

    # ONS style: cover/contents/notes sheets, and notes above the table that mention dates
    book = _xlsx_sheets({
        "Cover_sheet": [["System Average Price (SAP) of gas"], ["Release date:", "9 October 2026"],
                        ["Next release date:", "16 October 2026"]],
        "Contents": [["Worksheet name", "Worksheet title"], ["Table_1", "SAP of gas, daily, updated weekly"]],
        "Table_1": [["Table 1: System Average Price (SAP) of gas, pence per kilowatt hour"],
                    ["This worksheet contains one table. Data last updated 9 October 2026"],
                    ["Date", "SAP actual day (pence per kWh)", "SAP seven-day rolling average (pence per kWh)"],
                    [datetime(2026, 10, 5), 2.81, 2.7], [datetime(2026, 10, 6), 2.92, 2.75],
                    [datetime(2026, 10, 7), 3.05, 2.8], ["[x]", None, None]],
    })
    df = GasPrices.parse_ons_workbook(book)
    assert list(df["sap"]) == [2.81, 2.92, 3.05] and df.attrs["unit"] == "p/kWh"
    assert df["date"].iloc[0] == pd.Timestamp("2026-10-05")

    # the real ONS layout (Oct 2026): two-row headers over side-by-side series
    real = _xlsx_sheets({
        "Notes": [["System Average Price (SAP) of gas", None], ["Source: National Gas", None]],
        "System Average Price": [[None, "SAP, Actual Day", "SAP, preceding 7 Day rolling average"],
                                 ["Applicable For", "Value (p/kWh)", "Value (p/kWh)"],
                                 [datetime(2018, 1, 1), 1.7083, 1.8474], [datetime(2018, 1, 2), 1.75, 1.84],
                                 [datetime(2018, 1, 3), 1.8, 1.83]]})
    df = GasPrices.parse_ons_workbook(real)
    assert list(df["sap"]) == [1.7083, 1.75, 1.8] and df.attrs["unit"] == "p/kWh"

    # dates across the columns, prices in p/therm
    wide = _xlsx_sheets({"Data": [["Gas day", "01/10/2026", "02/10/2026", "03/10/2026"],
                                  ["System Average Price (p/therm)", 80.1, 82.0, 85.5],
                                  ["7-day average (p/therm)", 79, 80, 81]]})
    df = GasPrices.parse_ons_workbook(wide)
    assert list(df["sap"]) == [80.1, 82.0, 85.5] and df.attrs["unit"] == "p/therm"
    assert df["date"].iloc[1] == pd.Timestamp("2026-10-02")

    # unrecognised layout: the error describes the workbook so the parser can be fixed
    try:
        GasPrices.parse_ons_workbook(_xlsx_sheets({"Odd": [["Day", "Value"], ["a", "b"]]}))
        raise AssertionError("expected an error")
    except RuntimeError as exc:
        assert "Layout: [Odd" in str(exc) and "Day | Value" in str(exc)


def test_gas_finds_sap_in_nested_catalogue(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("gas_prices", store=store, settings=settings)
    src.http_get_json = lambda url, params=None: {"categories": [{"subCategories": [
        {"name": "System Average Price, 7 day rolling", "publicationId": "P7"},
        {"name": "System Average Price, Actual Day", "publicationId": "PSAP"}]}]}
    assert src._find_sap_publication() == ("PSAP", "System Average Price, Actual Day")


def test_carbon_eex_ukets_and_fx(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("carbon_allowances", store=store, settings=settings)
    eex = _xlsx([["EEX EUA auction report"], [None], [None], [None], [None],
                 ["Date", "Auction Name", "Status", "Auction Price €/tCO2", "Minimum Bid €/tCO2", "Auction Volume tCO2"],
                 ["2026-01-05", "EU Auction EUA", "Successful", 70.0, 60.0, 3_000_000],
                 ["2026-01-05", "DE Auction EUA", "Successful", 72.0, 61.0, 1_000_000],
                 ["2026-01-05", "EU Auction EUAA", "Successful", 99.0, 90.0, 500_000],   # aviation: excluded
                 ["2026-01-06", "EU Auction EUA", "Successful", 71.0, 60.0, 3_000_000]])
    ccm = """<table>
      <tr><th>Month</th><th>2-year reference period</th><th>Trigger price</th><th>6-month monitoring period</th>
          <th>Month 1 average price</th><th>Month 2 average price</th><th>Month 3 average price</th>
          <th>Month 4 average price</th><th>Month 5 average price</th><th>Month 6 average price</th>
          <th>CCM triggered? Yes or No</th></tr>
      <tr><td>Mar-26</td><td>x</td><td>£140.00</td><td>Sep, Oct, Nov, Dec, Jan, Feb</td>
          <td>£50.00</td><td>£51.00</td><td>£52.00</td><td>£53.00</td><td>£54.00</td><td>TBD</td><td>No</td></tr>
    </table>"""
    fx = ("KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
          "EXR.D.GBP.EUR.SP00.A,D,GBP,EUR,SP00,A,2026-01-02,0.85\n"
          "EXR.D.GBP.EUR.SP00.A,D,GBP,EUR,SP00,A,2026-01-06,0.86\n")
    raw = {"eex": {2026: eex}, "ukets_html": ccm, "fx_csv": fx, "errors": {}}
    df = src.validate(src.transform(raw))
    eua = df.dropna(subset=["eua_auction_eur"]).set_index("timestamp")
    assert abs(eua["eua_auction_eur"].iloc[0] - 70.5) < 1e-9          # volume-weighted 70 & 72; EUAA dropped
    assert abs(eua["eua_auction_gbp"].iloc[0] - 70.5 * 0.85) < 1e-9   # 5 Jan uses the 2 Jan rate
    uka = df.dropna(subset=["uka_avg_gbp"]).set_index("uka_month")
    assert uka.loc["2026-01", "uka_avg_gbp"] == 54.0 and "2026-02" not in uka.index  # TBD skipped
    jan = uka.loc["2026-01"]
    assert jan["timestamp"] == pd.Timestamp("2026-02-15T00:00Z")      # usable from 15 Feb (after month end)
    assert abs(jan["uka_minus_eua_gbp"] - (54.0 - jan["eua_month_avg_gbp"])) < 1e-9


def test_carbon_survives_one_site_failing(tmp_path):
    settings, store = make_env(tmp_path)
    src = build_source("carbon_allowances", store=store, settings=settings)

    def boom(*a, **k):
        raise RuntimeError("down")

    def text(url, params=None):
        if "gov.uk" in url:
            return ("<table><tr><th>Month</th><th>3-month monitoring period</th><th>Month 1 average price</th></tr>"
                    "<tr><td>Feb-26</td><td>Jan</td><td>£40.00</td></tr></table>")
        raise RuntimeError("no fx")

    src.http_get_bytes = boom
    src.http_get_text = text
    raw = src.fetch(date(2026, 1, 1), date(2026, 2, 20))
    assert "eua 2026" in raw["errors"] and "fx" in raw["errors"]
    assert list(src.transform(raw)["uka_avg_gbp"]) == [40.0]


# ── Ofgem certificates (REGO / ROC volumes) ─────────────────────────────────

def test_certificate_reports_long_wide_and_csv(tmp_path):
    from datetime import datetime

    from sources.certificates import find_report_link, parse_month_report, tech_group

    assert tech_group("Wind - Offshore") == "wind" and tech_group("Photovoltaic") == "solar"
    assert tech_group("Landfill Gas") == "bio" and tech_group("Hydro 20MW DNC or less") == "hydro"
    assert tech_group("Geothermal") == "other"
    # long layout with a title row, a country column and thousands separators
    long_book = _xlsx_sheets({"Report": [
        ["Certificates by Technology and Month of Issue (REGO)", None, None, None],
        ["Technology Group", "Country", "Month of Issue", "Certificates Issued"],
        ["Wind", "England", "Jan-2026", "1,000"], ["Wind", "Scotland", "Jan-2026", "500"],
        ["Photovoltaic", "England", "Jan-2026", 200], ["Wind", "England", "Feb-2026", 900],
        ["Biomass", "Wales", datetime(2026, 2, 1), 300]]})
    m = parse_month_report(long_book)
    assert m.loc[pd.Timestamp("2026-01-01"), "wind"] == 1500 and m.loc[pd.Timestamp("2026-01-01"), "solar"] == 200
    assert m.loc[pd.Timestamp("2026-02-01"), "bio"] == 300
    # wide layout: technologies down, months across, with a total row to ignore
    wide = _xlsx_sheets({"Sheet1": [
        ["Technology", datetime(2026, 1, 1), datetime(2026, 2, 1), datetime(2026, 3, 1)],
        ["Wind", 10, 20, 30], ["Hydro", 1, 2, 3], ["Total", 11, 22, 33]]})
    w = parse_month_report(wide)
    assert list(w["wind"]) == [10, 20, 30] and list(w["hydro"]) == [1, 2, 3]
    # CSV export
    csv = b"Month of Issue,Technology,Number of Certificates\n2026-01,Wind,5\n2026-01,Solar,6\n2026-02,Wind,7\n"
    c = parse_month_report(csv)
    assert c.loc[pd.Timestamp("2026-02-01"), "wind"] == 7
    # unrecognised layout -> error that describes it
    with pytest.raises(RuntimeError, match="Layout"):
        parse_month_report(_xlsx_sheets({"X": [["a", "b"], ["c", "d"]]}))
    html = ('<tr><td>Certificates by Technology and Month of Issue (REGO)</td><td><span>Updated today</span></td>'
            '<td><a href="https://ofgemcloud.sharepoint.com/x?download=1&amp;y=2">Download</a></td></tr>')
    assert find_report_link(html, "Certificates by Technology and Month of Issue (REGO)") == \
        "https://ofgemcloud.sharepoint.com/x?download=1&y=2"


def test_certificates_source_stamps_months_after_they_end(tmp_path):
    from datetime import datetime

    settings, store = make_env(tmp_path)
    src = build_source("certificates", store=store, settings=settings)
    book = _xlsx_sheets({"R": [["Technology", "Month of Issue", "Certificates"],
                               ["Wind", "2026-08", 100], ["Solar", "2026-08", 50], ["Wind", "2026-09", 120]]})
    src.http_get_text = lambda url, params=None: "<html>no links</html>"
    src.http_get_bytes = lambda url, params=None: book
    df = src.validate(src.transform(src.fetch(date(2026, 1, 1), date(2026, 10, 1))))
    first = df.iloc[0]
    # August's certificates become known on 1 September (London midnight = 23:00 UTC on 31 Aug)
    assert first[TS] == pd.Timestamp("2026-08-31T23:00", tz="UTC")
    assert first["rego_issued_total"] == 150 and first["rego_issued_wind"] == 100
    assert first["ro_issued_total"] == 150  # same fake file served for both schemes
    assert pd.Timestamp(datetime(2026, 9, 30, 23), tz="UTC") in set(df[TS])
