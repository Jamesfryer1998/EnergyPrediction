"""Paper trading: IG demo client, position accounting, strategy backtest, live cycle, web API."""
import json

import numpy as np
import pandas as pd

from helpers import make_env
from sources.synthetic import synthetic_gas_candles
from storage import TS


# ── IG client (fake HTTP) ────────────────────────────────────────────────────

class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.content = json.dumps(self._body).encode()
        self.text = self.content.decode()

    def json(self):
        return self._body


class FakeHTTP:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json, headers))
        assert url.startswith("https://demo-api.ig.com/")
        return FakeResp(200, {"currentAccountId": "DEMO1", "currencyIsoCode": "GBP",
                              "accounts": [{"accountId": "DEMO1", "accountName": "CFD", "accountType": "CFD"}]},
                        {"CST": "cst-token", "X-SECURITY-TOKEN": "xst-token"})

    def request(self, method, url, json=None, params=None, headers=None, timeout=None):
        self.calls.append((method, url, json, headers))
        assert url.startswith("https://demo-api.ig.com/gateway/deal")
        assert headers["CST"] == "cst-token" and headers["X-SECURITY-TOKEN"] == "xst-token"
        if "/prices/" in url:
            if params.get("from", "").startswith("2020"):
                return FakeResp(403, {"errorCode": "error.public-api.exceeded-account-historical-data-allowance"})
            return FakeResp(200, {"prices": [
                {"snapshotTimeUTC": "2026-10-08T09:00:00", "openPrice": {"bid": 90.0, "ask": 90.5},
                 "closePrice": {"bid": 91.0, "ask": 91.5}, "highPrice": {"bid": 92, "ask": 92.5},
                 "lowPrice": {"bid": 89, "ask": 89.5}, "lastTradedVolume": 120}],
                "metadata": {"allowance": {"remainingAllowance": 9990, "totalAllowance": 10000}}})
        if url.endswith("/positions/otc"):
            return FakeResp(200, {"dealReference": "REF1"})
        if "/confirms/" in url:
            return FakeResp(200, {"dealStatus": "ACCEPTED", "dealId": "D1", "level": 91.5, "size": json and 1 or 1,
                                  "direction": "BUY"})
        if url.endswith("/positions"):
            return FakeResp(200, {"positions": [{"position": {"dealId": "D1", "direction": "SELL", "size": 2, "level": 90},
                                                 "market": {"epic": "CC.D.GAS.IP", "bid": 90, "offer": 90.4}}]})
        if "/markets/" in url:
            return FakeResp(200, {"instrument": {"name": "UK Natural Gas", "expiry": "-", "valueOfOnePip": "10",
                                                 "currencies": [{"code": "GBP", "isDefault": True}]},
                                  "dealingRules": {"minDealSize": {"value": 0.5}},
                                  "snapshot": {"marketStatus": "TRADEABLE", "bid": 90.1, "offer": 90.6}})
        if url.endswith("/markets"):
            return FakeResp(200, {"markets": [
                {"epic": "CC.D.NG.UNC.IP", "instrumentName": "US Natural Gas", "expiry": "-", "instrumentType": "COMMODITIES"},
                {"epic": "CC.D.GAS.IP", "instrumentName": "UK Natural Gas (pence/therm)", "expiry": "-",
                 "instrumentType": "COMMODITIES"},
                {"epic": "CC.D.GAS.NOV", "instrumentName": "UK Natural Gas (pence/therm)", "expiry": "NOV-26",
                 "instrumentType": "COMMODITIES"}]})
        return FakeResp(404, {"errorCode": "not-found"})


def test_ig_client_demo_only_session_prices_and_orders(monkeypatch):
    from datetime import datetime

    import trading.ig as igmod
    from trading.ig import IGAllowanceExceeded, IGClient

    monkeypatch.setattr(igmod, "MIN_INTERVAL", 0)

    http = FakeHTTP()
    c = IGClient("key", "user", "pw", session=http)
    assert c.find_market("UK Natural Gas")["epic"] == "CC.D.GAS.IP"  # logs in first, picks undated UK market
    assert http.calls[0][0] == "POST" and http.calls[0][3]["X-IG-API-KEY"] == "key"
    df = c.prices("CC.D.GAS.IP", datetime(2026, 10, 8), datetime(2026, 10, 9))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-10-08T09:00", tz="UTC")
    assert df["mid_close"].iloc[0] == 91.25 and df["spread"].iloc[0] == 0.5
    assert c.last_allowance["remainingAllowance"] == 9990
    try:
        c.prices("CC.D.GAS.IP", datetime(2020, 1, 1), datetime(2020, 1, 2))
        raise AssertionError("expected allowance error")
    except IGAllowanceExceeded:
        pass
    conf = c.place_market_order("CC.D.GAS.IP", "BUY", 1)
    assert conf["status"] == "ACCEPTED" and conf["deal_id"] == "D1"
    order = next(x for x in http.calls if x[1].endswith("/positions/otc"))
    assert order[2]["forceOpen"] is False and order[2]["orderType"] == "MARKET"
    assert c.net_position("CC.D.GAS.IP") == -2
    assert c.instrument("CC.D.GAS.IP")["value_per_point"] == 10.0


class BadLoginHTTP(FakeHTTP):
    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json, headers))
        return FakeResp(400, {"errorCode": "validation.pattern.invalid.authenticationRequest.identifier"})


def test_bad_ig_login_stops_after_one_attempt(tmp_path, monkeypatch):
    import trading.ig as igmod
    from sources import build_source
    from trading.ig import IGClient, IGLoginError

    monkeypatch.setattr(igmod, "MIN_INTERVAL", 0)
    # an email as the username is rejected before anything is sent
    http = BadLoginHTTP()
    try:
        IGClient("key", "me@example.com", "pw", session=http).login()
        raise AssertionError("expected IGLoginError")
    except IGLoginError as exc:
        assert "not your email" in str(exc)
    assert http.calls == []
    # a rejected login stops the whole backfill after ONE attempt (not one per window)
    settings, store = make_env(tmp_path)
    src = build_source("ig_gas", store=store, settings=settings)
    http = BadLoginHTTP()
    src._client = IGClient("key", "baduser-test", "pw", session=http)
    res = src.update()
    assert res.status == "error" and "username" in res.message
    assert len(http.calls) == 1
    # and later runs don't retry until the back-off has passed
    try:
        IGClient("key", "baduser-test", "pw", session=http).login()
        raise AssertionError("expected IGLoginError")
    except IGLoginError as exc:
        assert "not retrying" in str(exc)
    assert len(http.calls) == 1
    igmod._login_blocked.clear()


# ── accounting ───────────────────────────────────────────────────────────────

def test_book_fills_at_bid_ask_flips_and_funding():
    from trading.strategy import Book, StrategyConfig, decide, nights_between

    cfg = StrategyConfig(value_per_point=10, starting_balance=1000, funding_rate=0.0365)
    b = Book(cfg)
    b.trade("2026-10-05T09:00Z", 1, bid=90, ask=91)           # buy 1 at the ask
    assert b.position == 1 and b.avg_price == 91 and abs(b.equity(90, 91) - 990) < 1e-9  # marked at the bid
    b.trade("2026-10-05T10:00Z", -2, bid=95, ask=96)          # sell 2: close the long at 95, open a short at 95
    assert b.position == -1 and b.avg_price == 95 and b.realised == 40
    b.trade("2026-10-05T11:00Z", 1, bid=93, ask=94)           # cover at the ask
    assert b.position == 0 and b.realised == 50 and len(b.closed) == 2
    assert abs(b.spread_cost - (0.5 + 2 * 0.5 + 0.5) * 10) < 1e-9  # half a spread per contract filled
    b.trade("2026-10-05T12:00Z", 1, bid=100, ask=100)
    assert abs(b.charge_funding(2, 100) - 1 * 100 * 10 * 0.0365 / 365 * 2) < 1e-12
    # Friday afternoon to Monday morning crosses three 22:00 funding cut-offs
    assert nights_between(pd.Timestamp("2026-10-09T15:00Z"), pd.Timestamp("2026-10-12T08:00Z"), "Europe/London") == 3
    assert nights_between(pd.Timestamp("2026-10-09T09:00Z"), pd.Timestamp("2026-10-09T15:00Z"), "Europe/London") == 0
    assert decide(2.0, 0, 1.0, 1) == 1 and decide(-2.0, 1, 1.0, 1) == -1
    assert decide(0.5, 1, 1.0, 1) == 1 and decide(-0.5, 1, 1.0, 1) == 0 and decide(0.5, 0, 1.0, 1) == 0


def test_features_do_not_look_ahead():
    from trading.strategy import TARGET, StrategyConfig, build_features

    cfg = StrategyConfig(horizon=6)
    c = synthetic_gas_candles(days=60).set_index(TS)
    full = build_features(c, None, cfg)
    cut = build_features(c.iloc[:400], None, cfg)
    cols = [x for x in full.columns if x != TARGET]
    pd.testing.assert_frame_equal(full.iloc[:400][cols], cut[cols])  # past rows don't depend on the future
    assert cut[TARGET].iloc[-6:].isna().all() and cut[TARGET].iloc[:-6].notna().all()


# ── backtest, live cycle, web ────────────────────────────────────────────────

def _env_with_candles(tmp_path, days=200):
    settings, store = make_env(tmp_path, trading={"backtest_days": 20, "train_days": 120, "retrain_every_days": 10,
                                                  "horizon_hours": 6, "model": "hgb", "threshold_k": 1.0})
    store.write("ig_gas", "GB", synthetic_gas_candles(days=days))
    return settings, store


def test_strategy_backtest_and_live_model(tmp_path):
    from trading import strategy

    settings, store = _env_with_candles(tmp_path)
    res = strategy.backtest(settings, store)
    sm = res["summary"]
    for k in ("pnl", "trades", "spread_cost", "funding", "buy_hold_pnl", "hit_rate", "max_drawdown"):
        assert k in sm
    assert sm["trades"] > 0 and sm["spread_cost"] > 0
    saved = strategy.load_backtest(settings, res["run_id"])
    assert saved["series"] and saved["summary"]["pnl"] == sm["pnl"]
    assert strategy.list_backtests(settings)[0]["run_id"] == res["run_id"]
    strategy.train(settings, store)
    sig = strategy.predict_latest(settings, store)
    assert np.isfinite(sig["pred"]) and sig["threshold"] > 0 and sig["bid"] < sig["ask"]


class FakeIG:
    """Stands in for IGClient inside run_cycle."""

    def __init__(self, tradeable=True):
        self.orders, self.tradeable, self.pos = [], tradeable, 0.0

    def find_market(self, term):
        return {"epic": "CC.D.GAS.IP"}

    def instrument(self, epic):
        return {"epic": epic, "name": "UK Natural Gas", "expiry": "-", "currency": "GBP", "value_per_point": 10.0}

    def prices(self, epic, start=None, end=None, resolution="HOUR", max_points=None):
        return pd.DataFrame(columns=["timestamp"])

    def quote(self, epic):
        return {"bid": 100.0, "offer": 100.5, "status": "TRADEABLE" if self.tradeable else "CLOSED"}

    def place_market_order(self, epic, direction, size, expiry="-", currency="GBP"):
        self.orders.append((direction, size))
        self.pos += size if direction == "BUY" else -size
        return {"status": "ACCEPTED", "deal_id": f"D{len(self.orders)}", "level": 100.5 if direction == "BUY" else 100.0}

    def net_position(self, epic):
        return self.pos


def test_paper_cycle_trades_and_mirrors_to_ig_demo(tmp_path):
    from trading import paper, strategy

    settings, store = _env_with_candles(tmp_path, days=150)
    strategy.train(settings, store)
    paper.update_state(settings, {"mirror_ig": True, "threshold_k": 0.0, "size": 2})
    ig = FakeIG()
    res = paper.run_cycle(settings, client=ig)
    assert res["position"] in (2.0, -2.0)                      # threshold 0: always takes a side
    assert ig.orders == [("BUY" if res["position"] > 0 else "SELL", 2.0)]
    fills = paper.Ledger(settings).rows("fills")
    assert fills[0]["ig_status"] == "ACCEPTED" and fills[0]["price"] == (100.5 if res["position"] > 0 else 100.0)
    st = paper.status(settings)
    assert st["position"] == res["position"] and st["config"]["value_per_point"] == 10.0
    closed = paper.run_cycle(settings, client=ig, force_target=0.0, reason="test close")
    assert closed["position"] == 0 and ig.pos == 0
    assert paper.Ledger(settings).rows("fills")[-1]["reason"] == "test close"
    # market closed: no trade
    paper.run_cycle(settings, client=FakeIG(tradeable=False))
    assert paper.Ledger(settings).rows("decisions")[-1]["action"] == "market closed"
    assert len(paper.Ledger(settings).rows("fills")) == 2


def test_trading_web_api_and_settings(tmp_path, monkeypatch):
    import web

    settings, store = _env_with_candles(tmp_path, days=150)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    client = web.create_app().test_client()
    st = client.get("/api/trading/status").get_json()
    assert st["position"] == 0 and st["state"]["enabled"] is False
    assert client.post("/api/trading/settings", json={"interval_minutes": 0}).status_code == 400
    assert client.post("/api/trading/settings", json={"interval_minutes": 1441}).status_code == 400
    assert client.post("/api/trading/settings", json={"interval_minutes": 1}).get_json()["interval_minutes"] == 1
    r = client.post("/api/trading/settings", json={"enabled": True, "interval_minutes": 30, "size": 3}).get_json()
    assert r["enabled"] is True and r["interval_minutes"] == 30 and r["size"] == 3
    assert client.get("/api/trading/chart?days=400").get_json()["candles"]
    assert client.get("/api/trading/backtests").get_json() == []
    assert client.post("/api/trading/reset").get_json()["reset"] is True


def test_strategy_step_is_skipped_without_ig_data(tmp_path):
    from pipeline.nightly import run_strategy

    settings, _ = make_env(tmp_path)
    assert run_strategy(settings)["status"] == "skipped"


def test_trading_runner_schedule():
    from datetime import datetime, timezone

    from web.trading_runner import TradingRunner

    tr = TradingRunner(lambda: None)
    wed = datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)
    assert not tr.due({"enabled": False}, wed)
    assert tr.due({"enabled": True, "last_cycle": None}, wed)
    assert not tr.due({"enabled": True, "last_cycle": "2026-10-07T09:30:00+00:00", "interval_minutes": 60}, wed)
    assert tr.due({"enabled": True, "last_cycle": "2026-10-07T08:59:00+00:00", "interval_minutes": 60}, wed)
    assert not tr.due({"enabled": True, "last_cycle": None}, datetime(2026, 10, 10, 10, tzinfo=timezone.utc))  # Sat


def test_trading_runner_one_minute_interval_and_failure_backoff():
    from datetime import datetime, timedelta, timezone

    from web.trading_runner import TradingRunner

    tr = TradingRunner(lambda: None)
    now = datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)
    st = {"enabled": True, "last_cycle": (now - timedelta(seconds=61)).isoformat(), "interval_minutes": 1}
    tr._last_attempt = now - timedelta(seconds=61)  # a successful cycle a minute ago doesn't hold the next one back
    assert tr.due(st, now)
    assert not tr.due({**st, "last_cycle": (now - timedelta(seconds=30)).isoformat()}, now)
    tr._last_failed = True  # after a failure, wait FAILURE_BACKOFF before retrying
    assert not tr.due(st, now)
    assert tr.due(st, now + timedelta(minutes=5))


def test_ig_session_is_shared_between_processes(tmp_path, monkeypatch):
    import os
    import time

    import trading.ig as igmod
    from trading.ig import IGClient

    monkeypatch.setattr(igmod, "MIN_INTERVAL", 0)
    sf = tmp_path / "ig_session.json"

    def logins(h):
        return sum(1 for c in h.calls if c[0] == "POST")

    h1 = FakeHTTP()
    IGClient("key", "user", "pw", session=h1, session_file=sf).market("CC.D.GAS.IP")
    assert logins(h1) == 1 and sf.exists()
    assert oct(os.stat(sf).st_mode & 0o777) == "0o600"  # tokens readable by the owner only

    h2 = FakeHTTP()  # a later cycle (new process) reuses the saved session: no new login
    IGClient("key", "user", "pw", session=h2, session_file=sf).market("CC.D.GAS.IP")
    assert logins(h2) == 0

    h3 = FakeHTTP()  # different credentials never pick up someone else's session
    IGClient("key2", "user", "pw", session=h3, session_file=sf).market("CC.D.GAS.IP")
    assert logins(h3) == 1

    data = json.loads(sf.read_text())
    data["login_at"] = time.time() - igmod.SESSION_MAX_AGE - 1  # stale session: log in again
    sf.write_text(json.dumps(data))
    h4 = FakeHTTP()
    IGClient("key2", "user", "pw", session=h4, session_file=sf).market("CC.D.GAS.IP")
    assert logins(h4) == 1


def test_strategy_step_fetches_ig_prices_first(tmp_path, monkeypatch):
    import trading.ig as igmod
    from pipeline import nightly

    settings, store = make_env(tmp_path)
    monkeypatch.setattr(igmod, "ig_configured", lambda: True)
    calls = []

    class FakeSource:
        def update(self):
            calls.append(1)
            store.write("ig_gas", "GB", synthetic_gas_candles(days=150))
            from sources import UpdateResult
            return UpdateResult("ig_gas", "GB", status="ok", rows=1)

    import sources

    monkeypatch.setattr(sources, "build_source", lambda *a, **k: FakeSource())
    monkeypatch.setattr(nightly, "run_isolated", lambda fn, *a, **k: {"status": "ok"})
    assert nightly.run_strategy(settings)["status"] == "ok" and calls == [1]


def test_ig_prices_one_request_per_run_and_single_download(tmp_path, monkeypatch):
    import trading.ig as igmod
    from sources import build_source
    from trading.ig import IGClient
    from trading.market import trading_dir

    monkeypatch.setattr(igmod, "MIN_INTERVAL", 0)
    settings, store = make_env(tmp_path, trading={"epic": "CC.D.GAS.IP", "history_points": 500})
    http = FakeHTTP()
    src = build_source("ig_gas", store=store, settings=settings)
    src._client = IGClient("key", "user", "pw", session=http)
    res = src.update()
    price_calls = [c for c in http.calls if "/prices/" in c[1]]
    assert res.status == "ok" and res.rows == 1 and len(price_calls) == 1  # whole history in one request
    assert store.last_timestamp("ig_gas", "GB") == pd.Timestamp("2026-10-08T09:00", tz="UTC")
    src.update()                                                           # incremental: one more
    assert len([c for c in http.calls if "/prices/" in c[1]]) == 2
    # while another process holds the download lock, it skips instead of calling IG
    import fcntl

    with open(trading_dir(settings) / "ig_download.lock", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        other = build_source("ig_gas", store=store, settings=settings)
        other._client = src._client
        assert other.update().status == "skipped"
    assert len([c for c in http.calls if "/prices/" in c[1]]) == 2


def test_funding_cutoffs_across_clock_changes():
    from trading.strategy import nights_between

    tz = "Europe/London"
    # 25 Oct 2026, clocks went back at 02:00: the 22:00 cut-off is 22:00 GMT (22:00 UTC)
    assert nights_between("2026-10-25T21:15:00+00:00", "2026-10-25T22:15:00+00:00", tz) == 1
    assert nights_between("2026-10-25T21:15:00+00:00", "2026-10-25T21:45:00+00:00", tz) == 0
    # summer: 22:00 BST is 21:00 UTC
    assert nights_between("2026-07-01T20:30:00+00:00", "2026-07-01T21:30:00+00:00", tz) == 1
    # a week spanning the change counts each night once
    assert nights_between("2026-10-21T12:00:00+00:00", "2026-10-28T12:00:00+00:00", tz) == 7
    assert nights_between(None, "2026-10-28T12:00:00+00:00", tz) == 0
