"""Live paper trading: one cycle = refresh quotes, predict, decide, fill on the paper book,
optionally mirror the same trade on the IG DEMO account, and record everything in SQLite.

The paper book (data/trading/paper.db) is the record of truth; IG demo fills are stored alongside
it so the two can be compared (slippage, rejections).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pandas as pd

from settings import Settings
from storage import ParquetStore
from .market import load_market, resolve_market, trading_dir
from .strategy import Book, StrategyConfig, decide, nights_between, predict_latest

log = logging.getLogger(__name__)

STATE_DEFAULT = {"enabled": False, "mirror_ig": False, "interval_minutes": 60, "size": None, "threshold_k": None,
                 "last_cycle": None, "last_result": None}

SCHEMA = """
CREATE TABLE IF NOT EXISTS book (id INTEGER PRIMARY KEY CHECK (id = 1), position REAL, avg_price REAL,
    realised REAL, funding REAL, spread_cost REAL, last_mark TEXT, started TEXT);
CREATE TABLE IF NOT EXISTS fills (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, side TEXT, size REAL, price REAL,
    position REAL, realised REAL, reason TEXT, ig_status TEXT, ig_deal_id TEXT, ig_level REAL, ig_reason TEXT);
CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, candle TEXT, pred REAL,
    threshold REAL, position REAL, target REAL, bid REAL, ask REAL, action TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, mid REAL, position REAL, realised REAL, funding REAL,
    unrealised REAL, equity REAL);
"""


# ── runtime switches (Trading tab) ──────────────────────────────────────────

def state_path(settings: Settings):
    return trading_dir(settings) / "state.json"


def load_state(settings: Settings) -> dict:
    try:
        return {**STATE_DEFAULT, **json.loads(state_path(settings).read_text())}
    except (OSError, json.JSONDecodeError):
        return dict(STATE_DEFAULT)


def save_state(settings: Settings, state: dict) -> dict:
    state = {**STATE_DEFAULT, **state}
    state_path(settings).write_text(json.dumps(state, indent=2, default=str))
    return state


def update_state(settings: Settings, changes: dict) -> dict:
    st = load_state(settings)
    if "enabled" in changes:
        st["enabled"] = bool(changes["enabled"])
    if "mirror_ig" in changes:
        st["mirror_ig"] = bool(changes["mirror_ig"])
    if "interval_minutes" in changes:
        v = int(changes["interval_minutes"])
        if not 1 <= v <= 1440:
            raise ValueError("interval must be 1-1440 minutes")
        st["interval_minutes"] = v
    for key, lo, hi in (("size", 0.01, 1000), ("threshold_k", 0.0, 50)):
        if key in changes:
            v = changes[key]
            if v in (None, ""):
                st[key] = None
            else:
                v = float(v)
                if not lo <= v <= hi:
                    raise ValueError(f"{key} must be between {lo} and {hi}")
                st[key] = v
    return save_state(settings, st)


# ── ledger ──────────────────────────────────────────────────────────────────

class Ledger:
    def __init__(self, settings: Settings):
        self.path = trading_dir(settings) / "paper.db"
        with closing(self._conn()) as c:
            c.executescript(SCHEMA)
            c.commit()

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def load_book(self, cfg: StrategyConfig) -> tuple[Book, str | None]:
        book = Book(cfg)
        with closing(self._conn()) as c:
            row = c.execute("SELECT * FROM book WHERE id = 1").fetchone()
        if row:
            book.position, book.avg_price = row["position"], row["avg_price"]
            book.realised, book.funding, book.spread_cost = row["realised"], row["funding"], row["spread_cost"]
            return book, row["last_mark"]
        return book, None

    def save_book(self, book: Book, last_mark: str) -> None:
        with closing(self._conn()) as c:
            c.execute("INSERT INTO book (id, position, avg_price, realised, funding, spread_cost, last_mark, started) "
                      "VALUES (1, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET position=excluded.position, "
                      "avg_price=excluded.avg_price, realised=excluded.realised, funding=excluded.funding, "
                      "spread_cost=excluded.spread_cost, last_mark=excluded.last_mark",
                      (book.position, book.avg_price, book.realised, book.funding, book.spread_cost, last_mark, last_mark))
            c.commit()

    def add_fill(self, fill: dict, ig: dict | None) -> None:
        ig = ig or {}
        with closing(self._conn()) as c:
            c.execute("INSERT INTO fills (ts, side, size, price, position, realised, reason, ig_status, ig_deal_id, "
                      "ig_level, ig_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (fill["ts"], fill["side"], fill["size"], fill["price"], fill["position"], fill["realised"],
                       fill["reason"], ig.get("status"), ig.get("deal_id"), ig.get("level"), ig.get("reason")))
            c.commit()

    def add_decision(self, d: dict) -> None:
        with closing(self._conn()) as c:
            c.execute("INSERT INTO decisions (ts, candle, pred, threshold, position, target, bid, ask, action, note) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?)",
                      tuple(d.get(k) for k in ("ts", "candle", "pred", "threshold", "position", "target", "bid", "ask",
                                               "action", "note")))
            c.commit()

    def add_equity(self, ts: str, mid: float, book: Book, bid: float, ask: float) -> None:
        with closing(self._conn()) as c:
            c.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?,?)",
                      (ts, mid, book.position, book.realised, book.funding, book.unrealised(bid, ask),
                       book.equity(bid, ask)))
            c.commit()

    def rows(self, table: str, limit: int = 200) -> list[dict]:
        if table not in ("fills", "decisions", "equity"):
            raise ValueError(table)
        order = "ts" if table == "equity" else "id"
        with closing(self._conn()) as c:
            rows = c.execute(f"SELECT * FROM {table} ORDER BY {order} DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows][::-1]

    def reset(self) -> None:
        with closing(self._conn()) as c:
            c.executescript("DELETE FROM book; DELETE FROM fills; DELETE FROM decisions; DELETE FROM equity;")
            c.commit()


# ── one trading cycle ───────────────────────────────────────────────────────

def run_cycle(settings: Settings, region: str = "GB", client=None, force_target: float | None = None,
              refresh: bool = True, reason: str = "") -> dict:
    """Refresh data, predict, trade the paper book (and mirror to the IG demo if switched on)."""
    from .ig import IGError, ig_configured

    store = ParquetStore(settings.data_dir)
    state = load_state(settings)
    cfg = StrategyConfig.from_settings(settings, size=state.get("size"), threshold_k=state.get("threshold_k"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    live = client is not None or ig_configured()
    if live and client is None:
        from .ig import IGClient

        client = IGClient(session_file=trading_dir(settings) / "ig_session.json")
    note = []

    # 1. fresh candles + a live quote (or the last candle when there's no IG login, e.g. demo data)
    if refresh and live:
        from sources import build_source

        src = build_source("ig_gas", store=store, settings=settings)
        src._client = client
        res = src.update()
        if res.status not in ("ok", "skipped", "not_implemented"):
            note.append(f"candle refresh: {res.message}")
    market = resolve_market(settings, client) if live else (load_market(settings) or {})
    cfg.value_per_point = StrategyConfig.from_settings(settings).value_per_point
    signal = predict_latest(settings, store, region) if force_target is None else None
    if live:
        q = client.quote(market["epic"])
        bid, ask, tradeable = q["bid"], q["offer"], q["status"] == "TRADEABLE"
    else:
        if signal is None:
            from .strategy import load_candles

            last = load_candles(store, region).iloc[-1]
            bid, ask = float(last["bid_close"]), float(last["ask_close"])
        else:
            bid, ask = signal["bid"], signal["ask"]
        tradeable = True
        note.append("no IG login: using the last candle as the quote")
    if bid is None or ask is None:
        raise IGError("IG returned no bid/offer for the market")
    mid = (bid + ask) / 2

    ledger = Ledger(settings)
    book, last_mark = ledger.load_book(cfg)
    book.charge_funding(nights_between(last_mark, now, cfg.timezone) if last_mark else 0, mid)

    pos_before = book.position
    if force_target is not None:
        target, pred, thr = force_target, None, None
    else:
        pred, thr = signal["pred"], cfg.threshold_k * signal["spread_med"]
        target = decide(pred, book.position, thr, cfg.size)
    action, fill, ig_res = "hold", None, None
    if not tradeable:
        action, target = "market closed", book.position
    elif abs(target - book.position) > 1e-9:
        delta = target - book.position
        why = reason or (f"predicted {pred:+.2f} over {cfg.horizon}h vs threshold ±{thr:.2f}" if pred is not None else "")
        fill = book.trade(now, delta, bid, ask, why)
        action = f"{fill['side'].lower()} {fill['size']:g} @ {fill['price']:.2f}"
        if state.get("mirror_ig") and live:
            try:
                ig_res = client.place_market_order(market["epic"], "BUY" if delta > 0 else "SELL", abs(delta),
                                                   market.get("expiry") or "-", market.get("currency") or "GBP")
                if ig_res.get("status") != "ACCEPTED":
                    note.append(f"IG demo order {ig_res.get('status')}: {ig_res.get('reason')}")
                else:
                    ig_pos = client.net_position(market["epic"])
                    if abs(ig_pos - book.position) > 1e-9:
                        note.append(f"IG demo position {ig_pos:g} differs from paper {book.position:g}")
            except IGError as exc:
                ig_res = {"status": "ERROR", "reason": str(exc)}
                note.append(f"IG demo order failed: {exc}")
        elif state.get("mirror_ig"):
            note.append("mirror to IG is on but there's no IG login in .env")
        ledger.add_fill(fill, ig_res)
    ledger.add_decision({"ts": now, "candle": (signal or {}).get("candle"), "pred": pred, "threshold": thr,
                         "position": pos_before, "target": target, "bid": bid, "ask": ask, "action": action,
                         "note": "; ".join(note) or None})
    ledger.save_book(book, now)
    ledger.add_equity(now, mid, book, bid, ask)
    result = {"ts": now, "action": action, "pred": pred, "threshold": thr, "position": book.position,
              "bid": bid, "ask": ask, "equity": float(book.equity(bid, ask)), "notes": note,
              "ig": ig_res, "epic": market.get("epic")}
    log.info("paper trade cycle: %s (pred %s, threshold %s, position %g, equity £%.2f)%s", action,
             "n/a" if pred is None else f"{pred:+.2f}", "n/a" if thr is None else f"±{thr:.2f}",
             book.position, result["equity"], f" — {'; '.join(note)}" if note else "")
    st = load_state(settings)
    st["last_cycle"], st["last_result"] = now, {k: result[k] for k in ("action", "pred", "position", "equity")}
    save_state(settings, st)
    return result


def status(settings: Settings) -> dict:
    """Snapshot for the dashboard (no network calls)."""
    from .ig import ig_configured
    from .strategy import model_path

    state = load_state(settings)
    cfg = StrategyConfig.from_settings(settings, size=state.get("size"), threshold_k=state.get("threshold_k"))
    ledger = Ledger(settings)
    book, last_mark = ledger.load_book(cfg)
    eq = ledger.rows("equity", 1)
    meta = None
    p = model_path(settings)
    if p.exists():
        import pickle

        try:
            with open(p, "rb") as fh:
                meta = pickle.load(fh)["meta"]
        except Exception:
            meta = None
    last = eq[-1] if eq else None
    return {
        "state": state, "ig_configured": ig_configured(), "market": load_market(settings),
        "position": book.position, "avg_price": book.avg_price, "realised": book.realised, "funding": book.funding,
        "spread_cost": book.spread_cost, "last_mark": last_mark,
        "equity": last["equity"] if last else cfg.starting_balance, "unrealised": last["unrealised"] if last else 0.0,
        "mid": last["mid"] if last else None, "starting_balance": cfg.starting_balance,
        "config": {"horizon": cfg.horizon, "model": cfg.model, "size": cfg.size, "threshold_k": cfg.threshold_k,
                   "value_per_point": cfg.value_per_point, "funding_rate": cfg.funding_rate},
        "model": meta,
    }


def candles_for_chart(settings: Settings, region: str = "GB", days: int = 7) -> pd.DataFrame:
    from .strategy import load_candles

    store = ParquetStore(settings.data_dir)
    c = load_candles(store, region, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days))
    return c
