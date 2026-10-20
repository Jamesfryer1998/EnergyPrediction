"""Gas CFD strategy: predict the mid-price move over the next few hours from hourly IG quotes
(plus slow GB fundamentals), trade only when the predicted move beats the spread, and test it
walk-forward with realistic fills (buy at the ask, sell at the bid, overnight funding).

Everything is paper trading: nothing here sends orders (see trading.paper for the demo mirror).
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from settings import Settings
from storage import TS, ParquetStore
from .market import load_market, trading_dir, value_per_point

log = logging.getLogger(__name__)

SOURCE = "ig_gas"
TARGET = "y_move"


# ── config ──────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    horizon: int = 6
    model: str = "hgb"
    train_days: int = 240
    retrain_every_days: int = 7
    backtest_days: int = 90
    threshold_k: float = 1.5
    size: float = 1.0
    starting_balance: float = 10_000.0
    funding_rate: float = 0.065
    value_per_point: float = 1.0
    timezone: str = "Europe/London"

    @classmethod
    def from_settings(cls, settings: Settings, **overrides) -> StrategyConfig:
        t = settings.get("trading", {}) or {}
        cfg = cls(horizon=int(t.get("horizon_hours", 6)), model=str(t.get("model", "hgb")),
                  train_days=int(t.get("train_days", 240)), retrain_every_days=int(t.get("retrain_every_days", 7)),
                  backtest_days=int(t.get("backtest_days", 90)), threshold_k=float(t.get("threshold_k", 1.5)),
                  size=float(t.get("size", 1)), starting_balance=float(t.get("starting_balance", 10_000)),
                  funding_rate=float(t.get("funding_rate", 0.065)),
                  value_per_point=value_per_point(settings, load_market(settings)), timezone=settings.timezone)
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg


# ── data and features ───────────────────────────────────────────────────────

def load_candles(store: ParquetStore, region: str = "GB", start=None, end=None) -> pd.DataFrame:
    df = store.read(SOURCE, region, start, end)
    if df.empty:
        return df
    df = df.drop_duplicates(TS, keep="last").set_index(TS).sort_index()
    return df[df["bid_close"].notna() & df["ask_close"].notna()]


def _daily_asof(store, region, name, col, index, lag_days=2):
    """A daily series as known at each candle: lagged `lag_days` (publication delay), then as-of."""
    df = store.read(name, region, index.min() - pd.Timedelta(days=60), index.max() + pd.Timedelta(days=1),
                    columns=[TS, col])
    if df.empty or col not in df:
        return pd.Series(np.nan, index=index)
    s = df.drop_duplicates(TS, keep="last").set_index(TS)[col].sort_index().dropna()
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def build_features(candles: pd.DataFrame, store: ParquetStore | None, cfg: StrategyConfig,
                   region: str = "GB") -> pd.DataFrame:
    """One row per candle; every feature is known when that candle closes. TARGET is the mid move
    over the next `horizon` candles (NaN for the latest rows, which is what gets predicted live)."""
    c = candles
    mid = (c["bid_close"] + c["ask_close"]) / 2
    hi = (c["bid_high"] + c["ask_high"]) / 2 if "bid_high" in c else mid
    lo = (c["bid_low"] + c["ask_low"]) / 2 if "bid_low" in c else mid
    d1 = mid.diff()
    f = pd.DataFrame(index=c.index)
    f["mid"] = mid
    for n in (1, 3, 6, 12, 24, 72, 120):
        f[f"ret_{n}"] = mid - mid.shift(n)
    for n in (24, 120):
        m, s = mid.rolling(n, min_periods=n // 2).mean(), mid.rolling(n, min_periods=n // 2).std()
        f[f"z_{n}"] = (mid - m) / s.replace(0, np.nan)
    f["vol_24"] = d1.rolling(24, min_periods=12).std()
    f["vol_120"] = d1.rolling(120, min_periods=60).std()
    f["range_24"] = hi.rolling(24, min_periods=12).max() - lo.rolling(24, min_periods=12).min()
    f["spread"] = c["ask_close"] - c["bid_close"]
    f["spread_med"] = f["spread"].rolling(120, min_periods=1).median()
    local = c.index.tz_convert(cfg.timezone)
    f["hour"] = local.hour
    f["dow"] = local.dayofweek
    f["gap_hours"] = pd.Series(c.index, index=c.index).diff().dt.total_seconds().div(3600).fillna(1)
    if "volume" in c:
        f["volume_rel"] = c["volume"] / c["volume"].rolling(120, min_periods=10).mean()
    if store is not None:  # slow fundamentals, as known at the time
        sap = _daily_asof(store, region, "gas_prices", "gas_sap_ptherm", c.index)
        f["sap_ptherm"] = sap
        f["basis_sap"] = mid - sap
        f["sap_chg_7d"] = sap - sap.shift(7 * 24)
        f["eua_gbp"] = _daily_asof(store, region, "carbon_allowances", "eua_auction_gbp", c.index)
        wx = store.read("weather", region, c.index.min() - pd.Timedelta(days=40), c.index.max() + pd.Timedelta(hours=1),
                        columns=[TS, "wx_mean__temperature_2m", "wx_mean__wind_speed_10m"])
        if not wx.empty:
            wx = wx.drop_duplicates(TS, keep="last").set_index(TS).sort_index()
            wx = wx[wx.index <= c.index.max()]  # actuals only (no forecasts) to avoid look-ahead
            t24 = wx["wx_mean__temperature_2m"].rolling("24h").mean()
            f["temp_24h"] = t24.reindex(t24.index.union(c.index)).ffill().reindex(c.index)
            f["temp_anom"] = f["temp_24h"] - wx["wx_mean__temperature_2m"].rolling("30D").mean().reindex(
                t24.index.union(c.index)).ffill().reindex(c.index)
            w24 = wx["wx_mean__wind_speed_10m"].rolling("24h").mean()
            f["wind_24h"] = w24.reindex(w24.index.union(c.index)).ffill().reindex(c.index)
    f[TARGET] = mid.shift(-cfg.horizon) - mid
    return f


def feature_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in (TARGET, "mid")]


# ── model ───────────────────────────────────────────────────────────────────

def fit(f: pd.DataFrame, cfg: StrategyConfig):
    from models import create_model

    train = f.dropna(subset=[TARGET])
    if len(train) < 200:
        raise ValueError(f"only {len(train)} usable hourly rows to train on; need at least 200")
    X, y = train[feature_columns(f)], train[TARGET]
    model = create_model(cfg.model)
    if model.self_validating:
        model.fit(X, y)
    else:
        cut = int(len(X) * 0.9)
        model.fit(X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:])
    return model


def decide(pred: float, position: float, threshold: float, size: float) -> float:
    """Target position. Enter (or flip) only when the predicted move beats the threshold; with a
    weaker signal keep a position that agrees with it, otherwise go flat."""
    if pred is None or not math.isfinite(pred):
        return position
    if pred > threshold:
        return size
    if pred < -threshold:
        return -size
    if (position > 0 and pred > 0) or (position < 0 and pred < 0):
        return position
    return 0.0


# ── simulation ──────────────────────────────────────────────────────────────

class Book:
    """Position accounting shared by the backtest and the live paper trader."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.position = 0.0
        self.avg_price = 0.0
        self.realised = 0.0
        self.funding = 0.0
        self.spread_cost = 0.0
        self.fills: list[dict] = []
        self.closed: list[float] = []  # P&L of each reduction (for win rate)

    def trade(self, ts, delta: float, bid: float, ask: float, reason: str = "") -> dict | None:
        if abs(delta) < 1e-12:
            return None
        price = ask if delta > 0 else bid
        mid = (bid + ask) / 2
        vpp = self.cfg.value_per_point
        self.spread_cost += abs(price - mid) * abs(delta) * vpp
        pos, realised = self.position, 0.0
        if pos == 0 or np.sign(pos) == np.sign(delta):  # open / add
            new = pos + delta
            self.avg_price = (self.avg_price * abs(pos) + price * abs(delta)) / abs(new)
            self.position = new
        else:  # reduce, close or flip
            closing = min(abs(delta), abs(pos))
            realised = (price - self.avg_price) * closing * np.sign(pos) * vpp
            self.realised += realised
            self.closed.append(realised)
            new = pos + delta
            if abs(new) < 1e-12:
                self.position, self.avg_price = 0.0, 0.0
            elif np.sign(new) == np.sign(pos):
                self.position = new
            else:  # flipped: remainder opens at this price
                self.position, self.avg_price = new, price
        fill = {"ts": pd.Timestamp(ts).isoformat(), "side": "BUY" if delta > 0 else "SELL", "size": abs(delta),
                "price": price, "position": self.position, "realised": realised, "reason": reason}
        self.fills.append(fill)
        return fill

    def unrealised(self, bid: float, ask: float) -> float:
        if self.position == 0:
            return 0.0
        mark = bid if self.position > 0 else ask  # what closing would get
        return (mark - self.avg_price) * self.position * self.cfg.value_per_point

    def charge_funding(self, nights: int, mid: float) -> float:
        if self.position == 0 or nights <= 0:
            return 0.0
        cost = float(abs(self.position) * mid * self.cfg.value_per_point * self.cfg.funding_rate / 365 * nights)
        self.funding += cost
        return cost

    def equity(self, bid: float, ask: float) -> float:
        return self.cfg.starting_balance + self.realised - self.funding + self.unrealised(bid, ask)


def nights_between(prev_ts, ts, tz: str) -> int:
    """How many 22:00 (UK) funding cut-offs fall in (prev_ts, ts]."""
    if prev_ts is None:
        return 0
    a, b = pd.Timestamp(prev_ts).tz_convert(tz), pd.Timestamp(ts).tz_convert(tz)
    first = a.normalize() + pd.Timedelta(hours=22)
    if first <= a:
        first += pd.Timedelta(days=1)
    if first > b:
        return 0
    return int((b - first) / pd.Timedelta(days=1)) + 1


def simulate(f: pd.DataFrame, candles: pd.DataFrame, preds: pd.Series, cfg: StrategyConfig) -> dict:
    """Decide at each candle close, fill at the next candle's open (ask to buy, bid to sell)."""
    book = Book(cfg)
    rows, idx = [], list(preds.index)
    prev_ts = None
    for i, ts in enumerate(idx):
        c = candles.loc[ts]
        book.charge_funding(nights_between(prev_ts, ts, cfg.timezone), (c["bid_close"] + c["ask_close"]) / 2)
        prev_ts = ts
        pred = float(preds.loc[ts])
        thr = cfg.threshold_k * float(f.at[ts, "spread_med"])
        target = decide(pred, book.position, thr, cfg.size)
        if i + 1 < len(idx) and target != book.position:
            nxt = candles.loc[idx[i + 1]]
            bid = nxt.get("bid_open") if pd.notna(nxt.get("bid_open")) else nxt["bid_close"]
            ask = nxt.get("ask_open") if pd.notna(nxt.get("ask_open")) else nxt["ask_close"]
            book.trade(idx[i + 1], target - book.position, float(bid), float(ask), f"pred {pred:+.2f} vs ±{thr:.2f}")
        rows.append({"ts": ts.isoformat(), "mid": float(f.at[ts, "mid"]), "pred": pred,
                     "actual": None if pd.isna(f.at[ts, TARGET]) else float(f.at[ts, TARGET]),
                     "position": book.position, "equity": book.equity(float(c["bid_close"]), float(c["ask_close"]))})
    series = pd.DataFrame(rows)
    return {"series": series, "fills": book.fills, "book": book}


def summarise(sim: dict, cfg: StrategyConfig, candles: pd.DataFrame) -> dict:
    s, book = sim["series"], sim["book"]
    if s.empty:
        return {}
    eq = s.set_index(pd.to_datetime(s["ts"]))["equity"]
    daily = eq.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if len(rets) > 2 and rets.std() > 0 else None
    dd = float((eq - eq.cummax()).min())
    pnl = float(eq.iloc[-1] - cfg.starting_balance)
    ok = s.dropna(subset=["actual"])
    hit = float((np.sign(ok["pred"]) == np.sign(ok["actual"])).mean()) if len(ok) else None
    mae = float((ok["pred"] - ok["actual"]).abs().mean()) if len(ok) else None
    mae0 = float(ok["actual"].abs().mean()) if len(ok) else None
    # buy and hold over the same window, with the same costs
    first, last = candles.loc[pd.Timestamp(s["ts"].iloc[0])], candles.loc[pd.Timestamp(s["ts"].iloc[-1])]
    bh = Book(cfg)
    bh.trade(s["ts"].iloc[0], cfg.size, float(first["bid_close"]), float(first["ask_close"]))
    bh.charge_funding(nights_between(s["ts"].iloc[0], s["ts"].iloc[-1], cfg.timezone),
                      float((first["bid_close"] + first["ask_close"]) / 2))
    bh_pnl = bh.equity(float(last["bid_close"]), float(last["ask_close"])) - cfg.starting_balance
    wins = [x for x in book.closed if x > 0]
    return {
        "pnl": pnl, "return_pct": 100 * pnl / cfg.starting_balance, "sharpe": sharpe, "max_drawdown": dd,
        "trades": len(book.fills), "round_trips": len(book.closed),
        "win_rate": len(wins) / len(book.closed) if book.closed else None,
        "spread_cost": book.spread_cost, "funding": book.funding,
        "exposure": float((s["position"] != 0).mean()),
        "hit_rate": hit, "mae": mae, "mae_no_change": mae0, "buy_hold_pnl": float(bh_pnl),
        "start": s["ts"].iloc[0], "end": s["ts"].iloc[-1], "candles": len(s),
    }


# ── walk-forward backtest ───────────────────────────────────────────────────

def backtest(settings: Settings, store: ParquetStore, region: str = "GB", progress=None, **overrides) -> dict:
    cfg = StrategyConfig.from_settings(settings, **overrides)
    candles = load_candles(store, region)
    if candles.empty:
        raise RuntimeError("no IG price history yet: press BACKTEST + TRAIN on the Trading tab (it downloads the history first)")
    end = candles.index.max()
    span = (end - candles.index.min()) / pd.Timedelta(days=1)
    if span < cfg.backtest_days + 30:
        # not enough history yet for the full window: test on the last ~40%, train on the rest
        days = max(7, int(span * 0.4))
        log.warning("only %.0f days of IG history: backtesting the last %d days instead of %d", span, days,
                    cfg.backtest_days)
        cfg.backtest_days = days
    test_start = end - pd.Timedelta(days=cfg.backtest_days)
    f = build_features(candles, store, cfg, region)
    test_idx = f.index[f.index >= test_start]
    if len(test_idx) < 10:
        raise RuntimeError(f"not enough candles in the last {cfg.backtest_days} days to backtest")
    preds = pd.Series(np.nan, index=test_idx)
    cut = test_idx[0]
    steps = max(1, math.ceil((end - cut) / pd.Timedelta(days=cfg.retrain_every_days)))
    k = 0
    while cut <= end:
        nxt = cut + pd.Timedelta(days=cfg.retrain_every_days)
        hist = f[(f.index < cut) & (f.index >= cut - pd.Timedelta(days=cfg.train_days))]
        # the target needs `horizon` future candles: drop rows whose outcome wasn't known at `cut`
        hist = hist.iloc[: max(0, len(hist) - cfg.horizon)]
        model = fit(hist, cfg)
        block = f[(f.index >= cut) & (f.index < nxt)]
        if len(block):
            preds.loc[block.index] = model.predict(block[feature_columns(f)])
        k += 1
        if progress:
            progress(k, steps, f"walk-forward {k}/{steps}")
        cut = nxt
    sim = simulate(f, candles, preds, cfg)
    summary = summarise(sim, cfg, candles)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_strategy"
    out = {"run_id": run_id, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "config": cfg.__dict__, "summary": summary, "fills": sim["fills"][-500:],
           "series": sim["series"].to_dict(orient="records")}
    d = trading_dir(settings) / "backtests"
    d.mkdir(exist_ok=True)
    (d / f"{run_id}.json").write_text(json.dumps(out, default=_json_default))
    log.info("strategy backtest %s: P&L £%.0f (%.1f%%), %d trades, hit rate %s, buy & hold £%.0f",
             run_id, summary["pnl"], summary["return_pct"], summary["trades"],
             f"{summary['hit_rate']:.0%}" if summary.get("hit_rate") is not None else "n/a", summary["buy_hold_pnl"])
    return {"run_id": run_id, "summary": summary}


def list_backtests(settings: Settings, limit: int = 20) -> list[dict]:
    d = trading_dir(settings) / "backtests"
    out = []
    for p in sorted(d.glob("*.json"), reverse=True)[:limit] if d.exists() else []:
        try:
            b = json.loads(p.read_text())
            out.append({"run_id": b["run_id"], "created": b["created"], "summary": b["summary"], "config": b["config"]})
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def load_backtest(settings: Settings, run_id: str) -> dict | None:
    p = trading_dir(settings) / "backtests" / f"{run_id}.json"
    if "/" in run_id or not p.exists():
        return None
    return json.loads(p.read_text())


# ── production model ────────────────────────────────────────────────────────

def model_path(settings: Settings):
    return trading_dir(settings) / "model.pkl"


def train(settings: Settings, store: ParquetStore, region: str = "GB") -> dict:
    cfg = StrategyConfig.from_settings(settings)
    candles = load_candles(store, region)
    if candles.empty:
        raise RuntimeError("no IG price history yet: press BACKTEST + TRAIN on the Trading tab (it downloads the history first)")
    f = build_features(candles, store, cfg, region)
    end = f.index.max()
    hist = f[f.index >= end - pd.Timedelta(days=cfg.train_days)]
    model = fit(hist, cfg)
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": int(hist[TARGET].notna().sum()),
            "model": cfg.model, "horizon": cfg.horizon, "data_to": end.isoformat()}
    with open(model_path(settings), "wb") as fh:
        pickle.dump({"model": model, "meta": meta, "features": feature_columns(f)}, fh)
    log.info("strategy model trained: %s, %d rows to %s", cfg.model, meta["rows"], meta["data_to"])
    return meta


def predict_latest(settings: Settings, store: ParquetStore, region: str = "GB") -> dict:
    """Prediction for the latest closed candle, with the threshold the trader compares it to."""
    cfg = StrategyConfig.from_settings(settings)
    p = model_path(settings)
    if not p.exists():
        train(settings, store, region)
    with open(p, "rb") as fh:
        saved = pickle.load(fh)
    candles = load_candles(store, region, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=60))
    if candles.empty:
        raise RuntimeError("no recent IG candles")
    f = build_features(candles, store, cfg, region)
    last = f.iloc[[-1]]
    pred = float(saved["model"].predict(last[saved["features"]])[0])
    return {"candle": last.index[0].isoformat(), "pred": pred, "mid": float(last["mid"].iloc[0]),
            "threshold": cfg.threshold_k * float(last["spread_med"].iloc[0]),
            "spread_med": float(last["spread_med"].iloc[0]),
            "bid": float(candles["bid_close"].iloc[-1]), "ask": float(candles["ask_close"].iloc[-1]),
            "model": saved["meta"]["model"], "horizon": saved["meta"]["horizon"],
            "model_trained": saved["meta"]["trained_at"]}


def _json_default(o):
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)
